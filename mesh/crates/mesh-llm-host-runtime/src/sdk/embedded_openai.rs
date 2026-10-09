use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::body::Body;
use axum::http::{Request, Response, header};
use http_body_util::BodyExt;
use serde_json::Value;
use skippy_inference_api::OpenAiBackend;
use tower::ServiceExt;

use super::EmbeddedServingController;

const MAX_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;

pub struct EmbeddedOpenAiResponse {
    pub status_code: u16,
    pub content_type: Option<String>,
    pub body: String,
}

pub struct EmbeddedSseEvent {
    pub event_type: Option<String>,
    pub data: String,
    pub raw: String,
}

pub struct EmbeddedOpenAiStream {
    status_code: u16,
    content_type: Option<String>,
    body: Body,
    decoder: SseDecoder,
}

impl EmbeddedOpenAiStream {
    pub fn status_code(&self) -> u16 {
        self.status_code
    }

    pub fn content_type(&self) -> Option<&str> {
        self.content_type.as_deref()
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status_code)
    }

    pub fn is_event_stream(&self) -> bool {
        self.content_type
            .as_deref()
            .is_some_and(|value| value.to_ascii_lowercase().starts_with("text/event-stream"))
    }

    pub async fn body_text(self) -> Result<String> {
        let body = self
            .body
            .collect()
            .await
            .context("read embedded OpenAI response body")?
            .to_bytes();
        String::from_utf8(body.to_vec()).context("embedded OpenAI response is UTF-8")
    }

    pub async fn next_event(&mut self) -> Result<Option<EmbeddedSseEvent>> {
        loop {
            if let Some(event) = self.decoder.pop() {
                return Ok(Some(event));
            }
            let Some(frame) = self.body.frame().await else {
                self.decoder.finish()?;
                return Ok(self.decoder.pop());
            };
            let frame = frame.context("read embedded OpenAI response body")?;
            if let Ok(data) = frame.into_data() {
                self.decoder.push(&data)?;
            }
        }
    }
}

impl EmbeddedServingController {
    pub async fn handles_openai_request(&self, body_json: &str) -> bool {
        let Ok(model) = request_model(body_json) else {
            return false;
        };
        self.loaded_model(&model).await.is_ok()
    }

    pub async fn openai_request(
        &self,
        path: &str,
        body_json: String,
    ) -> Result<EmbeddedOpenAiResponse> {
        let response = self.openai_response(path, body_json).await?;
        let status_code = response.status().as_u16();
        let content_type = response_content_type(&response);
        let body = response
            .into_body()
            .collect()
            .await
            .context("read embedded OpenAI response")?
            .to_bytes();
        let body = String::from_utf8(body.to_vec()).context("embedded OpenAI response is UTF-8")?;
        Ok(EmbeddedOpenAiResponse {
            status_code,
            content_type,
            body,
        })
    }

    pub async fn openai_stream(
        &self,
        path: &str,
        body_json: String,
    ) -> Result<EmbeddedOpenAiStream> {
        let response = self.openai_response(path, body_json).await?;
        let status_code = response.status().as_u16();
        let content_type = response_content_type(&response);
        Ok(EmbeddedOpenAiStream {
            status_code,
            content_type,
            body: response.into_body(),
            decoder: SseDecoder::default(),
        })
    }

    async fn openai_response(&self, path: &str, body_json: String) -> Result<Response<Body>> {
        let model = request_model(&body_json)?;
        let loaded = self.loaded_model(&model).await?;
        let handle = loaded
            .handle
            .clone()
            .context("model handle not available for embedded OpenAI request")?;
        let backend: Arc<dyn OpenAiBackend> = handle;
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body_json))
            .context("build embedded OpenAI request")?;
        skippy_inference_api::router_for(backend)
            .oneshot(request)
            .await
            .context("route embedded OpenAI request")
    }
}

fn request_model(body_json: &str) -> Result<String> {
    let body: Value = serde_json::from_str(body_json).context("parse embedded OpenAI request")?;
    body.get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .map(ToString::to_string)
        .context("embedded OpenAI request is missing model")
}

fn response_content_type(response: &Response<Body>) -> Option<String> {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string)
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    ready: VecDeque<EmbeddedSseEvent>,
    /// Bytes of `buffer[..scan_offset]` have been examined and contain no blank line.
    scan_offset: usize,
    /// Whether the line that begins at `scan_offset` has content.
    line_has_content: bool,
    /// Bytes examined by the incremental scan; a return to per-push whole-buffer
    /// rescanning would show up as super-linear growth here.
    #[cfg(test)]
    scanned_bytes: usize,
}

impl SseDecoder {
    fn pop(&mut self) -> Option<EmbeddedSseEvent> {
        self.ready.pop_front()
    }

    fn push(&mut self, input: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(input);
        while let Some(end) = self.next_frame_end() {
            if end > MAX_SSE_EVENT_BYTES {
                bail!("embedded OpenAI SSE event exceeds 8 MiB");
            }
            let raw = self.buffer.drain(..end).collect::<Vec<_>>();
            self.scan_offset = 0;
            self.line_has_content = false;
            if let Some(event) = parse_sse_event(raw)? {
                self.ready.push_back(event);
            }
        }
        if self.buffer.len() > MAX_SSE_EVENT_BYTES {
            bail!("embedded OpenAI SSE event exceeds 8 MiB");
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if self.buffer.len() > MAX_SSE_EVENT_BYTES {
            bail!("embedded OpenAI SSE event exceeds 8 MiB");
        }
        let raw = std::mem::take(&mut self.buffer);
        self.scan_offset = 0;
        self.line_has_content = false;
        if let Some(event) = parse_sse_event(raw)? {
            self.ready.push_back(event);
        }
        Ok(())
    }

    /// Returns the end of the first blank line in the buffered bytes, or `None` when
    /// no complete blank line has arrived. The scan continues where the previous call
    /// stopped, so each buffered byte is examined once instead of once per push.
    fn next_frame_end(&mut self) -> Option<usize> {
        let bytes = &self.buffer;
        let mut line_has_content = self.line_has_content;
        let mut index = self.scan_offset;
        while index < bytes.len() {
            #[cfg(test)]
            {
                self.scanned_bytes += 1;
            }
            match bytes[index] {
                b'\r' => {
                    if index + 1 == bytes.len() && line_has_content {
                        // A trailing CR may still pair with the next chunk's LF into a
                        // single line break, which is not a blank line.
                        self.scan_offset = index;
                        self.line_has_content = line_has_content;
                        return None;
                    }
                    let end = if bytes.get(index + 1) == Some(&b'\n') {
                        index + 2
                    } else {
                        index + 1
                    };
                    if !line_has_content {
                        return Some(end);
                    }
                    line_has_content = false;
                    index = end;
                }
                b'\n' => {
                    if !line_has_content {
                        return Some(index + 1);
                    }
                    line_has_content = false;
                    index += 1;
                }
                _ => {
                    line_has_content = true;
                    index += 1;
                }
            }
        }
        self.scan_offset = index;
        self.line_has_content = line_has_content;
        None
    }
}

fn parse_sse_event(raw: Vec<u8>) -> Result<Option<EmbeddedSseEvent>> {
    let raw = String::from_utf8(raw).context("embedded OpenAI SSE is UTF-8")?;
    let mut event_type = None;
    let mut data = Vec::new();
    for line in raw.split(['\r', '\n']) {
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = line.split_once(':').map_or((line, ""), |(field, value)| {
            (field, value.strip_prefix(' ').unwrap_or(value))
        });
        match field {
            "event" => event_type = Some(value.to_string()),
            "data" => data.push(value.to_string()),
            _ => {}
        }
    }
    if data.is_empty() && event_type.is_none() {
        return Ok(None);
    }
    Ok(Some(EmbeddedSseEvent {
        event_type,
        data: data.join("\n"),
        raw,
    }))
}

#[cfg(test)]
mod tests {
    use super::{MAX_SSE_EVENT_BYTES, SseDecoder, request_model};

    #[test]
    fn request_model_requires_a_non_empty_string() {
        assert_eq!(
            request_model(r#"{"model":"local-model"}"#).unwrap(),
            "local-model"
        );
        assert!(request_model(r#"{"model":""}"#).is_err());
        assert!(request_model(r#"{"messages":[]}"#).is_err());
    }

    #[test]
    fn sse_decoder_preserves_split_tool_call_frames() {
        let mut decoder = SseDecoder::default();
        decoder
            .push(b"event: response.function_call_arguments.delta\ndata: {\"delta\":\"{\\\"ci")
            .unwrap();
        assert!(decoder.pop().is_none());
        decoder.push(b"ty\\\":\\\"Sydney\\\"}\"}\n\n").unwrap();

        let event = decoder.pop().expect("complete event");
        assert_eq!(
            event.event_type.as_deref(),
            Some("response.function_call_arguments.delta")
        );
        assert!(event.data.contains("Sydney"));
        assert!(event.raw.ends_with("\n\n"));
    }

    #[test]
    fn sse_decoder_rejects_oversized_complete_frame() {
        let mut decoder = SseDecoder::default();
        let mut frame = b"data: ".to_vec();
        frame.resize(MAX_SSE_EVENT_BYTES, b'x');
        frame.extend_from_slice(b"\n\n");
        assert!(decoder.push(&frame).is_err());
        assert!(decoder.pop().is_none());
    }

    #[test]
    fn sse_decoder_rejects_oversized_frame_at_eof() {
        let mut decoder = SseDecoder::default();
        let mut frame = b"data: ".to_vec();
        frame.resize(MAX_SSE_EVENT_BYTES + 1, b'x');
        decoder.buffer = frame;
        assert!(decoder.finish().is_err());
        assert!(decoder.pop().is_none());
    }

    #[test]
    fn sse_decoder_scans_each_byte_once_across_chunks() {
        // A long unterminated frame used to be rescanned from the buffer start on
        // every push (O(n^2)). Each push must now examine only the bytes it added.
        const CHUNK: usize = 4 * 1024;
        let chunk = vec![b'x'; CHUNK];
        let mut decoder = SseDecoder::default();
        let mut per_push = Vec::new();
        for _ in 0..64 {
            let before = decoder.scanned_bytes;
            decoder.push(&chunk).unwrap();
            per_push.push(decoder.scanned_bytes - before);
        }
        let body = CHUNK * per_push.len();
        assert!(
            per_push.iter().all(|&scanned| scanned <= CHUNK),
            "each push must scan only its own bytes, got {per_push:?}"
        );
        assert!(
            decoder.scanned_bytes <= body,
            "total scanned {} exceeds body {body}",
            decoder.scanned_bytes
        );
    }

    #[test]
    fn sse_decoder_decodes_crlf_split_between_cr_and_lf() {
        let mut decoder = SseDecoder::default();
        decoder.push(b"data: one\r").unwrap();
        assert!(decoder.pop().is_none());
        decoder.push(b"\ndata: two\r\n\r\n").unwrap();
        let event = decoder.pop().expect("complete event");
        assert_eq!(event.data, "one\ntwo");
        assert_eq!(event.raw, "data: one\r\ndata: two\r\n\r\n");
    }
}
