use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::body::Body;
use axum::http::{Request, Response, header};
use http_body_util::BodyExt;
use openai_frontend::OpenAiBackend;
use serde_json::Value;
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
        openai_frontend::router_for(backend)
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
}

impl SseDecoder {
    fn pop(&mut self) -> Option<EmbeddedSseEvent> {
        self.ready.pop_front()
    }

    fn push(&mut self, input: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(input);
        while let Some(end) = sse_frame_end(&self.buffer) {
            let raw = self.buffer.drain(..end).collect::<Vec<_>>();
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
        let raw = std::mem::take(&mut self.buffer);
        if let Some(event) = parse_sse_event(raw)? {
            self.ready.push_back(event);
        }
        Ok(())
    }
}

fn sse_frame_end(bytes: &[u8]) -> Option<usize> {
    let mut line_start = 0;
    let mut index = 0;
    while index < bytes.len() {
        let line_end = match bytes[index] {
            b'\r' => {
                let end = if bytes.get(index + 1) == Some(&b'\n') {
                    index + 2
                } else {
                    index + 1
                };
                if index == line_start {
                    return Some(end);
                }
                end
            }
            b'\n' => {
                let end = index + 1;
                if index == line_start {
                    return Some(end);
                }
                end
            }
            _ => {
                index += 1;
                continue;
            }
        };
        line_start = line_end;
        index = line_end;
    }
    None
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
    use super::{SseDecoder, request_model};

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
}
