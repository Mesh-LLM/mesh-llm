use super::{
    ALPN_V1, ClientConfig, ClientTransport, STREAM_TUNNEL_HTTP, http_post_request,
    relay_mode_from_endpoint_addr, socket_addr,
};
use crate::events::{OpenAiStreamEvent, OpenAiStreamListener};
use iroh::Endpoint;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Notify;

const MAX_RESPONSE_HEADER_BYTES: usize = 64 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_CHUNK_LINE_BYTES: usize = 4 * 1024;

#[derive(Debug)]
pub(super) struct StreamFailure {
    pub(super) status_code: Option<u16>,
    pub(super) message: String,
    pub(super) body: Option<String>,
    pub(super) cancelled: bool,
}

impl StreamFailure {
    fn transport(message: impl Into<String>) -> Self {
        Self {
            status_code: None,
            message: message.into(),
            body: None,
            cancelled: false,
        }
    }

    fn cancelled() -> Self {
        Self {
            status_code: None,
            message: "cancelled".to_string(),
            body: None,
            cancelled: true,
        }
    }

    fn http(status_code: u16, body: String) -> Self {
        Self {
            status_code: Some(status_code),
            message: format!("HTTP request failed with status {status_code}"),
            body: Some(body),
            cancelled: false,
        }
    }
}

struct ResponseHead {
    status_code: u16,
    content_type: Option<String>,
    chunked: bool,
    content_length: Option<usize>,
}

#[derive(Debug, Eq, PartialEq)]
struct SseEvent {
    event_type: Option<String>,
    data: String,
    raw: String,
}

pub(super) async fn run(
    config: &ClientConfig,
    path: &str,
    body_json: String,
    request_id: &str,
    cancelled: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    listener: Arc<dyn OpenAiStreamListener>,
) -> Result<(), StreamFailure> {
    let request = match &config.transport {
        ClientTransport::DirectMesh => {
            http_post_request(path, "mesh.local", &config.user_agent, body_json)
        }
        ClientTransport::OpenAiHttp { api_base_url } => http_post_request(
            path,
            &socket_addr(api_base_url).map_err(StreamFailure::transport)?,
            &config.user_agent,
            body_json,
        ),
    };

    match &config.transport {
        ClientTransport::DirectMesh => {
            stream_direct_mesh(
                config,
                request,
                request_id,
                cancelled,
                cancel_notify,
                listener,
            )
            .await
        }
        ClientTransport::OpenAiHttp { api_base_url } => {
            stream_http(
                api_base_url,
                request,
                request_id,
                cancelled,
                cancel_notify,
                listener,
            )
            .await
        }
    }
}

async fn stream_direct_mesh(
    config: &ClientConfig,
    request: String,
    request_id: &str,
    cancelled: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    listener: Arc<dyn OpenAiStreamListener>,
) -> Result<(), StreamFailure> {
    let addr = super::decode_invite_endpoint_addr(config.invite_token.as_str())
        .map_err(StreamFailure::transport)?;
    let mut builder = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_V1.to_vec()])
        .bind_addr(std::net::SocketAddr::from(([0, 0, 0, 0], 0)))
        .map_err(|error| StreamFailure::transport(format!("build mesh endpoint: {error}")))?;
    builder = builder.relay_mode(relay_mode_from_endpoint_addr(&addr));
    let endpoint = race_cancel(builder.bind(), cancelled.as_ref(), cancel_notify.as_ref())
        .await?
        .map_err(|error| StreamFailure::transport(format!("bind mesh endpoint: {error}")))?;

    let result = async {
        if addr.relay_urls().next().is_some() {
            let _ = race_cancel(
                tokio::time::timeout(config.connect_timeout, endpoint.online()),
                cancelled.as_ref(),
                cancel_notify.as_ref(),
            )
            .await?;
        }
        let connection = race_cancel(
            tokio::time::timeout(config.connect_timeout, endpoint.connect(addr, ALPN_V1)),
            cancelled.as_ref(),
            cancel_notify.as_ref(),
        )
        .await?
        .map_err(|_| StreamFailure::transport("connect mesh endpoint: timed out"))?
        .map_err(|error| StreamFailure::transport(format!("connect mesh endpoint: {error}")))?;
        let (mut send, mut recv) = race_cancel(
            connection.open_bi(),
            cancelled.as_ref(),
            cancel_notify.as_ref(),
        )
        .await?
        .map_err(|error| StreamFailure::transport(format!("open mesh request stream: {error}")))?;
        race_cancel(
            send.write_all(&[STREAM_TUNNEL_HTTP]),
            cancelled.as_ref(),
            cancel_notify.as_ref(),
        )
        .await?
        .map_err(|error| {
            StreamFailure::transport(format!("write mesh request stream type: {error}"))
        })?;
        race_cancel(
            send.write_all(request.as_bytes()),
            cancelled.as_ref(),
            cancel_notify.as_ref(),
        )
        .await?
        .map_err(|error| StreamFailure::transport(format!("write mesh request: {error}")))?;
        send.finish()
            .map_err(|error| StreamFailure::transport(format!("finish mesh request: {error}")))?;

        let result =
            consume_response(&mut recv, request_id, cancelled, cancel_notify, listener).await;
        connection.close(0u32.into(), b"mesh-client-stream-complete");
        result
    }
    .await;

    endpoint.close().await;
    result
}

async fn stream_http(
    api_base_url: &str,
    request: String,
    request_id: &str,
    cancelled: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    listener: Arc<dyn OpenAiStreamListener>,
) -> Result<(), StreamFailure> {
    let address = socket_addr(api_base_url).map_err(StreamFailure::transport)?;
    let mut stream = race_cancel(
        TcpStream::connect(&address),
        cancelled.as_ref(),
        cancel_notify.as_ref(),
    )
    .await?
    .map_err(|error| StreamFailure::transport(format!("connect {address}: {error}")))?;
    race_cancel(
        stream.write_all(request.as_bytes()),
        cancelled.as_ref(),
        cancel_notify.as_ref(),
    )
    .await?
    .map_err(|error| StreamFailure::transport(format!("write request: {error}")))?;
    race_cancel(
        stream.shutdown(),
        cancelled.as_ref(),
        cancel_notify.as_ref(),
    )
    .await?
    .map_err(|error| StreamFailure::transport(format!("shutdown request: {error}")))?;
    consume_response(&mut stream, request_id, cancelled, cancel_notify, listener).await
}

async fn consume_response<R: AsyncRead + Unpin>(
    reader: &mut R,
    request_id: &str,
    cancelled: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    listener: Arc<dyn OpenAiStreamListener>,
) -> Result<(), StreamFailure> {
    let (head, body_prefix) = read_response_head(reader, &cancelled, &cancel_notify).await?;
    let mut body = BodyDecoder::new(head.chunked, head.content_length);

    if !(200..300).contains(&head.status_code) {
        let bytes =
            read_body_to_end(reader, &mut body, body_prefix, &cancelled, &cancel_notify).await?;
        let body = String::from_utf8_lossy(&bytes).into_owned();
        return Err(StreamFailure::http(head.status_code, body));
    }

    listener.on_event(OpenAiStreamEvent::Started {
        request_id: request_id.to_string(),
        status_code: head.status_code,
        content_type: head.content_type.clone(),
    });

    if !head
        .content_type
        .as_deref()
        .is_some_and(|value| value.to_ascii_lowercase().starts_with("text/event-stream"))
    {
        let bytes =
            read_body_to_end(reader, &mut body, body_prefix, &cancelled, &cancel_notify).await?;
        let received = String::from_utf8_lossy(&bytes).into_owned();
        return Err(StreamFailure {
            status_code: Some(head.status_code),
            message: "streaming response did not use text/event-stream".to_string(),
            body: Some(received),
            cancelled: false,
        });
    }

    let mut sse = SseDecoder::default();
    emit_decoded(
        &mut body,
        &mut sse,
        &body_prefix,
        request_id,
        listener.as_ref(),
    )?;
    let mut read_buffer = [0_u8; 16 * 1024];
    while !body.is_done() {
        let count = read_with_cancel(
            reader,
            &mut read_buffer,
            cancelled.as_ref(),
            cancel_notify.as_ref(),
        )
        .await?;
        if count == 0 {
            break;
        }
        emit_decoded(
            &mut body,
            &mut sse,
            &read_buffer[..count],
            request_id,
            listener.as_ref(),
        )?;
    }
    body.finish()?;
    for event in sse.finish()? {
        emit_sse(listener.as_ref(), request_id, event);
    }
    Ok(())
}

async fn read_response_head<R: AsyncRead + Unpin>(
    reader: &mut R,
    cancelled: &AtomicBool,
    cancel_notify: &Notify,
) -> Result<(ResponseHead, Vec<u8>), StreamFailure> {
    let mut received = Vec::new();
    let mut read_buffer = [0_u8; 4096];
    loop {
        if let Some(header_end) = received.windows(4).position(|window| window == b"\r\n\r\n") {
            let body_prefix = received.split_off(header_end + 4);
            let head = parse_response_head(&received)?;
            return Ok((head, body_prefix));
        }
        if received.len() >= MAX_RESPONSE_HEADER_BYTES {
            return Err(StreamFailure::transport(
                "OpenAI response headers exceed 64 KiB",
            ));
        }
        let count = read_with_cancel(reader, &mut read_buffer, cancelled, cancel_notify).await?;
        if count == 0 {
            return Err(StreamFailure::transport(
                "connection closed before OpenAI response headers completed",
            ));
        }
        received.extend_from_slice(&read_buffer[..count]);
    }
}

fn parse_response_head(bytes: &[u8]) -> Result<ResponseHead, StreamFailure> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut parsed = httparse::Response::new(&mut headers);
    let status = parsed
        .parse(bytes)
        .map_err(|error| StreamFailure::transport(format!("parse HTTP response: {error}")))?;
    if !status.is_complete() {
        return Err(StreamFailure::transport("incomplete HTTP response headers"));
    }
    let status_code = parsed
        .code
        .ok_or_else(|| StreamFailure::transport("HTTP response is missing a status code"))?;
    let content_type = header(&parsed, "content-type")
        .map(|value| String::from_utf8_lossy(value).trim().to_string());
    let chunked = header(&parsed, "transfer-encoding").is_some_and(|value| {
        String::from_utf8_lossy(value)
            .split(',')
            .any(|part| part.trim().eq_ignore_ascii_case("chunked"))
    });
    let content_length = header(&parsed, "content-length")
        .map(|value| String::from_utf8_lossy(value).trim().parse::<usize>())
        .transpose()
        .map_err(|error| StreamFailure::transport(format!("invalid Content-Length: {error}")))?;
    Ok(ResponseHead {
        status_code,
        content_type,
        chunked,
        content_length,
    })
}

fn header<'a>(response: &'a httparse::Response<'a, 'a>, name: &str) -> Option<&'a [u8]> {
    response
        .headers
        .iter()
        .find(|header| header.name.eq_ignore_ascii_case(name))
        .map(|header| header.value)
}

async fn read_body_to_end<R: AsyncRead + Unpin>(
    reader: &mut R,
    body: &mut BodyDecoder,
    prefix: Vec<u8>,
    cancelled: &AtomicBool,
    cancel_notify: &Notify,
) -> Result<Vec<u8>, StreamFailure> {
    let mut output = body.push(&prefix)?;
    let mut read_buffer = [0_u8; 16 * 1024];
    while !body.is_done() {
        let count = read_with_cancel(reader, &mut read_buffer, cancelled, cancel_notify).await?;
        if count == 0 {
            break;
        }
        output.extend(body.push(&read_buffer[..count])?);
        if output.len() > MAX_ERROR_BODY_BYTES {
            return Err(StreamFailure::transport(
                "OpenAI error response exceeds 64 MiB",
            ));
        }
    }
    body.finish()?;
    Ok(output)
}

async fn read_with_cancel<R: AsyncRead + Unpin>(
    reader: &mut R,
    buffer: &mut [u8],
    cancelled: &AtomicBool,
    cancel_notify: &Notify,
) -> Result<usize, StreamFailure> {
    if cancelled.load(Ordering::Acquire) {
        return Err(StreamFailure::cancelled());
    }
    tokio::select! {
        biased;
        _ = cancel_notify.notified() => Err(StreamFailure::cancelled()),
        result = reader.read(buffer) => result
            .map_err(|error| StreamFailure::transport(format!("read OpenAI response: {error}"))),
    }
}

async fn race_cancel<F: Future>(
    future: F,
    cancelled: &AtomicBool,
    cancel_notify: &Notify,
) -> Result<F::Output, StreamFailure> {
    if cancelled.load(Ordering::Acquire) {
        return Err(StreamFailure::cancelled());
    }
    tokio::select! {
        biased;
        _ = cancel_notify.notified() => Err(StreamFailure::cancelled()),
        output = future => Ok(output),
    }
}

fn emit_decoded(
    body: &mut BodyDecoder,
    sse: &mut SseDecoder,
    wire: &[u8],
    request_id: &str,
    listener: &dyn OpenAiStreamListener,
) -> Result<(), StreamFailure> {
    let decoded = body.push(wire)?;
    for event in sse.push(&decoded)? {
        emit_sse(listener, request_id, event);
    }
    Ok(())
}

fn emit_sse(listener: &dyn OpenAiStreamListener, request_id: &str, event: SseEvent) {
    listener.on_event(OpenAiStreamEvent::Sse {
        request_id: request_id.to_string(),
        event_type: event.event_type,
        data: event.data,
        raw: event.raw,
    });
}

enum BodyDecoder {
    Identity { remaining: Option<usize> },
    Chunked(ChunkedDecoder),
}

impl BodyDecoder {
    fn new(chunked: bool, content_length: Option<usize>) -> Self {
        if chunked {
            Self::Chunked(ChunkedDecoder::default())
        } else {
            Self::Identity {
                remaining: content_length,
            }
        }
    }

    fn push(&mut self, input: &[u8]) -> Result<Vec<u8>, StreamFailure> {
        match self {
            Self::Identity { remaining } => {
                let count = remaining.map_or(input.len(), |value| value.min(input.len()));
                if let Some(value) = remaining {
                    *value -= count;
                }
                Ok(input[..count].to_vec())
            }
            Self::Chunked(decoder) => decoder.push(input),
        }
    }

    fn is_done(&self) -> bool {
        match self {
            Self::Identity { remaining } => remaining.is_some_and(|value| value == 0),
            Self::Chunked(decoder) => decoder.finished,
        }
    }

    fn finish(&self) -> Result<(), StreamFailure> {
        match self {
            Self::Identity {
                remaining: Some(remaining),
            } if *remaining != 0 => Err(StreamFailure::transport(format!(
                "OpenAI response ended with {remaining} body bytes missing"
            ))),
            Self::Chunked(decoder) if !decoder.finished => Err(StreamFailure::transport(
                "OpenAI chunked response ended before the terminating chunk",
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Default)]
struct ChunkedDecoder {
    buffer: Vec<u8>,
    remaining: Option<usize>,
    reading_trailers: bool,
    finished: bool,
}

impl ChunkedDecoder {
    fn push(&mut self, input: &[u8]) -> Result<Vec<u8>, StreamFailure> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.buffer.extend_from_slice(input);
        let mut output = Vec::new();
        loop {
            if self.reading_trailers {
                if self.buffer.starts_with(b"\r\n") {
                    self.buffer.drain(..2);
                    self.finished = true;
                    break;
                }
                if let Some(trailer_end) = self
                    .buffer
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                {
                    if trailer_end + 4 > MAX_RESPONSE_HEADER_BYTES {
                        return Err(StreamFailure::transport(
                            "malformed chunked response: trailers exceed 64 KiB",
                        ));
                    }
                    self.buffer.drain(..trailer_end + 4);
                    self.finished = true;
                }
                if !self.finished && self.buffer.len() > MAX_RESPONSE_HEADER_BYTES {
                    return Err(StreamFailure::transport(
                        "malformed chunked response: trailers exceed 64 KiB",
                    ));
                }
                break;
            }

            if let Some(remaining) = self.remaining {
                if remaining > 0 {
                    let count = remaining.min(self.buffer.len());
                    output.extend_from_slice(&self.buffer[..count]);
                    self.buffer.drain(..count);
                    self.remaining = Some(remaining - count);
                    if count == 0 || count < remaining {
                        break;
                    }
                    continue;
                }
                if self.buffer.len() < 2 {
                    break;
                }
                if !self.buffer.starts_with(b"\r\n") {
                    return Err(StreamFailure::transport(
                        "malformed chunked response: missing chunk terminator",
                    ));
                }
                self.buffer.drain(..2);
                self.remaining = None;
                continue;
            }

            let Some(line_end) = self.buffer.windows(2).position(|window| window == b"\r\n") else {
                if self.buffer.len() > MAX_CHUNK_LINE_BYTES {
                    return Err(StreamFailure::transport(
                        "malformed chunked response: chunk size line exceeds 4 KiB",
                    ));
                }
                break;
            };
            if line_end > MAX_CHUNK_LINE_BYTES {
                return Err(StreamFailure::transport(
                    "malformed chunked response: chunk size line exceeds 4 KiB",
                ));
            }
            let size_text = std::str::from_utf8(&self.buffer[..line_end]).map_err(|error| {
                StreamFailure::transport(format!("invalid chunk size: {error}"))
            })?;
            let size_text = size_text.split(';').next().unwrap_or(size_text).trim();
            let size = usize::from_str_radix(size_text, 16).map_err(|error| {
                StreamFailure::transport(format!("invalid chunk size '{size_text}': {error}"))
            })?;
            self.buffer.drain(..line_end + 2);
            if size == 0 {
                self.reading_trailers = true;
                continue;
            }
            self.remaining = Some(size);
        }
        Ok(output)
    }
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
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
    fn push(&mut self, input: &[u8]) -> Result<Vec<SseEvent>, StreamFailure> {
        self.buffer.extend_from_slice(input);
        let mut events = Vec::new();
        while let Some(end) = self.next_frame_end() {
            let raw = self.buffer.drain(..end).collect::<Vec<_>>();
            self.scan_offset = 0;
            self.line_has_content = false;
            if let Some(event) = parse_sse_event(&raw)? {
                events.push(event);
            }
        }
        if self.buffer.len() > MAX_SSE_EVENT_BYTES {
            return Err(StreamFailure::transport("OpenAI SSE event exceeds 8 MiB"));
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<SseEvent>, StreamFailure> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        let raw = std::mem::take(&mut self.buffer);
        self.scan_offset = 0;
        self.line_has_content = false;
        Ok(parse_sse_event(&raw)?.into_iter().collect())
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

fn parse_sse_event(raw: &[u8]) -> Result<Option<SseEvent>, StreamFailure> {
    let raw = String::from_utf8(raw.to_vec())
        .map_err(|error| StreamFailure::transport(format!("OpenAI SSE is not UTF-8: {error}")))?;
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
    Ok(Some(SseEvent {
        event_type,
        data: data.join("\n"),
        raw,
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        BodyDecoder, MAX_CHUNK_LINE_BYTES, MAX_RESPONSE_HEADER_BYTES, SseDecoder, SseEvent,
        consume_response,
    };
    use crate::events::{OpenAiStreamEvent, OpenAiStreamListener};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::io::AsyncWriteExt;
    use tokio::sync::Notify;

    #[derive(Default)]
    struct RecordingListener {
        events: Mutex<Vec<OpenAiStreamEvent>>,
    }

    impl OpenAiStreamListener for RecordingListener {
        fn on_event(&self, event: OpenAiStreamEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    #[test]
    fn decodes_tool_call_sse_across_http_chunks() {
        let first = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"ci\"}}]}}]}\n\n";
        let second = b"event: response.function_call_arguments.delta\ndata: {\"delta\":\"ty\\\":\\\"Sydney\\\"}\"}\n\n";
        let mut wire = Vec::new();
        for payload in [first.as_slice(), second.as_slice(), b"data: [DONE]\n\n"] {
            wire.extend_from_slice(format!("{:X}\r\n", payload.len()).as_bytes());
            wire.extend_from_slice(payload);
            wire.extend_from_slice(b"\r\n");
        }
        wire.extend_from_slice(b"0\r\n\r\n");

        let mut body = BodyDecoder::new(true, None);
        let mut sse = SseDecoder::default();
        let mut events = Vec::new();
        for byte in wire.chunks(3) {
            let decoded = body.push(byte).expect("chunked bytes decode");
            events.extend(sse.push(&decoded).expect("SSE frames decode"));
        }
        body.finish().expect("chunked body terminates");
        events.extend(sse.finish().expect("SSE decoder finishes"));

        assert_eq!(events.len(), 3);
        assert!(events[0].data.contains("tool_calls"));
        assert_eq!(
            events[1],
            SseEvent {
                event_type: Some("response.function_call_arguments.delta".to_string()),
                data: r#"{"delta":"ty\":\"Sydney\"}"}"#.to_string(),
                raw: String::from_utf8(second.to_vec()).unwrap(),
            }
        );
        assert_eq!(events[2].data, "[DONE]");
    }

    #[test]
    fn emits_chunk_payload_before_the_http_chunk_terminator_arrives() {
        let mut body = BodyDecoder::new(true, None);

        assert_eq!(
            body.push(b"5\r\nhel").expect("partial chunk decodes"),
            b"hel"
        );
        assert_eq!(body.push(b"lo").expect("chunk remainder decodes"), b"lo");
        assert!(body.push(b"\r\n0\r\n\r\n").expect("chunk ends").is_empty());
        body.finish().expect("chunked body terminates");
    }

    #[test]
    fn accepts_cr_only_sse_line_endings() {
        let raw = b"event: update\rdata: one\rdata: two\r\r";
        let event = SseDecoder::default()
            .push(raw)
            .expect("CR-only SSE frame decodes")
            .pop()
            .expect("event emitted");

        assert_eq!(event.event_type.as_deref(), Some("update"));
        assert_eq!(event.data, "one\ntwo");
    }

    #[test]
    fn rejects_unterminated_oversized_chunk_size_line() {
        let mut body = BodyDecoder::new(true, None);
        let error = body
            .push(&vec![b'F'; MAX_CHUNK_LINE_BYTES + 1])
            .expect_err("oversized chunk size line fails");

        assert!(error.message.contains("chunk size line exceeds 4 KiB"));
    }

    #[test]
    fn rejects_unterminated_oversized_chunk_trailers() {
        let mut body = BodyDecoder::new(true, None);
        body.push(b"0\r\n")
            .expect("terminating chunk starts trailers");
        let error = body
            .push(&vec![b'x'; MAX_RESPONSE_HEADER_BYTES + 1])
            .expect_err("oversized trailers fail");

        assert!(error.message.contains("trailers exceed 64 KiB"));
    }

    #[test]
    fn preserves_multiline_data_and_unknown_sse_fields_in_raw_frame() {
        let raw = b"id: 42\nevent: response.output_text.delta\ndata: {\"delta\":\"hello\"}\ndata: {\"more\":true}\nretry: 1000\n\n";
        let event = SseDecoder::default()
            .push(raw)
            .expect("SSE frame decodes")
            .pop()
            .expect("event emitted");

        assert_eq!(
            event.event_type.as_deref(),
            Some("response.output_text.delta")
        );
        assert_eq!(event.data, "{\"delta\":\"hello\"}\n{\"more\":true}");
        assert_eq!(event.raw.as_bytes(), raw);
    }

    #[tokio::test]
    async fn consumes_http_response_and_preserves_tool_call_event() {
        let payload = b"event: response.function_call_arguments.delta\ndata: {\"delta\":\"{\\\"city\\\":\\\"Sydney\\\"}\"}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
            payload.len(),
            String::from_utf8_lossy(payload)
        );
        let (mut reader, mut writer) = tokio::io::duplex(response.len());
        writer.write_all(response.as_bytes()).await.unwrap();
        writer.shutdown().await.unwrap();

        let listener = Arc::new(RecordingListener::default());
        consume_response(
            &mut reader,
            "request-1",
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
            listener.clone(),
        )
        .await
        .unwrap();

        let events = listener.events.lock().unwrap();
        assert!(matches!(
            events.first(),
            Some(OpenAiStreamEvent::Started {
                status_code: 200,
                ..
            })
        ));
        assert!(matches!(
            events.get(1),
            Some(OpenAiStreamEvent::Sse {
                event_type: Some(event_type),
                data,
                ..
            }) if event_type == "response.function_call_arguments.delta"
                && data.contains("Sydney")
        ));
    }

    #[tokio::test]
    async fn preserves_http_error_status_and_body() {
        let body = r#"{"error":{"message":"rate limited"}}"#;
        let response = format!(
            "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (mut reader, mut writer) = tokio::io::duplex(response.len());
        writer.write_all(response.as_bytes()).await.unwrap();
        writer.shutdown().await.unwrap();

        let failure = consume_response(
            &mut reader,
            "request-1",
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
            Arc::new(RecordingListener::default()),
        )
        .await
        .expect_err("non-success status fails the stream");

        assert_eq!(failure.status_code, Some(429));
        assert_eq!(failure.body.as_deref(), Some(body));
        assert!(failure.message.contains("429"));
    }

    #[tokio::test]
    async fn cancellation_interrupts_blocked_response_read() {
        let (mut reader, _writer) = tokio::io::duplex(64);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_notify = Arc::new(Notify::new());
        let listener = Arc::new(RecordingListener::default());
        let task_cancelled = cancelled.clone();
        let task_notify = cancel_notify.clone();

        let task = tokio::spawn(async move {
            consume_response(
                &mut reader,
                "request-1",
                task_cancelled,
                task_notify,
                listener,
            )
            .await
        });
        tokio::task::yield_now().await;
        cancelled.store(true, Ordering::Release);
        cancel_notify.notify_one();

        let failure = task
            .await
            .expect("response task joins")
            .expect_err("response read is cancelled");
        assert!(failure.cancelled);
    }

    #[test]
    fn scans_each_byte_once_across_chunks() {
        // A long unterminated frame used to be rescanned from the buffer start on
        // every push (O(n^2)). Each push must now examine only the bytes it added.
        const CHUNK: usize = 4 * 1024;
        let chunk = vec![b'x'; CHUNK];
        let mut decoder = SseDecoder::default();
        let mut per_push = Vec::new();
        for _ in 0..64 {
            let before = decoder.scanned_bytes;
            assert!(
                decoder
                    .push(&chunk)
                    .expect("unterminated SSE frame buffers")
                    .is_empty()
            );
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
    fn decodes_crlf_frame_split_between_cr_and_lf() {
        let mut decoder = SseDecoder::default();
        assert!(
            decoder
                .push(b"data: one\r")
                .expect("partial frame buffers")
                .is_empty()
        );
        let events = decoder
            .push(b"\ndata: two\r\n\r\n")
            .expect("frame completes");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "one\ntwo");
        assert_eq!(events[0].raw, "data: one\r\ndata: two\r\n\r\n");
    }
}
