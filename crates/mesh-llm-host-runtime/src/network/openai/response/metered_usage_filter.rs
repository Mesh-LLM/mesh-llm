//! Hides backend stream usage from paid callers that did not request it.
//!
//! The payer always asks the backend for `stream_options.include_usage` so the
//! provider can settle from reported usage. The plain OpenAI frontend never
//! shows usage to a caller that did not opt in (`router.rs` drops usage
//! chunks), so the metered path must not either. This filter rewrites only the
//! client-visible HTTP/SSE bytes; provider evidence and settlement never see
//! it.
//!
//! Usage-only events (empty or absent `choices`) are dropped. A usage object
//! attached to a content chunk is removed so the content still reaches the
//! client. Every other event, including errors and `[DONE]`, is forwarded
//! byte-for-byte. Malformed framing or an over-limit event switches to
//! verbatim passthrough rather than corrupting or truncating the response.
//!
//! Limitation: suppression is best effort. After a fallback the rest of the
//! response, including any later usage event, reaches the client unfiltered.
//! Fallback stays framing-consistent with what was already emitted: chunked
//! responses keep `Transfer-Encoding: chunked`, held-back decoded bytes are
//! re-chunked, and the remaining raw input always starts on an original chunk
//! boundary (only whole chunks are consumed). Identity responses already had
//! `Content-Length` removed and are delimited by connection close.

use serde_json::Value;

const LIMIT: usize = 2 * 1024 * 1024;

enum Mode {
    Headers,
    Filter { chunked: bool },
    Passthrough,
}

pub(super) struct UsageFilter {
    mode: Mode,
    /// Undecoded response bytes (headers or HTTP chunk framing).
    raw: Vec<u8>,
    /// Decoded SSE body bytes not yet forming a complete event.
    event: Vec<u8>,
}

impl Default for UsageFilter {
    fn default() -> Self {
        Self {
            mode: Mode::Headers,
            raw: Vec::new(),
            event: Vec::new(),
        }
    }
}

impl UsageFilter {
    /// Feed backend response bytes; returns the bytes to write to the client.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        if matches!(self.mode, Mode::Passthrough) {
            out.extend_from_slice(bytes);
            return out;
        }
        self.raw.extend_from_slice(bytes);
        if matches!(self.mode, Mode::Headers) {
            self.headers(&mut out);
        }
        match self.mode {
            Mode::Headers => {}
            Mode::Passthrough => out.append(&mut self.raw),
            Mode::Filter { chunked: true } => self.chunks(&mut out),
            Mode::Filter { chunked: false } => {
                let body = std::mem::take(&mut self.raw);
                self.body(&body, false, &mut out);
            }
        }
        out
    }

    /// Flush anything held back at end of stream, verbatim.
    pub fn finish(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let chunked = matches!(self.mode, Mode::Filter { chunked: true });
        emit(&mut out, &std::mem::take(&mut self.event), chunked);
        out.append(&mut self.raw);
        self.mode = Mode::Passthrough;
        out
    }

    fn headers(&mut self, out: &mut Vec<u8>) {
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut response = httparse::Response::new(&mut headers);
        let offset = match response.parse(&self.raw) {
            Ok(httparse::Status::Complete(offset)) => offset,
            Ok(httparse::Status::Partial) if self.raw.len() <= 64 * 1024 => return,
            _ => {
                self.mode = Mode::Passthrough;
                return;
            }
        };
        let has = |name: &str, value: &[u8]| {
            response.headers.iter().any(|header| {
                header.name.eq_ignore_ascii_case(name)
                    && header
                        .value
                        .get(..value.len())
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(value))
            })
        };
        let filterable = response.code.is_some_and(|code| (200..300).contains(&code))
            && has("content-type", b"text/event-stream")
            && response.headers.iter().all(|header| {
                !header.name.eq_ignore_ascii_case("content-encoding")
                    || header.value.eq_ignore_ascii_case(b"identity")
            });
        if !filterable {
            self.mode = Mode::Passthrough;
            return;
        }
        // Only exactly one `Transfer-Encoding: chunked` (or none) is
        // understood; any other coding (`gzip, chunked`, `chunkedjunk`, ...)
        // passes through untouched before any transformation.
        let mut codings = response
            .headers
            .iter()
            .filter(|header| header.name.eq_ignore_ascii_case("transfer-encoding"));
        let chunked = match (codings.next(), codings.next()) {
            (None, _) => false,
            (Some(header), None) if header.value.trim_ascii().eq_ignore_ascii_case(b"chunked") => {
                true
            }
            _ => {
                self.mode = Mode::Passthrough;
                return;
            }
        };
        // The body length changes, so drop any Content-Length. The backend
        // response is `Connection: close`, so EOF still delimits it.
        let head = &self.raw[..offset];
        let mut start = 0;
        while let Some(end) = head[start..].windows(2).position(|w| w == b"\r\n") {
            let line = &head[start..start + end + 2];
            let lower = line.to_ascii_lowercase();
            if !lower.starts_with(b"content-length:") {
                out.extend_from_slice(line);
            }
            start += end + 2;
        }
        out.extend_from_slice(&head[start..]);
        self.raw.drain(..offset);
        self.mode = Mode::Filter { chunked };
    }

    fn chunks(&mut self, out: &mut Vec<u8>) {
        loop {
            let (offset, length) = match httparse::parse_chunk_size(&self.raw) {
                Ok(httparse::Status::Complete(parsed)) => parsed,
                Ok(httparse::Status::Partial) if self.raw.len() <= 1024 => return,
                _ => return self.fallback(true, out),
            };
            if length > LIMIT as u64 {
                return self.fallback(true, out);
            }
            let length = length as usize;
            if length == 0 {
                // Terminal chunk and trailers pass through unchanged.
                emit(out, &std::mem::take(&mut self.event), true);
                out.append(&mut self.raw);
                self.mode = Mode::Passthrough;
                return;
            }
            if self.raw.len() < offset + length + 2 {
                return;
            }
            if &self.raw[offset + length..offset + length + 2] != b"\r\n" {
                return self.fallback(true, out);
            }
            let data = self.raw[offset..offset + length].to_vec();
            self.raw.drain(..offset + length + 2);
            self.body(&data, true, out);
            if matches!(self.mode, Mode::Passthrough) {
                out.append(&mut self.raw);
                return;
            }
        }
    }

    fn body(&mut self, bytes: &[u8], chunked: bool, out: &mut Vec<u8>) {
        self.event.extend_from_slice(bytes);
        let mut filtered = Vec::new();
        while let Some(end) = event_end(&self.event) {
            let event = self.event.drain(..end).collect::<Vec<_>>();
            filter_event(&event, &mut filtered);
        }
        emit(out, &filtered, chunked);
        if self.event.len() > LIMIT {
            emit(out, &std::mem::take(&mut self.event), chunked);
            self.mode = Mode::Passthrough;
        }
    }

    fn fallback(&mut self, chunked: bool, out: &mut Vec<u8>) {
        tracing::warn!("metered stream framing unrecognised; forwarding unfiltered");
        emit(out, &std::mem::take(&mut self.event), chunked);
        out.append(&mut self.raw);
        self.mode = Mode::Passthrough;
    }
}

fn emit(out: &mut Vec<u8>, bytes: &[u8], chunked: bool) {
    if bytes.is_empty() {
        return;
    }
    if chunked {
        out.extend_from_slice(format!("{:x}\r\n", bytes.len()).as_bytes());
        out.extend_from_slice(bytes);
        out.extend_from_slice(b"\r\n");
    } else {
        out.extend_from_slice(bytes);
    }
}

/// End offset of the first complete SSE event (through its blank line).
/// Accepts CRLF, LF and CR line endings. A trailing CR is treated as
/// incomplete because it may be the first half of CRLF.
fn event_end(bytes: &[u8]) -> Option<usize> {
    let mut line_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let terminator = match bytes[i] {
            b'\n' => 1,
            b'\r' if i + 1 == bytes.len() => return None,
            b'\r' if bytes[i + 1] == b'\n' => 2,
            b'\r' => 1,
            _ => {
                i += 1;
                continue;
            }
        };
        if i == line_start {
            return Some(i + terminator);
        }
        i += terminator;
        line_start = i;
    }
    None
}

fn lines(event: &[u8]) -> Vec<&[u8]> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < event.len() {
        let terminator = match event[i] {
            b'\r' if event.get(i + 1) == Some(&b'\n') => 2,
            b'\r' | b'\n' => 1,
            _ => {
                i += 1;
                continue;
            }
        };
        lines.push(&event[start..i]);
        i += terminator;
        start = i;
    }
    if start < event.len() {
        lines.push(&event[start..]);
    }
    lines
}

fn filter_event(event: &[u8], out: &mut Vec<u8>) {
    let lines = lines(event);
    let mut data = Vec::new();
    let mut has_data = false;
    for line in &lines {
        let value = if *line == b"data" {
            Some(&b""[..])
        } else {
            line.strip_prefix(b"data:")
        };
        if let Some(value) = value {
            if has_data {
                data.push(b'\n');
            }
            has_data = true;
            data.extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
        }
    }
    let Ok(Value::Object(mut chunk)) = serde_json::from_slice::<Value>(&data) else {
        return out.extend_from_slice(event);
    };
    if chunk.get("usage").is_none_or(Value::is_null) || chunk.contains_key("error") {
        return out.extend_from_slice(event);
    }
    // Only a well-formed chat chunk is touched: no `choices` array means an
    // unknown event shape, which passes through byte for byte.
    let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
        return out.extend_from_slice(event);
    };
    if choices.is_empty() {
        return;
    }
    chunk.remove("usage");
    for line in &lines {
        if !line.is_empty() && *line != b"data" && !line.starts_with(b"data:") {
            out.extend_from_slice(line);
            out.push(b'\n');
        }
    }
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(
        &serde_json::to_vec(&Value::Object(chunk)).expect("JSON object serializes"),
    );
    out.extend_from_slice(b"\n\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTENT: &str =
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\r\n\r\n";
    const USAGE_ONLY: &str =
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n";
    const CONTENT_USAGE: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"}}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n";
    const ERROR: &str = "data: {\"error\":{\"message\":\"boom\"}}\n\n";
    const DONE: &str = "data: [DONE]\n\n";

    fn decode_chunked(mut body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let httparse::Status::Complete((offset, length)) =
                httparse::parse_chunk_size(body).unwrap()
            else {
                panic!("partial chunk")
            };
            let length = length as usize;
            if length == 0 {
                return out;
            }
            out.extend_from_slice(&body[offset..offset + length]);
            assert_eq!(&body[offset + length..offset + length + 2], b"\r\n");
            body = &body[offset + length + 2..];
        }
    }

    fn run(response: &[u8], step: usize) -> Vec<u8> {
        let mut filter = UsageFilter::default();
        let mut out = Vec::new();
        for piece in response.chunks(step) {
            out.extend(filter.push(piece));
        }
        out.extend(filter.finish());
        out
    }

    fn split(out: &[u8]) -> (String, Vec<u8>) {
        let end = out.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        (
            String::from_utf8(out[..end].to_vec()).unwrap(),
            out[end..].to_vec(),
        )
    }

    fn events() -> String {
        format!("{CONTENT}{CONTENT_USAGE}{ERROR}{USAGE_ONLY}{DONE}")
    }

    fn expected() -> String {
        format!(
            "{CONTENT}data: {{\"choices\":[{{\"delta\":{{\"content\":\"!\"}},\"index\":0}}]}}\n\n{ERROR}{DONE}"
        )
    }

    #[test]
    fn chunked_stream_hides_usage_at_every_split() {
        let body = events();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n0\r\n\r\n",
            body.len()
        );
        for step in 1..=response.len() {
            let (head, body) = split(&run(response.as_bytes(), step));
            assert!(head.contains("Transfer-Encoding: chunked"));
            let body = decode_chunked(&body);
            let body = String::from_utf8(body).unwrap();
            assert_eq!(body, expected(), "step {step}");
        }
    }

    #[test]
    fn many_http_chunks_and_coalesced_events_are_filtered() {
        let mut response = String::from(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\n\r\n",
        );
        // Split events across HTTP chunks at awkward places, including CRLF.
        let body = events();
        for piece in body.as_bytes().chunks(7) {
            response.push_str(&format!("{:x}\r\n", piece.len()));
            response.push_str(std::str::from_utf8(piece).unwrap());
            response.push_str("\r\n");
        }
        response.push_str("0\r\n\r\n");
        let (_, out) = split(&run(response.as_bytes(), 4096));
        assert_eq!(String::from_utf8(decode_chunked(&out)).unwrap(), expected());
    }

    #[test]
    fn identity_stream_drops_content_length_and_usage() {
        let body = events();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        for step in [1, 3, 16, response.len()] {
            let (head, body) = split(&run(response.as_bytes(), step));
            assert!(!head.to_ascii_lowercase().contains("content-length"));
            assert!(head.contains("Connection: close\r\n"));
            assert_eq!(String::from_utf8(body).unwrap(), expected(), "step {step}");
        }
    }

    #[test]
    fn cr_only_and_multiline_data_events_are_parsed() {
        let body = "event: x\rdata: {\"choices\":[],\r\ndata: \"usage\":{\"prompt_tokens\":1}}\r\rdata: [DONE]\r\r";
        let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{body}");
        for step in 1..=response.len() {
            let (_, out) = split(&run(response.as_bytes(), step));
            assert_eq!(out, b"data: [DONE]\r\r", "step {step}");
        }
    }

    #[test]
    fn non_stream_errors_and_unrecognised_framing_pass_through() {
        for response in [
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"usage\":{\"prompt_tokens\":1}}".to_owned(),
            format!("HTTP/1.1 500 Error\r\nContent-Type: text/event-stream\r\n\r\n{USAGE_ONLY}"),
            format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Encoding: gzip\r\n\r\n{USAGE_ONLY}"),
            format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n{USAGE_ONLY}"),
            format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: gzip, chunked\r\n\r\n{USAGE_ONLY}"),
            format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunkedjunk\r\n\r\n{USAGE_ONLY}"),
            format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n{USAGE_ONLY}"),
        ] {
            assert_eq!(run(response.as_bytes(), 5), response.as_bytes());
        }
    }

    #[test]
    fn event_without_choices_array_is_preserved() {
        for event in [
            "data: {\"usage\":{\"prompt_tokens\":1}}\n\n",
            "data: {\"choices\":null,\"usage\":{\"prompt_tokens\":1}}\n\n",
        ] {
            let mut out = Vec::new();
            filter_event(event.as_bytes(), &mut out);
            assert_eq!(out, event.as_bytes());
        }
    }

    #[test]
    fn error_event_with_usage_is_preserved() {
        let error = "data: {\"error\":{\"message\":\"boom\"},\"usage\":{\"prompt_tokens\":1}}\n\n";
        let mut out = Vec::new();
        filter_event(error.as_bytes(), &mut out);
        assert_eq!(out, error.as_bytes());
    }

    #[test]
    fn incomplete_trailing_event_is_flushed_and_buffering_is_bounded() {
        let partial = "data: {\"choices\":[{\"delta\":{}}]";
        let response =
            format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{partial}");
        let (_, out) = split(&run(response.as_bytes(), 3));
        assert_eq!(out, partial.as_bytes());

        let mut filter = UsageFilter::default();
        let mut out = filter.push(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
        let big = vec![b'a'; LIMIT + 1];
        out.extend(filter.push(&big));
        assert!(filter.event.is_empty() && filter.raw.is_empty());
        out.extend(filter.push(USAGE_ONLY.as_bytes()));
        assert!(out.ends_with(USAGE_ONLY.as_bytes()));
    }

    #[test]
    fn chunked_fallback_after_transformed_event_keeps_valid_framing() {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        let first = format!("{USAGE_ONLY}{CONTENT_USAGE}");
        let partial = "data: {\"choices\":[{\"delta\":{}}]";
        let mut filter = UsageFilter::default();
        let mut out = filter.push(head.as_bytes());
        out.extend(filter.push(format!("{:x}\r\n{first}\r\n", first.len()).as_bytes()));
        // Held-back partial event, then an invalid chunk-size line.
        out.extend(filter.push(format!("{:x}\r\n{partial}\r\n", partial.len()).as_bytes()));
        out.extend(filter.push(b"zz\r\n"));
        assert!(matches!(filter.mode, Mode::Passthrough));
        out.extend(filter.push(b"junk"));
        out.extend(filter.finish());
        let (h, body) = split(&out);
        assert!(
            h.contains("Transfer-Encoding: chunked")
                && !h.to_ascii_lowercase().contains("content-length")
        );
        // Filtered + held-back bytes are well-formed chunks; raw resumes at the bad line.
        let tail_at = body.windows(4).position(|w| w == b"zz\r\n").unwrap();
        let decoded = decode_chunked(&[&body[..tail_at], b"0\r\n\r\n"].concat());
        assert_eq!(
            String::from_utf8(decoded).unwrap(),
            format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"!\"}},\"index\":0}}]}}\n\n{partial}"
            )
        );
        assert_eq!(&body[tail_at..], b"zz\r\njunk");
    }

    #[test]
    fn chunked_buffer_limit_after_transformed_event_keeps_framing_and_trailers() {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        let big = "a".repeat(LIMIT + 1);
        let mut response = String::from(head);
        for piece in [CONTENT_USAGE, USAGE_ONLY, big.as_str(), USAGE_ONLY, DONE] {
            response.push_str(&format!("{:x}\r\n{piece}\r\n", piece.len()));
        }
        response.push_str("0\r\nX-Trailer: 1\r\n\r\n");
        for step in [7, 4096, response.len()] {
            let (h, body) = split(&run(response.as_bytes(), step));
            assert!(h.contains("Transfer-Encoding: chunked"));
            let decoded = String::from_utf8(decode_chunked(&body)).unwrap();
            // Before fallback usage is hidden; after it, passthrough (documented).
            assert_eq!(
                decoded,
                format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"!\"}},\"index\":0}}]}}\n\n{big}{USAGE_ONLY}{DONE}"
                ),
                "step {step}"
            );
            assert!(body.ends_with(b"0\r\nX-Trailer: 1\r\n\r\n"));
        }
    }

    #[test]
    fn identity_buffer_limit_after_transformed_event_is_plain_body() {
        let big = "a".repeat(LIMIT + 1);
        let body = format!("{CONTENT_USAGE}{USAGE_ONLY}{big}{USAGE_ONLY}{DONE}");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        for step in [7, 4096, response.len()] {
            let (h, out) = split(&run(response.as_bytes(), step));
            assert!(!h.to_ascii_lowercase().contains("content-length"));
            assert_eq!(
                String::from_utf8(out).unwrap(),
                format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"!\"}},\"index\":0}}]}}\n\n{big}{USAGE_ONLY}{DONE}"
                ),
                "step {step}"
            );
        }
    }
}
