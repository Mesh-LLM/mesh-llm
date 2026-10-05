//! Bounded SSE evidence for pipeline relays, independent of HTTP success status.
use super::super::common::{parse_token_usage_from_json_body, sse_data_frame_is_openai_error};
use mesh_llm_events::logging::events::TokenUsage;

#[derive(Default)]
pub(super) struct SseUsageParser {
    line: Vec<u8>,
    data: Vec<u8>,
    discard_frame: bool,
    discard_line: bool,
    event_error: bool,
    done: bool,
    error: bool,
    pub(super) usage: Option<TokenUsage>,
}

impl SseUsageParser {
    pub(super) fn outcome(&self) -> Option<&'static str> {
        if self.error {
            Some("backend_error")
        } else if !self.done {
            Some("transport_error")
        } else {
            None
        }
    }

    pub(super) fn push(&mut self, bytes: &[u8]) {
        for byte in bytes {
            if *byte == b'\n' {
                self.finish_line();
            } else if self.line.len() < 64 * 1024 {
                self.line.push(*byte);
            } else {
                self.discard_frame = true;
                self.discard_line = true;
            }
        }
    }

    fn finish_line(&mut self) {
        let line = std::mem::take(&mut self.line);
        let line = line.strip_suffix(b"\r").unwrap_or(&line);
        if self.discard_line {
            self.discard_line = false;
            return;
        }
        if line.is_empty() {
            if !self.discard_frame {
                self.finish_frame();
            }
            self.data.clear();
            self.event_error = false;
            self.discard_frame = false;
        } else if let Some(event) = line.strip_prefix(b"event:") {
            self.event_error = event.strip_prefix(b" ").unwrap_or(event) == b"error";
        } else if let Some(data) = line.strip_prefix(b"data:") {
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if self.data.len().saturating_add(data.len()).saturating_add(1) <= 64 * 1024 {
                self.data.extend_from_slice(data);
                self.data.push(b'\n');
            } else {
                self.discard_frame = true;
                self.data.clear();
            }
        }
    }

    fn finish_frame(&mut self) {
        self.error |= self.event_error;
        let data = self.data.strip_suffix(b"\n").unwrap_or(&self.data);
        if data == b"[DONE]" {
            self.done = true;
            return;
        }
        self.error |= std::str::from_utf8(data).is_ok_and(sse_data_frame_is_openai_error);
        if let Some(usage) = parse_token_usage_from_json_body(data) {
            self.usage = Some(usage);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn error_transcript_is_backend_failure_even_with_http_200_and_done() {
        let bytes = b"event: error\r\ndata: {\r\ndata: \"error\":{\"message\":\"failed\"}}\r\n\r\ndata: [DONE]\r\n\r\n";
        for size in 1..=bytes.len() {
            let mut parser = SseUsageParser::default();
            for fragment in bytes.chunks(size) {
                parser.push(fragment);
            }
            assert_eq!(parser.outcome(), Some("backend_error"));
        }
    }
    #[test]
    fn clean_http_eof_without_done_is_incomplete_stream() {
        let mut parser = SseUsageParser::default();
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n");
        assert_eq!(parser.outcome(), Some("transport_error"));
        parser.push(b"data: [DONE]\n\n");
        assert_eq!(parser.outcome(), None);
    }
    #[test]
    fn explicit_error_event_is_authoritative_without_top_level_error_data() {
        let bytes = b"event: error\r\ndata: {\"message\":\"failed\"}\r\n\r\ndata: [DONE]\r\n\r\n";
        for size in 1..=bytes.len() {
            let mut parser = SseUsageParser::default();
            for fragment in bytes.chunks(size) {
                parser.push(fragment);
            }
            assert_eq!(parser.outcome(), Some("backend_error"));
            assert!(
                !parser.event_error,
                "event state must reset after each frame"
            );
        }
    }
    #[test]
    fn oversized_frame_never_makes_terminal_marker_in_its_tail_authoritative() {
        let mut parser = SseUsageParser::default();
        parser.push(&vec![b'x'; 128 * 1024]);
        assert!(parser.line.len() <= 64 * 1024);
        parser.push(b"\ndata: [DONE]\n\n");
        assert_eq!(parser.outcome(), Some("transport_error"));
        parser.push(b"data: [DONE]\n\n");
        assert_eq!(parser.outcome(), None);
    }
}
