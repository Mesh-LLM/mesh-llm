//! Streaming HTTP chunked framing verification and raw SSE inspection for
//! passthrough (`ResponseAdapter::None`) response relays.

use super::common::{parse_token_usage_from_json_body, sse_data_frame_is_openai_error};
use mesh_llm_events::logging::events::TokenUsage;

const MAX_CHUNK_LINE_BYTES: usize = 8 * 1024;
const MAX_TRAILER_BYTES: usize = 64 * 1024;
const MAX_SSE_LINE_BYTES: usize = 256 * 1024;
const MAX_SSE_EVENT_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkedFramingStatus {
    Complete,
    Truncated,
}

enum ChunkState {
    Size(Vec<u8>),
    Data(u64),
    DataCrLf(u8),
    Trailers { line: Vec<u8>, total: usize },
    Done,
    Invalid,
}

/// Incremental HTTP/1.1 `Transfer-Encoding: chunked` framing validator.
///
/// Decodes entity bytes on the fly without buffering the whole response so a
/// raw passthrough relay can inspect SSE payloads across arbitrary chunk
/// boundaries and detect premature EOF before the terminating `0\r\n\r\n`.
pub(super) struct ChunkedFramingTracker {
    state: ChunkState,
}

impl Default for ChunkedFramingTracker {
    fn default() -> Self {
        Self {
            state: ChunkState::Size(Vec::new()),
        }
    }
}

impl ChunkedFramingTracker {
    pub(super) fn feed(&mut self, mut bytes: &[u8], mut on_entity: impl FnMut(&[u8])) {
        while !bytes.is_empty() {
            match &mut self.state {
                ChunkState::Size(line) => {
                    let byte = bytes[0];
                    bytes = &bytes[1..];
                    line.push(byte);
                    if line.len() > MAX_CHUNK_LINE_BYTES {
                        self.state = ChunkState::Invalid;
                    } else if line.ends_with(b"\r\n") {
                        self.state = match parse_chunk_size(line) {
                            Some(0) => ChunkState::Trailers {
                                line: Vec::new(),
                                total: 0,
                            },
                            Some(size) => ChunkState::Data(size),
                            None => ChunkState::Invalid,
                        };
                    }
                }
                ChunkState::Data(remaining) => {
                    let count = usize::try_from(*remaining)
                        .unwrap_or(usize::MAX)
                        .min(bytes.len());
                    on_entity(&bytes[..count]);
                    bytes = &bytes[count..];
                    *remaining -= count as u64;
                    if *remaining == 0 {
                        self.state = ChunkState::DataCrLf(0);
                    }
                }
                ChunkState::DataCrLf(pos) => {
                    let byte = bytes[0];
                    bytes = &bytes[1..];
                    if byte != b"\r\n"[*pos as usize] {
                        self.state = ChunkState::Invalid;
                    } else {
                        *pos += 1;
                        if *pos == 2 {
                            self.state = ChunkState::Size(Vec::new());
                        }
                    }
                }
                ChunkState::Trailers { line, total } => {
                    let byte = bytes[0];
                    bytes = &bytes[1..];
                    line.push(byte);
                    *total += 1;
                    if *total > MAX_TRAILER_BYTES {
                        self.state = ChunkState::Invalid;
                    } else if line.ends_with(b"\r\n") {
                        if line.len() == 2 {
                            self.state = ChunkState::Done;
                        } else {
                            line.clear();
                        }
                    }
                }
                ChunkState::Done | ChunkState::Invalid => return,
            }
        }
    }

    pub(super) fn finish(&self) -> ChunkedFramingStatus {
        match self.state {
            ChunkState::Done => ChunkedFramingStatus::Complete,
            _ => ChunkedFramingStatus::Truncated,
        }
    }
}

fn parse_chunk_size(line: &[u8]) -> Option<u64> {
    let line = std::str::from_utf8(line.strip_suffix(b"\r\n")?).ok()?;
    let size = line.split(';').next()?.trim();
    if size.is_empty() || !size.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(size, 16).ok()
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct RawSseOutcome {
    pub(super) usage: Option<TokenUsage>,
    pub(super) saw_output_frame: bool,
    pub(super) saw_done: bool,
    pub(super) saw_error: bool,
}

/// Incremental SSE inspector for raw passthrough streams.
///
/// Ignores comment/keep-alive lines (`: ...`), signals `on_first_output` on the
/// first parsed non-error, non-`[DONE]` output frame, extracts token usage from
/// JSON frames, and detects explicit SSE error frames (`event: error` or
/// OpenAI-shaped `{"error": ...}` payloads) while original bytes are relayed
/// untouched.
#[derive(Default)]
pub(super) struct RawSseInspector {
    line: Vec<u8>,
    line_overflowed: bool,
    line_overflowed_is_data: bool,
    event_data: Vec<u8>,
    event_overflowed: bool,
    event_is_error: bool,
    has_data_line: bool,
    saw_output_frame: bool,
    saw_done: bool,
    saw_error: bool,
    usage: Option<TokenUsage>,
}

impl RawSseInspector {
    pub(super) fn feed(&mut self, bytes: &[u8], mut on_first_output: impl FnMut()) {
        for &byte in bytes {
            if byte == b'\n' {
                self.flush_line(&mut on_first_output);
            } else if !self.line_overflowed {
                if self.line.len() < MAX_SSE_LINE_BYTES {
                    self.line.push(byte);
                } else {
                    self.line_overflowed = true;
                    self.line_overflowed_is_data = self.line.starts_with(b"data:");
                    self.line.clear();
                }
            }
        }
    }

    /// Finalize SSE inspection at EOF.
    ///
    /// Per the SSE specification, a pending event that never received its
    /// terminating blank line (`\n\n` or `\r\n\r\n`) is discarded so a stream
    /// cut off right after `data: [DONE]` never counts as cleanly terminated.
    pub(super) fn finish(self) -> RawSseOutcome {
        RawSseOutcome {
            usage: self.usage,
            saw_output_frame: self.saw_output_frame,
            saw_done: self.saw_done,
            saw_error: self.saw_error,
        }
    }

    fn flush_line(&mut self, on_first_output: &mut impl FnMut()) {
        if self.line_overflowed {
            let is_data = self.line_overflowed_is_data;
            self.line_overflowed = false;
            self.line_overflowed_is_data = false;
            self.line.clear();
            if is_data {
                self.has_data_line = true;
                self.event_overflowed = true;
                self.event_data.clear();
            }
            return;
        }
        let trimmed = self.line.strip_suffix(b"\r").unwrap_or(&self.line);
        if trimmed.is_empty() {
            self.line.clear();
            self.dispatch_event(on_first_output);
            return;
        }
        if trimmed.starts_with(b":") {
            self.line.clear();
            return;
        }
        if let Some(rest) = trimmed.strip_prefix(b"event:") {
            self.event_is_error = trim_ascii_whitespace(rest).eq_ignore_ascii_case(b"error");
        } else if let Some(rest) = trimmed.strip_prefix(b"data:") {
            let payload = strip_leading_ascii_whitespace(rest);
            if !self.event_overflowed {
                let extra = payload.len() + usize::from(self.has_data_line);
                if self.event_data.len() + extra <= MAX_SSE_EVENT_BYTES {
                    if self.has_data_line {
                        self.event_data.push(b'\n');
                    }
                    self.event_data.extend_from_slice(payload);
                } else {
                    self.event_overflowed = true;
                    self.event_data.clear();
                }
            }
            self.has_data_line = true;
        }
        self.line.clear();
    }

    fn dispatch_event(&mut self, on_first_output: &mut impl FnMut()) {
        if !self.has_data_line && !self.event_is_error {
            return;
        }
        let event_is_error = self.event_is_error;
        let event_overflowed = self.event_overflowed;
        let payload = std::mem::take(&mut self.event_data);
        self.event_is_error = false;
        self.has_data_line = false;
        self.event_overflowed = false;

        if event_is_error {
            self.saw_error = true;
            return;
        }
        if event_overflowed {
            self.signal_first_output(on_first_output);
            return;
        }
        let payload = trim_ascii_whitespace(&payload);
        if payload.is_empty() {
            return;
        }
        if payload == b"[DONE]" {
            self.saw_done = true;
            return;
        }
        if std::str::from_utf8(payload).is_ok_and(sse_data_frame_is_openai_error) {
            self.saw_error = true;
            return;
        }
        self.signal_first_output(on_first_output);
        if let Some(usage) = parse_token_usage_from_json_body(payload) {
            self.usage = Some(usage);
        }
    }

    fn signal_first_output(&mut self, on_first_output: &mut impl FnMut()) {
        if !self.saw_output_frame {
            self.saw_output_frame = true;
            on_first_output();
        }
    }
}

fn strip_leading_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .map_or(&[], |start| &bytes[start..])
}

fn trim_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    let bytes = strip_leading_ascii_whitespace(bytes);
    bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(&[], |end| &bytes[..=end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_tracker_validates_complete_and_truncated_streams_for_every_split() {
        let complete = b"8;ext=1\r\ndata: hi\r\n2\r\n\n\n\r\n0\r\nX-Trailer: ok\r\n\r\n";
        for split in 1..=complete.len() {
            let mut tracker = ChunkedFramingTracker::default();
            let mut entity = Vec::new();
            for part in complete.chunks(split) {
                tracker.feed(part, |bytes| entity.extend_from_slice(bytes));
            }
            assert_eq!(tracker.finish(), ChunkedFramingStatus::Complete);
            assert_eq!(entity, b"data: hi\n\n");
        }

        // Every strict prefix before the terminating CRLF is truncated.
        for cut in 0..complete.len() {
            let mut tracker = ChunkedFramingTracker::default();
            tracker.feed(&complete[..cut], |_| {});
            assert_eq!(
                tracker.finish(),
                ChunkedFramingStatus::Truncated,
                "cut at {cut} must be truncated"
            );
        }
    }

    #[test]
    fn sse_inspector_ignores_keepalive_comments_and_signals_on_first_output_frame() {
        let mut inspector = RawSseInspector::default();
        let mut first_output_calls = 0usize;
        inspector.feed(b": keep-alive\n\n: another comment\r\n\r\n", || {
            first_output_calls += 1;
        });
        assert_eq!(
            first_output_calls, 0,
            "keep-alive comments must not trigger FirstOutput"
        );

        inspector.feed(b"data: {\"choices\":[{\"text\":\"hello\"}]}\n\n", || {
            first_output_calls += 1;
        });
        assert_eq!(
            first_output_calls, 1,
            "first parsed output frame must trigger FirstOutput immediately"
        );

        inspector.feed(
            b"data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\ndata: [DONE]\n\n",
            || first_output_calls += 1,
        );
        let outcome = inspector.finish();
        assert_eq!(first_output_calls, 1);
        assert!(outcome.saw_output_frame);
        assert!(outcome.saw_done);
        assert!(!outcome.saw_error);
        let usage = outcome.usage.expect("usage extracted");
        assert_eq!(usage.prompt_tokens, Some(5));
        assert_eq!(usage.completion_tokens, Some(3));
    }

    #[test]
    fn sse_inspector_discards_unterminated_final_event_at_eof() {
        for truncated in [
            &b"data: [DONE]"[..],
            &b"data: [DONE]\n"[..],
            &b"data: [DONE]\r\n"[..],
        ] {
            let mut inspector = RawSseInspector::default();
            let mut first_output_calls = 0usize;
            inspector.feed(truncated, || first_output_calls += 1);
            let outcome = inspector.finish();
            assert_eq!(first_output_calls, 0);
            assert!(
                !outcome.saw_done,
                "unterminated {truncated:?} must not mark saw_done"
            );
        }
    }

    #[test]
    fn sse_inspector_detects_openai_and_event_errors_without_first_output() {
        for payload in [
            &b"data: {\"error\":{\"message\":\"backend exploded\"}}\n\ndata: [DONE]\n\n"[..],
            &b"event: error\ndata: {\"message\":\"boom\"}\n\n"[..],
        ] {
            let mut inspector = RawSseInspector::default();
            let mut first_output_calls = 0usize;
            inspector.feed(payload, || first_output_calls += 1);
            let outcome = inspector.finish();
            assert_eq!(first_output_calls, 0);
            assert!(outcome.saw_error);
            assert!(!outcome.saw_output_frame);
        }

        // A later `event:` line in the same event block overwrites `event: error`.
        let mut inspector = RawSseInspector::default();
        let mut first_output_calls = 0usize;
        inspector.feed(
            b"event: error\nevent: message\ndata: {\"choices\":[{\"text\":\"ok\"}]}\n\n",
            || first_output_calls += 1,
        );
        let outcome = inspector.finish();
        assert_eq!(first_output_calls, 1);
        assert!(!outcome.saw_error);
        assert!(outcome.saw_output_frame);
    }
}
