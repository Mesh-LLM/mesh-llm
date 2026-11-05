//! Reads a peer's chat completion as it is relayed to the client: the
//! answered text and the receipt, from a JSON body or an SSE stream. It
//! sees the entity bytes the client receives, so the audit checks the
//! text the client got.

use std::sync::Mutex;

use skippy_inference_api::wire_bytes::{WireBytesCommitment, WireBytesObserver};

use super::WatchedTurn;

/// The most body a collector keeps: a JSON body, or one SSE line.
const MAX_KEPT: usize = 64 << 20;

#[derive(Default)]
pub(super) struct Reply {
    pub(super) content: String,
    pub(super) reasoning: String,
    /// The answer called tools or stopped otherwise than the text shows,
    /// so the text is not the sampled tokens' text.
    pub(super) not_plain_text: bool,
    pub(super) receipt: Option<serde_json::Value>,
}

impl Reply {
    fn read_message(&mut self, message: &serde_json::Value) {
        let text = |field: &str| message[field].as_str().unwrap_or_default();
        self.content.push_str(text("content"));
        self.reasoning.push_str(text("reasoning_content"));
        if message
            .get("tool_calls")
            .is_some_and(|calls| !calls.is_null())
        {
            self.not_plain_text = true;
        }
    }

    fn read_event(&mut self, value: &serde_json::Value) {
        let choices = value["choices"].as_array();
        if choices.is_some_and(|c| c.len() > 1) {
            self.not_plain_text = true;
        }
        if let Some(choice) = choices.and_then(|c| c.first()) {
            self.read_message(&choice["delta"]);
            self.read_message(&choice["message"]);
        }
        if let Some(receipt) = value.get("receipt").filter(|r| !r.is_null()) {
            self.receipt = Some(receipt.clone());
        }
    }
}

#[derive(Default)]
struct State {
    status: Option<u16>,
    /// The first byte decides: `{` is a JSON body, else SSE.
    sse: Option<bool>,
    kept: Vec<u8>,
    overflow: bool,
    reply: Reply,
}

impl State {
    fn push(&mut self, mut bytes: &[u8]) {
        if self.sse.is_none() {
            let start = bytes.iter().position(|b| !b.is_ascii_whitespace());
            match start {
                Some(start) => self.sse = Some(bytes[start] != b'{'),
                None => return,
            }
        }
        if self.sse == Some(false) {
            self.keep(bytes);
            return;
        }
        while let Some(end) = bytes.iter().position(|&b| b == b'\n') {
            self.keep(&bytes[..end]);
            let line = std::mem::take(&mut self.kept);
            if !std::mem::take(&mut self.overflow) {
                self.read_line(&line);
            }
            bytes = &bytes[end + 1..];
        }
        self.keep(bytes);
    }

    fn keep(&mut self, bytes: &[u8]) {
        if self.kept.len() + bytes.len() > MAX_KEPT {
            self.overflow = true;
            self.kept.clear();
        } else if !self.overflow {
            self.kept.extend_from_slice(bytes);
        }
    }

    fn read_line(&mut self, line: &[u8]) {
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        let data = data.trim_ascii();
        if data == b"[DONE]" {
            return;
        }
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) {
            self.reply.read_event(&value);
        }
    }

    fn finish(mut self) -> Option<Reply> {
        if self.status != Some(200) || self.overflow {
            return None;
        }
        match self.sse {
            Some(false) => {
                let value: serde_json::Value = serde_json::from_slice(&self.kept).ok()?;
                self.reply.read_event(&value);
            }
            Some(true) => {
                let line = std::mem::take(&mut self.kept);
                self.read_line(&line);
            }
            None => return None,
        }
        Some(self.reply)
    }
}

/// Watches one relayed response for the turn's receipt, and hands the turn
/// to the audit worker when the response ends whole.
pub(super) struct Collector {
    state: Mutex<Option<State>>,
    turn: Mutex<Option<WatchedTurn>>,
}

impl Collector {
    pub(super) fn new(turn: WatchedTurn) -> Self {
        Self {
            state: Mutex::new(Some(State::default())),
            turn: Mutex::new(Some(turn)),
        }
    }

    fn with_state(&self, f: impl FnOnce(&mut State)) {
        if let Ok(mut state) = self.state.lock()
            && let Some(state) = state.as_mut()
        {
            f(state);
        }
    }
}

impl WireBytesObserver for Collector {
    fn response_status(&self, status_code: u16) {
        self.with_state(|state| state.status = Some(status_code));
    }

    fn try_chunk(&self, _offset: u64, bytes: &[u8]) -> bool {
        self.with_state(|state| state.push(bytes));
        true
    }

    fn finish(&self, commitment: WireBytesCommitment) {
        let state = self.state.lock().ok().and_then(|mut s| s.take());
        let turn = self.turn.lock().ok().and_then(|mut t| t.take());
        let (Some(state), Some(turn)) = (state, turn) else {
            return;
        };
        if commitment.incomplete.is_some() {
            return;
        }
        match state.finish() {
            Some(reply) => super::worker::submit(turn, reply),
            None => tracing::debug!("receipt audit: the response was not a whole chat completion"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bytes: &[&[u8]]) -> Option<Reply> {
        let mut state = State {
            status: Some(200),
            ..State::default()
        };
        for chunk in bytes {
            state.push(chunk);
        }
        state.finish()
    }

    #[test]
    fn a_json_body_gives_its_text_and_receipt() {
        let body = br#"{"choices":[{"message":{"content":"Paris.","reasoning_content":"hm"}}],"receipt":{"receipt_id":"r"}}"#;
        let reply = read(&[&body[..10], &body[10..]]).unwrap();
        assert_eq!(reply.content, "Paris.");
        assert_eq!(reply.reasoning, "hm");
        assert_eq!(reply.receipt.unwrap()["receipt_id"], "r");
        assert!(!reply.not_plain_text);
    }

    #[test]
    fn a_stream_joins_its_deltas_across_chunk_boundaries() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Par\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"is.\"}}]}\n\n",
            "data: {\"choices\":[],\"receipt\":{\"receipt_id\":\"r\"}}\n\n",
            "data: [DONE]\n\n",
        )
        .as_bytes();
        let reply = read(&[&stream[..30], &stream[30..71], &stream[71..]]).unwrap();
        assert_eq!(reply.content, "Paris.");
        assert_eq!(reply.receipt.unwrap()["receipt_id"], "r");
    }

    #[test]
    fn tool_calls_mark_the_text_as_not_plain() {
        let body = br#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c"}]}}]}"#;
        assert!(read(&[body]).unwrap().not_plain_text);
    }

    #[test]
    fn an_error_status_gives_nothing() {
        let mut state = State {
            status: Some(500),
            ..State::default()
        };
        state.push(br#"{"error":{}}"#);
        assert!(state.finish().is_none());
    }
}
