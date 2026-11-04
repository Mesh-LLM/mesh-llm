//! Reads a peer's chat completion as it is relayed to the client: the
//! answered text and the receipt, from a JSON body or an SSE stream. It
//! sees the entity bytes the client receives, so the audit checks the
//! text the client got.
//!
//! The peer decides how much it sends, so what is kept is bounded: a JSON
//! body or one SSE line by [`MAX_KEPT`], which bounds the receipt too, and
//! the answered text of the whole response by [`MAX_TEXT`]. Text past the
//! bound is not kept and the text check does not run. A line or a body
//! past the bound is not read, and a receipt it held was not seen.

use std::sync::Mutex;

use skippy_inference_api::wire_bytes::{WireBytesCommitment, WireBytesObserver};

use super::WatchedTurn;

/// The most of a JSON body, or of one SSE line, that is read. A receipt
/// names every token of its turn and is the largest thing either holds.
const MAX_KEPT: usize = 16 << 20;
/// The most answered text kept across a response: content and reasoning
/// together.
const MAX_TEXT: usize = 4 << 20;

/// A chat completion a peer answered with.
#[derive(Default)]
pub(super) struct Reply {
    pub(super) content: String,
    pub(super) reasoning: String,
    /// The peer says it stopped at the token limit, where the text may end
    /// inside a character.
    pub(super) cut_short: bool,
    /// Why the text is not the sampled tokens' text for an honest peer:
    /// it called tools, answered several choices, or sent more text than
    /// this node keeps.
    pub(super) text_unavailable: Option<&'static str>,
    pub(super) receipt: Option<serde_json::Value>,
    /// A line or the body was too large to read, so a receipt in it was
    /// not seen.
    pub(super) unread: bool,
    /// The answer reports an error: the peer says itself that it served no
    /// whole turn, and the client was told so.
    pub(super) errored: bool,
}

impl Reply {
    fn push_text(&mut self, reasoning: bool, text: &str) {
        if self.content.len() + self.reasoning.len() + text.len() > MAX_TEXT {
            self.text_unavailable = Some("the answer is longer than this node keeps");
            return;
        }
        match reasoning {
            true => self.reasoning.push_str(text),
            false => self.content.push_str(text),
        }
    }

    fn read_message(&mut self, message: &serde_json::Value) {
        let text = |field: &str| message[field].as_str().unwrap_or_default();
        self.push_text(false, text("content"));
        self.push_text(true, text("reasoning_content"));
        let called = |field: &str| message.get(field).is_some_and(|calls| !calls.is_null());
        if called("tool_calls") || called("function_call") {
            self.text_unavailable = Some("the answer called tools");
        }
    }

    fn read_event(&mut self, value: &serde_json::Value) {
        let choices = value["choices"].as_array();
        if choices.is_some_and(|c| c.len() > 1) {
            self.text_unavailable = Some("the answer holds several choices");
        }
        if let Some(choice) = choices.and_then(|c| c.first()) {
            self.read_message(&choice["delta"]);
            self.read_message(&choice["message"]);
            match choice["finish_reason"].as_str() {
                None | Some("stop") => {}
                Some("length") => self.cut_short = true,
                Some(_) => self.text_unavailable = Some("the answer did not end as plain text"),
            }
        }
        if let Some(receipt) = value.get("receipt").filter(|r| !r.is_null()) {
            self.receipt = Some(receipt.clone());
        }
        if value.get("error").is_some_and(|error| !error.is_null()) {
            self.errored = true;
        }
    }
}

/// What a relayed response gave the audit.
pub(super) enum Collected {
    /// A whole answer with status 200.
    Reply(Reply),
    /// No turn to audit, and why: the response broke off, or was an error.
    NotATurn(String),
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
            self.end_line();
            bytes = &bytes[end + 1..];
        }
        self.keep(bytes);
    }

    fn keep(&mut self, bytes: &[u8]) {
        if self.kept.len() + bytes.len() > MAX_KEPT {
            self.overflow = true;
            self.kept = Vec::new();
        } else if !self.overflow {
            self.kept.extend_from_slice(bytes);
        }
    }

    fn end_line(&mut self) {
        let line = std::mem::take(&mut self.kept);
        match std::mem::take(&mut self.overflow) {
            true => self.reply.unread = true,
            false => self.read_line(&line),
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

    fn finish(mut self) -> Collected {
        match (self.status, self.sse) {
            (Some(200), Some(true)) => self.end_line(),
            (Some(200), Some(false)) if self.overflow => self.reply.unread = true,
            (Some(200), Some(false)) => {
                // a body that is not JSON holds no receipt, like one without
                if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&self.kept) {
                    self.reply.read_event(&value);
                }
            }
            (Some(200), None) => return Collected::NotATurn("the response had no body".into()),
            (Some(status), _) => {
                return Collected::NotATurn(format!("the peer answered with status {status}"));
            }
            (None, _) => return Collected::NotATurn("the response had no status".into()),
        }
        if self.reply.errored {
            return Collected::NotATurn("the peer's answer reported an error".into());
        }
        if self.reply.unread {
            self.reply.text_unavailable = Some("part of the response was too large to read");
        }
        Collected::Reply(self.reply)
    }
}

/// Watches one relayed response for the turn's receipt, and hands the turn
/// to the audit when the response ends.
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
        let collected = if commitment.incomplete.is_some() {
            Collected::NotATurn("the response did not end whole".into())
        } else if !commitment.side_stream_complete {
            Collected::NotATurn("this node did not read the whole response".into())
        } else {
            state.finish()
        };
        // a verdict that needs more than a log line is acted on in its own
        // time: the response is done, and nothing waits for it here
        drop(super::intake::receive(turn, collected));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(status: u16, bytes: &[&[u8]]) -> Collected {
        let mut state = State {
            status: Some(status),
            ..State::default()
        };
        for chunk in bytes {
            state.push(chunk);
        }
        state.finish()
    }

    fn read(bytes: &[&[u8]]) -> Reply {
        match collect(200, bytes) {
            Collected::Reply(reply) => reply,
            Collected::NotATurn(why) => panic!("not a turn: {why}"),
        }
    }

    fn delta(content: &str) -> String {
        format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{content}\"}}}}]}}\n\n")
    }

    #[test]
    fn a_json_body_gives_its_text_and_receipt() {
        let body = br#"{"choices":[{"message":{"content":"Paris.","reasoning_content":"hm"},"finish_reason":"stop"}],"receipt":{"receipt_id":"r"}}"#;
        let reply = read(&[&body[..10], &body[10..]]);
        assert_eq!(reply.content, "Paris.");
        assert_eq!(reply.reasoning, "hm");
        assert_eq!(reply.receipt.unwrap()["receipt_id"], "r");
        assert_eq!(reply.text_unavailable, None);
        assert!(!reply.cut_short && !reply.unread);
    }

    #[test]
    fn a_stream_joins_its_deltas_across_chunk_boundaries() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Par\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"is.\"},\"finish_reason\":\"length\"}]}\n\n",
            "data: {\"choices\":[],\"receipt\":{\"receipt_id\":\"r\"}}\n\n",
            "data: [DONE]\n\n",
        )
        .as_bytes();
        let reply = read(&[&stream[..30], &stream[30..71], &stream[71..]]);
        assert_eq!(reply.content, "Paris.");
        assert_eq!(reply.receipt.unwrap()["receipt_id"], "r");
        assert!(reply.cut_short, "the peer stopped at the token limit");
        assert_eq!(reply.text_unavailable, None);
    }

    #[test]
    fn answers_that_are_not_plain_text_have_no_text_check() {
        for body in [
            &br#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c"}]}}]}"#[..],
            br#"{"choices":[{"message":{"content":"a"}},{"message":{"content":"b"}}]}"#,
            br#"{"choices":[{"message":{"content":"a"},"finish_reason":"content_filter"}]}"#,
        ] {
            assert!(read(&[body]).text_unavailable.is_some());
        }
    }

    #[test]
    fn an_answer_that_reports_an_error_is_not_a_turn() {
        // a stream that began with status 200 and then failed, and a body
        // that says so: the client was told, and no turn was served
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Par\"}}]}\n\n",
            "data: {\"error\":{\"message\":\"context overflow\"}}\n\n",
        )
        .as_bytes();
        let body = br#"{"error":{"message":"the model was unloaded"}}"#;
        for answer in [stream, &body[..]] {
            let Collected::NotATurn(why) = collect(200, &[answer]) else {
                panic!("an error is a turn");
            };
            assert!(why.contains("reported an error"), "{why}");
        }
        // an answer that names no error is a turn, and owes its receipt
        let reply = read(&[br#"{"choices":[],"error":null}"#]);
        assert!(reply.receipt.is_none());
    }

    #[test]
    fn a_response_that_is_no_answer_is_not_a_turn() {
        for status in [500, 404, 429] {
            let Collected::NotATurn(why) = collect(status, &[br#"{"error":{}}"#]) else {
                panic!("status {status} is a turn");
            };
            assert!(why.contains(&status.to_string()), "{why}");
        }
        assert!(matches!(collect(200, &[]), Collected::NotATurn(_)));
        // a 200 that is not JSON is an answer without a receipt
        let reply = read(&[b"{ not json"]);
        assert!(reply.receipt.is_none() && !reply.unread);
    }

    #[test]
    fn the_answered_text_is_capped_across_the_whole_stream() {
        // each line is small; together they are past the cap
        let piece = "x".repeat(1 << 20);
        let line = delta(&piece);
        let mut state = State {
            status: Some(200),
            ..State::default()
        };
        for _ in 0..6 {
            state.push(line.as_bytes());
        }
        state.push(b"data: {\"choices\":[],\"receipt\":{\"receipt_id\":\"r\"}}\n\n");
        let Collected::Reply(reply) = state.finish() else {
            panic!("a turn");
        };
        assert!(reply.content.len() <= MAX_TEXT, "{}", reply.content.len());
        assert_eq!(reply.content.len(), 4 << 20);
        assert!(reply.text_unavailable.is_some(), "the text check is off");
        // the receipt after the text is still read
        assert_eq!(reply.receipt.unwrap()["receipt_id"], "r");
        assert!(!reply.unread);
    }

    #[test]
    fn reasoning_counts_toward_the_text_cap() {
        let piece = "y".repeat(3 << 20);
        let line = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":\"{piece}\"}}}}]}}\n\n"
        );
        let reply = read(&[line.as_bytes(), line.as_bytes()]);
        assert_eq!(reply.reasoning.len(), 3 << 20);
        assert!(reply.text_unavailable.is_some());
    }

    #[test]
    fn a_line_past_the_bound_is_not_read_and_says_so() {
        // a receipt of a size no turn has: the line is dropped, in pieces,
        // without being kept
        let mut state = State {
            status: Some(200),
            ..State::default()
        };
        state.push(delta("Paris.").as_bytes());
        state.push(b"data: {\"choices\":[],\"receipt\":{\"tokens\":[");
        let filler = vec![b'1'; 1 << 20];
        for _ in 0..17 {
            state.push(&filler);
            assert!(state.kept.len() <= MAX_KEPT);
        }
        assert!(state.kept.is_empty(), "nothing of the line is kept");
        state.push(b"]}}\n\ndata: [DONE]\n\n");
        let Collected::Reply(reply) = state.finish() else {
            panic!("a turn");
        };
        assert!(reply.unread && reply.receipt.is_none());
        assert!(reply.text_unavailable.is_some());
        assert_eq!(reply.content, "Paris.");
    }

    #[test]
    fn a_json_body_past_the_bound_is_not_read() {
        let mut state = State {
            status: Some(200),
            ..State::default()
        };
        state.push(b"{\"receipt\":{\"tokens\":[");
        let filler = vec![b'1'; 1 << 20];
        for _ in 0..17 {
            state.push(&filler);
        }
        let Collected::Reply(reply) = state.finish() else {
            panic!("a turn");
        };
        assert!(reply.unread && reply.receipt.is_none());
    }
}
