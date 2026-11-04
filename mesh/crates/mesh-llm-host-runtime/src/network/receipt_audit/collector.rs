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
    /// Why the text is not the sampled tokens' text for an honest peer: it
    /// sent more text than this node keeps, or a part too large to read.
    pub(super) text_unavailable: Option<&'static str>,
    /// How the answer is other than one choice of plain text, when it is:
    /// it called tools, holds several choices, or ended another way. Only
    /// a request that offered tools or asked for several choices allows
    /// that (`intake::contradiction`).
    pub(super) shape: Option<&'static str>,
    pub(super) receipt: Option<serde_json::Value>,
    /// A line or the body was too large to read, so a receipt in it was
    /// not seen.
    pub(super) unread: bool,
    /// The answer reports an error with status 200. A request for a receipt
    /// owes one all the same (`intake`): were it no turn, a peer would add
    /// an error to every turn it cheated on.
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
        // an empty list of calls is no call
        let called = |field: &str| match message.get(field) {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Array(calls)) => !calls.is_empty(),
            Some(_) => true,
        };
        if called("tool_calls") || called("function_call") {
            self.shape = Some("the answer called tools");
        }
    }

    fn read_event(&mut self, value: &serde_json::Value) {
        let choices = value["choices"].as_array();
        if choices.is_some_and(|c| c.len() > 1) {
            self.shape = Some("the answer holds several choices");
        }
        if let Some(choice) = choices.and_then(|c| c.first()) {
            self.read_message(&choice["delta"]);
            self.read_message(&choice["message"]);
            match choice["finish_reason"].as_str() {
                None | Some("stop") => {}
                Some("length") => self.cut_short = true,
                Some(_) => self.shape = Some("the answer did not end as plain text"),
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
    /// The stream's last event, `[DONE]`, came.
    done: bool,
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

    /// Whether the response was a stream that ended with its last event.
    /// What came after it is not read, and a transport that broke off
    /// after it lost nothing of the turn.
    fn ended_its_stream(&self) -> bool {
        self.sse == Some(true) && self.done
    }

    fn read_line(&mut self, line: &[u8]) {
        if self.done {
            return;
        }
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        let data = data.trim_ascii();
        if data == b"[DONE]" {
            self.done = true;
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
        // a verdict that needs more than a log line is acted on in its own
        // time: the response is done, and nothing waits for it here
        drop(super::intake::receive(turn, collected(state, &commitment)));
    }
}

/// What a response came to, once the bytes `state` read ended as
/// `commitment` says.
fn collected(state: State, commitment: &WireBytesCommitment) -> Collected {
    if !commitment.side_stream_complete {
        Collected::NotATurn("this node did not read the whole response".into())
    } else if commitment.incomplete.is_some() && !state.ended_its_stream() {
        Collected::NotATurn("the response did not end whole".into())
    } else {
        state.finish()
    }
}

/// The collector as the watch attaches it, without a turn behind it: what
/// a relay gives the audit, for the tests of the relays.
#[cfg(test)]
pub(crate) mod probe {
    use std::sync::Mutex;

    use skippy_inference_api::wire_bytes::{WireBytesCommitment, WireBytesObserver};

    use super::{Collected, State, collected};

    #[derive(Default)]
    pub(crate) struct CollectorProbe {
        state: Mutex<State>,
        collected: Mutex<Option<Collected>>,
    }

    impl CollectorProbe {
        /// The receipt the response carried, or why it was no turn; `None`
        /// while the response has not ended.
        pub(crate) fn receipt(&self) -> Option<Result<Option<serde_json::Value>, String>> {
            let collected = self.collected.lock().unwrap();
            Some(match collected.as_ref()? {
                Collected::Reply(reply) => Ok(reply.receipt.clone()),
                Collected::NotATurn(why) => Err(why.clone()),
            })
        }
    }

    impl WireBytesObserver for CollectorProbe {
        fn response_status(&self, status_code: u16) {
            self.state.lock().unwrap().status = Some(status_code);
        }

        fn try_chunk(&self, _offset: u64, bytes: &[u8]) -> bool {
            self.state.lock().unwrap().push(bytes);
            true
        }

        fn finish(&self, commitment: WireBytesCommitment) {
            let state = std::mem::take(&mut *self.state.lock().unwrap());
            *self.collected.lock().unwrap() = Some(collected(state, &commitment));
        }
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
    fn answers_that_are_not_one_choice_of_plain_text_say_so() {
        for body in [
            &br#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c"}]}}]}"#[..],
            br#"{"choices":[{"message":{"content":"a","function_call":{"name":"f"}}}]}"#,
            br#"{"choices":[{"message":{"content":"a"}},{"message":{"content":"b"}}]}"#,
            br#"{"choices":[{"message":{"content":"a"},"finish_reason":"content_filter"}]}"#,
            br#"{"choices":[{"message":{"content":"a"},"finish_reason":"tool_calls"}]}"#,
        ] {
            let reply = read(&[body]);
            assert!(reply.shape.is_some(), "{}", String::from_utf8_lossy(body));
            assert_eq!(reply.text_unavailable, None);
        }
        // an empty list of calls, or none, is plain text
        for body in [
            &br#"{"choices":[{"message":{"content":"a","tool_calls":[]},"finish_reason":"stop"}]}"#
                [..],
            br#"{"choices":[{"message":{"content":"a","tool_calls":null,"function_call":null}}]}"#,
            br#"{"choices":[{"message":{"content":"a"},"finish_reason":"length"}]}"#,
        ] {
            assert_eq!(
                read(&[body]).shape,
                None,
                "{}",
                String::from_utf8_lossy(body)
            );
        }
    }

    #[test]
    fn an_answer_that_reports_an_error_is_still_a_turn() {
        // a stream that began with status 200 and then failed, and a body
        // that says so: a request for a receipt owes one all the same
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Par\"}}]}\n\n",
            "data: {\"error\":{\"message\":\"context overflow\"}}\n\n",
        )
        .as_bytes();
        let body = br#"{"error":{"message":"the model was unloaded"}}"#;
        for answer in [stream, &body[..]] {
            let reply = read(&[answer]);
            assert!(reply.errored && reply.receipt.is_none());
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
