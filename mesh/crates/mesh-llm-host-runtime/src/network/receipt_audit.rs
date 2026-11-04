//! Audits of the turns this node's peers serve with receipts, when
//! `[receipts] audit` is on or the request's trust policy requires receipts
//! (`mesh/docs/design/RECEIPTS.md`, `network/request_trust.rs`).
//!
//! A chat completion a local client sends through this node to a peer names
//! a fresh receipts VRF key (`x-receipt-client-key`), so a peer that issues
//! receipts binds the turn to it and returns the signed receipt with the
//! response. The VRF of the turn root picks about one turn in ten
//! ([`AUDIT_RATE`]) to audit, so the peer cannot tell which. The audit runs
//! off the request path on one worker thread: it challenges the peer within
//! the challenge window, reads the bytes the checks need from the claimed
//! Hugging Face GGUF by range, and checks the opening on the CPU.
//!
//! A turn whose audit fails blocks its peer locally for seven days, with
//! the failure kept as evidence that `skippy recheck-receipt` checks again
//! from public data. A turn that cannot be audited (no receipt, a model that
//! is not one Hugging Face file, a check this node cannot run, a busy
//! worker, a peer that does not answer the challenge) is no audit: it says
//! nothing about the peer.

use iroh::EndpointId;
use skippy_receipts::vrf;

use crate::mesh;
use crate::network::http_framing::http_header_terminator;

mod collector;
mod opener;
mod worker;

/// The share of receipted turns this node audits.
pub(crate) const AUDIT_RATE: f64 = 0.1;
const CLIENT_KEY_HEADER: &str = "x-receipt-client-key";
const CHAT_PATH: &str = "/v1/chat/completions";

/// A turn this node watches for a receipt to audit.
pub(crate) struct WatchedTurn {
    node: mesh::Node,
    host: EndpointId,
    key: vrf::SecretKey,
    /// The model the request named, which the challenge names too.
    model: String,
    /// The last user message's text.
    prompt: Option<String>,
    /// The request asked for something that changes the text from the
    /// sampled tokens' text: stop strings, tools, a response format or
    /// several choices.
    not_plain_text: bool,
}

/// What forwarding a watched request needs: the request naming this node's
/// key, and the observer that collects the response for the audit.
pub(crate) struct Watch {
    pub(crate) request: Vec<u8>,
    pub(crate) observer: std::sync::Arc<dyn skippy_inference_api::wire_bytes::WireBytesObserver>,
}

/// When this node audits the turn `request` asks `host` for: the request
/// to forward instead, and the observer to attach to the client stream.
/// `None` for a request this node leaves alone: auditing is off and its
/// trust policy does not require receipts, it is not a chat completion, or
/// its client named a receipt key of its own.
pub(crate) async fn watch(node: &mesh::Node, host: EndpointId, request: &[u8]) -> Option<Watch> {
    let receipts = node.config_state.lock().await.config().receipts.clone();
    // a malformed policy never reaches a peer; were it to, it is audited
    let required =
        crate::network::request_trust::requires_receipts(request, receipts.require).unwrap_or(true);
    if !receipts.audit && !required {
        return None;
    }
    let parsed = ChatRequest::parse(request)?;
    let key = vrf::SecretKey::generate(&mut rand::rng());
    let request = with_header(
        request,
        parsed.header_end,
        CLIENT_KEY_HEADER,
        &hex::encode(key.public()),
    );
    let turn = WatchedTurn {
        node: node.clone(),
        host,
        key,
        model: parsed.model,
        prompt: parsed.prompt,
        not_plain_text: parsed.not_plain_text,
    };
    Some(Watch {
        request,
        observer: std::sync::Arc::new(collector::Collector::new(turn)),
    })
}

/// What the audit needs from a chat completion request.
struct ChatRequest {
    header_end: usize,
    model: String,
    prompt: Option<String>,
    not_plain_text: bool,
}

impl ChatRequest {
    fn parse(raw: &[u8]) -> Option<Self> {
        let mut headers = [httparse::EMPTY_HEADER; 128];
        let mut request = httparse::Request::new(&mut headers);
        let httparse::Status::Complete(header_end) = request.parse(raw).ok()? else {
            return None;
        };
        let path = request.path?.split('?').next()?;
        if request.method? != "POST" || path != CHAT_PATH {
            return None;
        }
        if request
            .headers
            .iter()
            .any(|h| h.name.eq_ignore_ascii_case(CLIENT_KEY_HEADER))
        {
            return None;
        }
        let body: serde_json::Value = serde_json::from_slice(&raw[header_end..]).ok()?;
        let present = |field: &str| body.get(field).is_some_and(|v| !v.is_null());
        let not_plain_text = present("stop")
            || present("tools")
            || present("response_format")
            || body["n"].as_u64().is_some_and(|n| n > 1);
        Some(Self {
            header_end,
            model: body["model"].as_str()?.to_owned(),
            prompt: last_user_text(&body["messages"]),
            not_plain_text,
        })
    }
}

/// The text of the last user message: its string content, or its text
/// parts joined.
fn last_user_text(messages: &serde_json::Value) -> Option<String> {
    let message = messages
        .as_array()?
        .iter()
        .rev()
        .find(|m| m["role"] == "user")?;
    match &message["content"] {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

/// `raw` with `name: value` added as the last header.
fn with_header(raw: &[u8], header_end: usize, name: &str, value: &str) -> Vec<u8> {
    // the head ends with an empty line, in the line ending the request
    // uses; the header goes before it
    let line_ending = http_header_terminator(&raw[..header_end]).map_or(&b"\r\n"[..], |(_, e)| e);
    let at = header_end - line_ending.len();
    let mut out = Vec::with_capacity(raw.len() + name.len() + value.len() + 4);
    out.extend_from_slice(&raw[..at]);
    out.extend_from_slice(format!("{name}: {value}").as_bytes());
    out.extend_from_slice(line_ending);
    out.extend_from_slice(&raw[at..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: &str, extra: &str) -> Vec<u8> {
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{extra}Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    #[test]
    fn a_chat_request_is_parsed_and_gets_the_key_header() {
        let raw = request(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},{"role":"user","content":[{"type":"text","text":"Paris?"}]}]}"#,
            "",
        );
        let parsed = ChatRequest::parse(&raw).unwrap();
        assert_eq!(parsed.model, "m");
        assert_eq!(parsed.prompt.as_deref(), Some("Paris?"));
        assert!(!parsed.not_plain_text);
        let with = with_header(&raw, parsed.header_end, CLIENT_KEY_HEADER, "ab");
        let text = String::from_utf8(with).unwrap();
        assert!(
            text.contains("\r\nx-receipt-client-key: ab\r\n\r\n{"),
            "{text}"
        );
        let mut headers = [httparse::EMPTY_HEADER; 16];
        let mut reparsed = httparse::Request::new(&mut headers);
        assert!(reparsed.parse(text.as_bytes()).unwrap().is_complete());
    }

    #[test]
    fn requests_the_audit_leaves_alone() {
        let own_key = request(
            r#"{"model":"m","messages":[]}"#,
            "X-Receipt-Client-Key: cd\r\n",
        );
        assert!(ChatRequest::parse(&own_key).is_none());
        let embeddings = b"POST /v1/embeddings HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        assert!(ChatRequest::parse(embeddings).is_none());
        let stop = request(r#"{"model":"m","messages":[],"stop":["\n"]}"#, "");
        assert!(ChatRequest::parse(&stop).unwrap().not_plain_text);
    }
}
