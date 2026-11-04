//! What an audit needs from the chat completion request this node forwards
//! to a peer: the model and the id it asks under, the sampling parameters
//! it sets, the prompt, and whether the response's text will be exactly the
//! text of the sampled tokens.
//!
//! The request is the local client's, so nothing here is hostile. It is
//! read as the peer will read it: the bytes that are forwarded.

use serde_json::Value;
use skippy_receipts::sampler_request::RequestedSampling;

use crate::network::http_framing::chunked::try_decode_chunked_body;
use crate::network::http_framing::http_header_terminator;

pub(super) const CLIENT_KEY_HEADER: &str = "x-receipt-client-key";
/// The seed a receipt request asks a sampled turn to draw with
/// (`skippy_inference_api::RECEIPT_SEED_HEADER`).
pub(super) const SEED_HEADER: &str = "x-receipt-seed";
const REQUEST_ID_HEADER: &str = "x-request-id";
const CHAT_PATH: &str = "/v1/chat/completions";

/// Request fields that turn on a penalty, or a sampler beyond temperature,
/// top-k, top-p and min-p.
const PENALTY_FIELDS: [&str; 16] = [
    "repeat_penalty",
    "repetition_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "typical_p",
    "top_nsigma",
    "top_n_sigma",
    "dry",
    "xtc",
    "mirostat",
    "mirostat_mode",
    "mirostat_entropy",
    "mirostat_learning_rate",
    "dynatemp_range",
    "dynatemp_exponent",
];
/// Request fields that make the node shape its answer: a format it must
/// parse into, or tools it may call. Each makes it build a grammar too.
const SHAPING_FIELDS: [&str; 6] = [
    "json_schema",
    "response_format",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
];
/// Request fields that offer the model tools it may call.
const TOOL_FIELDS: [&str; 4] = ["tools", "tool_choice", "functions", "function_call"];
/// Request fields that set how many tokens the model may reason for.
const REASONING_BUDGET_FIELDS: [&str; 4] = [
    "reasoning_budget_tokens",
    "thinking_budget_tokens",
    "reasoning_budget",
    "thinking_budget",
];
/// Request fields that change the reasoning budget another way, so the
/// tokens the request allowed are not one number this node can name.
const OTHER_REASONING_FIELDS: [&str; 5] = [
    "reasoning",
    "reasoning_effort",
    "enable_thinking",
    "thinking",
    "chat_template_kwargs",
];
/// Reasoning formats whose answer is the sampled text with its `<think>`
/// markers removed, which the text check allows for.
const THINK_FORMATS: [&str; 3] = ["none", "auto", "deepseek"];

/// A chat completion request this node audits the answer of.
pub(super) struct ChatRequest {
    pub(super) header_end: usize,
    pub(super) model: String,
    /// The id the request asks under, which the peer issues its receipt
    /// under: the one canonical `x-request-id`.
    pub(super) asked_id: Option<String>,
    /// The last user message's text, when the receipt's prompt tokens must
    /// hold it.
    pub(super) prompt: Option<String>,
    pub(super) sampling: RequestedSampling,
    /// The seed the client's own receipt request named, when it named one
    /// the peer will read.
    pub(super) seed_header: Option<u32>,
    /// Why the response's text is not exactly the sampled tokens' text for
    /// an honest peer, when the request makes it so. The text check then
    /// does not run: it is unavailable, never a failure.
    pub(super) text_unavailable: Option<&'static str>,
    /// Why an honest peer's answer may call tools or hold several choices,
    /// when the request allows it. An answer of that shape to a request
    /// that does not allow it contradicts the request.
    pub(super) shaped_answer: Option<&'static str>,
}

impl ChatRequest {
    /// The request `raw` holds. `Ok(None)` for one this node does not
    /// audit: not a chat completion, or one whose client named a receipt
    /// key of its own. An error for a chat completion this node cannot
    /// read, which it cannot watch.
    pub(super) fn read(raw: &[u8]) -> Result<Option<Self>, &'static str> {
        let mut headers = [httparse::EMPTY_HEADER; 128];
        let mut request = httparse::Request::new(&mut headers);
        let Ok(httparse::Status::Complete(header_end)) = request.parse(raw) else {
            return Err("this node cannot read the request's head");
        };
        let path = request.path.and_then(|path| path.split('?').next());
        if request.method != Some("POST") || path != Some(CHAT_PATH) {
            return Ok(None);
        }
        let named = |name: &'static str| {
            request
                .headers
                .iter()
                .filter(move |h| h.name.eq_ignore_ascii_case(name))
        };
        if named(CLIENT_KEY_HEADER).next().is_some() {
            return Ok(None);
        }
        let ids = named(REQUEST_ID_HEADER).map(|h| std::str::from_utf8(h.value).ok());
        let asked_id = skippy_inference_api::parse_single_request_id(ids)
            .map(|id| id.as_uuid().hyphenated().to_string());
        // the peer reads the first such header, and a seed above 0
        let seed_header = named(SEED_HEADER)
            .next()
            .and_then(|h| std::str::from_utf8(h.value).ok())
            .and_then(|text| text.trim().parse::<u32>().ok())
            .filter(|seed| *seed != 0);
        let chunked = named("transfer-encoding").any(|h| {
            std::str::from_utf8(h.value)
                .is_ok_and(|value| value.trim().eq_ignore_ascii_case("chunked"))
        });
        let body = request_body(&raw[header_end..], chunked)
            .ok_or("this node cannot read the request's body as JSON")?;
        let model = body["model"].as_str().ok_or("the request names no model")?;
        Ok(Some(Self {
            header_end,
            model: model.to_owned(),
            asked_id,
            prompt: checkable_prompt(&body["messages"]),
            sampling: requested_sampling(&body),
            seed_header,
            text_unavailable: text_unavailable(&body),
            shaped_answer: shaped_answer(&body),
        }))
    }
}

/// The JSON body of a request, its chunks joined when it came in chunks.
fn request_body(wire: &[u8], chunked: bool) -> Option<Value> {
    if !chunked {
        return serde_json::from_slice(wire).ok();
    }
    // the entity is no longer than its framing, which ingress bounded
    let (_, entity) = try_decode_chunked_body(wire, wire.len()).ok().flatten()?;
    serde_json::from_slice(&entity).ok()
}

/// Why an honest peer's answer to `body` may be other than one choice of
/// plain text: the request offered tools or asked for several choices.
fn shaped_answer(body: &Value) -> Option<&'static str> {
    if any_present(body, &TOOL_FIELDS) {
        Some("the request offered tools, so the answer may call them")
    } else if body["n"].as_u64().is_some_and(|n| n > 1) {
        Some("the request asked for several choices")
    } else {
        None
    }
}

fn present(body: &Value, field: &str) -> bool {
    match body.get(field) {
        None | Some(Value::Null) => false,
        Some(Value::Array(entries)) => !entries.is_empty(),
        Some(Value::String(text)) => !text.is_empty(),
        Some(_) => true,
    }
}

fn any_present(body: &Value, fields: &[&str]) -> bool {
    fields.iter().any(|field| present(body, field))
}

/// The sampling parameters `body` sets. A parameter the request leaves out
/// is `None`: the node's default applies, which this node cannot know.
fn requested_sampling(body: &Value) -> RequestedSampling {
    let number = |field: &str| body.get(field).and_then(Value::as_f64);
    RequestedSampling {
        temperature: number("temperature"),
        top_p: number("top_p"),
        top_k: body.get("top_k").and_then(Value::as_i64),
        min_p: number("min_p"),
        seed: body.get("seed").and_then(Value::as_u64),
        reasoning_budget: reasoning_budget(body),
        grammar: Some(present(body, "grammar") || any_present(body, &SHAPING_FIELDS)),
        penalties: Some(any_present(body, &PENALTY_FIELDS)),
        logit_bias: Some(present(body, "logit_bias")),
        sampler_chain: Some(any_present(body, &["samplers", "sampler_sequence"])),
        ignore_eos: Some(body.get("ignore_eos").and_then(Value::as_bool) == Some(true)),
        // a peer's operator may configure a default that turns a filter
        // off, which this node cannot know: a filter the request left
        // unset is not held against the peer. The draw is replayed either
        // way, so the peer gains no choice of token from it.
        unset_filters_on: None,
    }
}

/// The reasoning tokens the request allows, when it says so with one
/// number and nothing else that changes the budget.
fn reasoning_budget(body: &Value) -> Option<i64> {
    if any_present(body, &OTHER_REASONING_FIELDS) {
        return None;
    }
    let mut set = REASONING_BUDGET_FIELDS
        .iter()
        .filter(|field| present(body, field));
    let (field, None) = (set.next()?, set.next()) else {
        return None;
    };
    body[*field].as_i64().filter(|tokens| *tokens >= -1)
}

/// Why an honest peer's answer to `body` need not be exactly the text of
/// the tokens it sampled.
fn text_unavailable(body: &Value) -> Option<&'static str> {
    let messages = body["messages"].as_array().map_or(&[][..], Vec::as_slice);
    let format = body.get("reasoning_format").and_then(Value::as_str);
    if present(body, "stop") {
        Some("the request set stop strings, which cut the text")
    } else if any_present(body, &SHAPING_FIELDS) {
        Some("the request set tools or a response format, which shape the text")
    } else if body["n"].as_u64().is_some_and(|n| n > 1) {
        Some("the request asked for several choices")
    } else if body["logprobs"].as_bool() == Some(true) || present(body, "top_logprobs") {
        Some("the request asked for log probabilities")
    } else if continues_an_answer(body, messages) {
        Some(
            "the request continues an assistant message, so the answer need not start at the sampled tokens",
        )
    } else if messages.iter().any(has_media) {
        Some("the request holds content that is not text")
    } else if format.is_some_and(|format| !THINK_FORMATS.contains(&format)) {
        Some("the request asked for a reasoning format the text check does not know")
    } else if body["echo"].as_bool() == Some(true) {
        Some("the request asked for the prompt to be echoed")
    } else {
        None
    }
}

/// Whether the model continues a message the client started: the last
/// message is the assistant's, or the request says so.
fn continues_an_answer(body: &Value, messages: &[Value]) -> bool {
    let last = messages.last();
    last.is_some_and(|m| m["role"] == "assistant" || m["prefix"].as_bool() == Some(true))
        || body["continue_final_message"].as_bool() == Some(true)
        || body["add_generation_prompt"].as_bool() == Some(false)
}

/// Whether a message holds a content part that is not text: an image, a
/// sound, a file.
fn has_media(message: &Value) -> bool {
    message["content"].as_array().is_some_and(|parts| {
        parts
            .iter()
            .any(|part| part.get("text").and_then(Value::as_str).is_none())
    })
}

/// The last user message's text, when the text of the receipt's prompt
/// tokens must hold it: one text in characters every tokenizer keeps. A
/// tokenizer may write other characters another way, and a template may put
/// its own text between parts, so such a prompt is not checked.
fn checkable_prompt(messages: &Value) -> Option<String> {
    let message = messages
        .as_array()?
        .iter()
        .rev()
        .find(|m| m["role"] == "user")?;
    let text = match &message["content"] {
        Value::String(text) => text.as_str(),
        Value::Array(parts) if parts.len() == 1 => parts[0]["text"].as_str()?,
        _ => return None,
    };
    let kept = |c: char| c.is_ascii_graphic() || c.is_ascii_whitespace();
    text.chars().all(kept).then(|| text.to_owned())
}

/// `raw` with each `name: value` of `headers` added after its last header.
pub(super) fn with_headers(raw: &[u8], header_end: usize, headers: &[(&str, String)]) -> Vec<u8> {
    // the head ends with an empty line, in the line ending the request
    // uses; the headers go before it
    let line_ending = http_header_terminator(&raw[..header_end]).map_or(&b"\r\n"[..], |(_, e)| e);
    let at = header_end - line_ending.len();
    let mut out = Vec::with_capacity(raw.len() + 128 * headers.len());
    out.extend_from_slice(&raw[..at]);
    for (name, value) in headers {
        out.extend_from_slice(format!("{name}: {value}").as_bytes());
        out.extend_from_slice(line_ending);
    }
    out.extend_from_slice(&raw[at..]);
    out
}

/// The headers of this node's receipt request for a turn whose own request
/// asks for `sampling`, and what the turn's audit then holds the peer to.
/// The client's VRF key always. A seed too when the body fixes none: a
/// sampled turn then draws with a seed the peer did not choose, which the
/// audit replays. The seed is `seed_header`, the one the client's request
/// already names, else `fresh`.
pub(super) fn receipt_headers(
    key: &str,
    mut sampling: RequestedSampling,
    seed_header: Option<u32>,
    fresh: u32,
) -> (Vec<(&'static str, String)>, RequestedSampling) {
    let mut headers = vec![(CLIENT_KEY_HEADER, key.to_owned())];
    if sampling.client_seed().is_none() {
        let seed = seed_header.unwrap_or(fresh);
        if seed_header.is_none() {
            headers.push((SEED_HEADER, seed.to_string()));
        }
        sampling.seed = Some(u64::from(seed));
    }
    (headers, sampling)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "0b0e7c1e-58a5-4b3f-9c36-0d2f6f1f8a10";

    fn parse(raw: &[u8]) -> Option<ChatRequest> {
        ChatRequest::read(raw).expect("a request this node can read")
    }

    fn request(body: &str, extra: &str) -> Vec<u8> {
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{extra}Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    fn parsed(body: &str) -> ChatRequest {
        parse(&request(body, "")).expect("a chat request")
    }

    fn chat(fields: &str) -> String {
        format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}]{fields}}}"#)
    }

    #[test]
    fn a_chat_request_is_parsed_and_gets_the_key_header() {
        let raw = request(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},{"role":"user","content":[{"type":"text","text":"Paris?"}]}]}"#,
            &format!("X-Request-Id: {ID}\r\n"),
        );
        let parsed = parse(&raw).unwrap();
        assert_eq!(parsed.model, "m");
        assert_eq!(parsed.prompt.as_deref(), Some("Paris?"));
        assert_eq!(parsed.asked_id.as_deref(), Some(ID));
        assert_eq!(parsed.text_unavailable, None);
        let (headers, sampling) = receipt_headers("ab", parsed.sampling.clone(), None, 77);
        // the body fixes no seed: this node names one, and holds the peer to it
        assert_eq!(sampling.seed, Some(77));
        let with = with_headers(&raw, parsed.header_end, &headers);
        let text = String::from_utf8(with).unwrap();
        assert!(
            text.contains("\r\nx-receipt-client-key: ab\r\nx-receipt-seed: 77\r\n\r\n{"),
            "{text}"
        );
        let mut headers = [httparse::EMPTY_HEADER; 16];
        let mut reparsed = httparse::Request::new(&mut headers);
        assert!(reparsed.parse(text.as_bytes()).unwrap().is_complete());
    }

    #[test]
    fn a_request_that_fixes_a_seed_keeps_it() {
        // a seed in the body is the client's: no header, and it is the one
        // the audit holds the peer to
        let seeded = parsed(&chat(r#","temperature":0.7,"seed":1234"#));
        let (headers, sampling) =
            receipt_headers("ab", seeded.sampling.clone(), seeded.seed_header, 77);
        assert_eq!(headers, vec![(CLIENT_KEY_HEADER, "ab".to_owned())]);
        assert_eq!(sampling.seed, Some(1234));
        // seed 0 in the body asks the peer for a random one, like none
        let zero = parsed(&chat(r#","seed":0"#));
        let (headers, sampling) = receipt_headers("ab", zero.sampling.clone(), None, 77);
        assert_eq!(headers.len(), 2);
        assert_eq!(sampling.seed, Some(77));
        // a seed header the client sent is the one the peer reads
        let raw = request(&chat(""), "X-Receipt-Seed: 55\r\n");
        let own = parse(&raw).unwrap();
        assert_eq!(own.seed_header, Some(55));
        let (headers, sampling) = receipt_headers("ab", own.sampling.clone(), own.seed_header, 77);
        assert_eq!(headers.len(), 1);
        assert_eq!(sampling.seed, Some(55));
        // one the peer would not read is as none
        for bad in ["0", "-3", "abc", "4294967296"] {
            let raw = request(&chat(""), &format!("X-Receipt-Seed: {bad}\r\n"));
            assert_eq!(parse(&raw).unwrap().seed_header, None, "{bad}");
        }
    }

    /// A client may send its body in chunks; the peer reads the same JSON.
    #[test]
    fn a_chunked_body_is_read_as_its_entity() {
        let body = chat(r#","temperature":0.5"#);
        let (head, tail) = body.split_at(10);
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nX-Request-Id: {ID}\r\n\r\n{:x}\r\n{head}\r\n{:x};ext=1\r\n{tail}\r\n0\r\n\r\n",
            head.len(),
            tail.len()
        );
        let parsed = parse(raw.as_bytes()).expect("a chunked chat request");
        assert_eq!(parsed.model, "m");
        assert_eq!(parsed.sampling.temperature, Some(0.5));
        assert_eq!(parsed.asked_id.as_deref(), Some(ID));
        // the chunk framing is kept for the peer, with the headers added
        let with = with_headers(raw.as_bytes(), parsed.header_end, &[("x-a", "b".into())]);
        assert!(String::from_utf8(with).unwrap().ends_with("0\r\n\r\n"));
        // framing that does not end is a chat completion this node cannot
        // read, which a request that needs receipts is not served unwatched for
        let cut = &raw[..raw.len() - 5];
        assert!(ChatRequest::read(cut.as_bytes()).is_err());
        let not_json = request("{\"model\":", "");
        assert!(ChatRequest::read(&not_json).is_err());
    }

    #[test]
    fn only_tools_or_several_choices_allow_another_answer_shape() {
        for fields in [
            r#","tools":[{"type":"function"}]"#,
            r#","tool_choice":"auto""#,
            r#","functions":[{"name":"f"}]"#,
            r#","n":3"#,
        ] {
            assert!(parsed(&chat(fields)).shaped_answer.is_some(), "{fields}");
        }
        for fields in [
            "",
            r#","tools":[],"n":1"#,
            r#","response_format":{"type":"json_object"}"#,
        ] {
            assert_eq!(parsed(&chat(fields)).shaped_answer, None, "{fields}");
        }
    }

    #[test]
    fn requests_the_audit_leaves_alone() {
        let own_key = request(
            r#"{"model":"m","messages":[]}"#,
            "X-Receipt-Client-Key: cd\r\n",
        );
        assert!(parse(&own_key).is_none());
        let embeddings = b"POST /v1/embeddings HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        assert!(parse(embeddings).is_none());
    }

    #[test]
    fn the_id_is_the_one_canonical_request_id() {
        // no id, an id that is no UUID, or two of them: the peer issues its
        // receipt under an id of its own, which this node cannot name
        for extra in [
            String::new(),
            "x-request-id: not-a-uuid\r\n".to_owned(),
            format!("x-request-id: {ID}\r\nx-request-id: {ID}\r\n"),
        ] {
            let raw = request(&chat(""), &extra);
            assert_eq!(parse(&raw).unwrap().asked_id, None, "{extra}");
        }
    }

    #[test]
    fn the_sampling_the_request_set_is_carried() {
        let plain = parsed(&chat("")).sampling;
        // nothing set: the node's defaults apply, and it claims no
        // constraint the request did not send
        assert_eq!(
            plain,
            RequestedSampling {
                grammar: Some(false),
                penalties: Some(false),
                logit_bias: Some(false),
                sampler_chain: Some(false),
                ignore_eos: Some(false),
                ..RequestedSampling::default()
            }
        );
        let set = parsed(&chat(
            r#","temperature":0.7,"top_p":0.9,"top_k":40,"min_p":0.05,"seed":7,"reasoning_budget":-1"#,
        ))
        .sampling;
        assert_eq!(set.temperature, Some(0.7));
        assert_eq!(set.top_p, Some(0.9));
        assert_eq!(set.top_k, Some(40));
        assert_eq!(set.min_p, Some(0.05));
        assert_eq!(set.seed, Some(7));
        assert_eq!(set.reasoning_budget, Some(-1));
        for (fields, grammar, penalties, bias, chain) in [
            (r#","grammar":"root ::= \"a\"""#, true, false, false, false),
            (
                r#","response_format":{"type":"json_object"}"#,
                true,
                false,
                false,
                false,
            ),
            (
                r#","tools":[{"type":"function"}]"#,
                true,
                false,
                false,
                false,
            ),
            (r#","repeat_penalty":1.1"#, false, true, false, false),
            (r#","presence_penalty":0.5"#, false, true, false, false),
            (r#","dry":{"multiplier":0.8}"#, false, true, false, false),
            (r#","mirostat_mode":2"#, false, true, false, false),
            (r#","logit_bias":{"5":1.0}"#, false, false, true, false),
            (r#","samplers":["top_k"]"#, false, false, false, true),
            // a field set to nothing sends nothing
            (
                r#","tools":[],"logit_bias":null,"grammar":"""#,
                false,
                false,
                false,
                false,
            ),
        ] {
            let sampling = parsed(&chat(fields)).sampling;
            assert_eq!(sampling.grammar, Some(grammar), "{fields}");
            assert_eq!(sampling.penalties, Some(penalties), "{fields}");
            assert_eq!(sampling.logit_bias, Some(bias), "{fields}");
            assert_eq!(sampling.sampler_chain, Some(chain), "{fields}");
        }
    }

    #[test]
    fn a_reasoning_budget_is_carried_only_when_it_is_one_number() {
        for (fields, budget) in [
            (r#","thinking_budget_tokens":512"#, Some(512)),
            (r#","reasoning_budget":0"#, Some(0)),
            // two controls, an effort or a template option: the node
            // settles the budget its own way
            (r#","reasoning_budget":512,"thinking_budget":64"#, None),
            (r#","reasoning_budget":-1,"reasoning_effort":"low""#, None),
            (r#","reasoning_budget":-1,"enable_thinking":false"#, None),
            (
                r#","reasoning_budget":9,"chat_template_kwargs":{"enable_thinking":false}"#,
                None,
            ),
            (r#","reasoning_budget":-2"#, None),
            (r#","reasoning_budget":"many""#, None),
        ] {
            assert_eq!(
                parsed(&chat(fields)).sampling.reasoning_budget,
                budget,
                "{fields}"
            );
        }
    }

    #[test]
    fn requests_whose_answer_is_not_the_sampled_text_have_no_text_check() {
        for fields in [
            r#","stop":["\n"]"#,
            r#","stop":"END""#,
            r#","tools":[{"type":"function"}]"#,
            r#","tool_choice":"auto""#,
            r#","response_format":{"type":"json_object"}"#,
            r#","json_schema":{"type":"object"}"#,
            r#","n":2"#,
            r#","logprobs":true"#,
            r#","top_logprobs":3"#,
            r#","continue_final_message":true"#,
            r#","add_generation_prompt":false"#,
            r#","reasoning_format":"deepseek-legacy""#,
            r#","reasoning_format":"hidden""#,
            r#","echo":true"#,
        ] {
            assert!(parsed(&chat(fields)).text_unavailable.is_some(), "{fields}");
        }
        for fields in [
            "",
            r#","stop":[],"n":1,"logprobs":false,"tools":[]"#,
            r#","reasoning_format":"deepseek""#,
            r#","temperature":0,"max_tokens":16,"stream":true"#,
            // a grammar constrains the tokens, and the text is still theirs
            r#","grammar":"root ::= \"a\"""#,
        ] {
            assert_eq!(parsed(&chat(fields)).text_unavailable, None, "{fields}");
        }
    }

    #[test]
    fn a_prefilled_answer_and_media_have_no_text_check() {
        let prefill = r#"{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Sure,"}]}"#;
        assert!(parsed(prefill).text_unavailable.is_some());
        let prefix = r#"{"model":"m","messages":[{"role":"user","content":"hi","prefix":true}]}"#;
        assert!(parsed(prefix).text_unavailable.is_some());
        let image = r#"{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"data:"}}]}]}"#;
        let image = parsed(image);
        assert!(image.text_unavailable.is_some());
        assert_eq!(image.prompt, None);
        // an earlier turn's image changes the prompt's text too
        let earlier = r#"{"model":"m","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]},{"role":"assistant","content":"ok"},{"role":"user","content":"and now?"}]}"#;
        assert!(parsed(earlier).text_unavailable.is_some());
    }

    #[test]
    fn only_a_prompt_every_tokenizer_keeps_is_checked() {
        let user = |content: &str| {
            parsed(&format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":{content}}}]}}"#
            ))
            .prompt
        };
        assert_eq!(
            user(r#""What is 2+2?\n""#).as_deref(),
            Some("What is 2+2?\n")
        );
        // a tokenizer may normalize these, and its text then differs
        assert_eq!(user(r#""ﬁn de siècle""#), None);
        assert_eq!(user(r#""bell \u0007""#), None);
        // parts a template may join its own way
        assert_eq!(
            user(r#"[{"type":"text","text":"a"},{"type":"text","text":"b"}]"#),
            None
        );
        assert_eq!(user("7"), None);
    }
}
