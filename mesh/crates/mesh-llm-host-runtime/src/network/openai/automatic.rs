//! The automatic routing directive and its serving modes.
//!
//! Mesh exposes exactly one automatic directive. A client that does not want
//! to name a model sends [`DIRECTIVE`] and the mesh decides how to serve the
//! request as well as it can. `auto` is accepted as a deprecated alias so
//! existing OpenAI-compatible clients keep working.
//!
//! The directive resolves to one of two modes. Both are first-class: neither
//! is a failure path.
//!
//! * [`ServingMode::Committee`] — fan out to a Mixture-of-Agents committee.
//!   Chosen when the request permits it and the mesh can actually field one.
//! * [`ServingMode::SingleModel`] — serve from one capability-selected model.
//!   Chosen when the request states something a committee cannot honour.
//!
//! Mode selection reads *declarations the client made*, not guesses about
//! intent:
//!
//! * **Media.** MoA aggregation compares and synthesises drafts as strings, so
//!   a committee has no defined semantics for image or audio input — and the
//!   text-extraction step drops non-text content blocks outright. A media
//!   request must reach a model whose runtime advertises the modality.
//! * **Streaming.** Committee workers are called non-streaming because the
//!   arbiter needs complete drafts to detect divergence, so committee "streams"
//!   are synthesised after the fact. A client asking for `stream: true` gets a
//!   single model, which streams tokens for real.
//!
//! Committee-plus-streaming is deliberately not reachable by sending
//! `stream: true`; it would need its own opt-in.

use serde_json::Value;

use crate::network::router;

/// The one automatic routing directive clients should send.
pub(crate) const DIRECTIVE: &str = mesh_mixture_of_agents::VIRTUAL_MODEL_NAME;

/// Accepted spelling of [`DIRECTIVE`] retained for compatibility.
///
/// Historically `auto` selected a single "good" model while `mesh` convened a
/// committee. Those were never real alternatives — the committee path already
/// served a single model whenever one was all the mesh could field — so the
/// two names described one intent and are now one directive.
pub(crate) const DEPRECATED_ALIAS: &str = "auto";

/// How the mesh will serve a request that used the automatic directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServingMode {
    /// Fan out to a committee and aggregate the drafts.
    Committee,
    /// Serve from a single capability-selected model.
    SingleModel(SingleModelReason),
}

/// Why a request that asked for automatic routing is being served by one model.
///
/// Carried so logs, tests, and the management API can distinguish a deliberate
/// mode choice from a mesh that simply could not field a committee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SingleModelReason {
    /// The request carries image, audio, or file content.
    MediaInput,
    /// The client asked for a streamed response.
    StreamRequested,
    /// The request named no model, so it never opted into committee serving.
    ModelUnspecified,
    /// The endpoint is not chat-shaped, so a committee has no messages to
    /// fan out.
    NonChatRequest,
}

impl SingleModelReason {
    /// Stable identifier for logs and route observers.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::MediaInput => "media_input",
            Self::StreamRequested => "stream_requested",
            Self::ModelUnspecified => "model_unspecified",
            Self::NonChatRequest => "non_chat_request",
        }
    }
}

/// The forwarded endpoint a committee needs in order to fan out.
///
/// MoA builds worker calls as chat completions from a `messages` array, so it
/// can only serve requests that arrive on (or are normalised onto) that
/// endpoint. `/v1/completions` has a `prompt`, not `messages`.
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// True when `path` is the forwarded chat-completions endpoint.
///
/// Reads the forwarded path, not the client's: `/v1/responses` is normalised
/// onto chat completions before routing, so it is committee-eligible.
fn is_chat_shaped_path(path: &str) -> bool {
    path.split('?').next().unwrap_or(path) == CHAT_COMPLETIONS_PATH
}

/// True when `model` names the automatic directive under any accepted spelling.
///
/// A request with no `model` at all is also automatic, but that is the caller's
/// observation to make — this answers only about a name that was supplied.
pub(crate) fn is_directive(model: &str) -> bool {
    model == DIRECTIVE || model == DEPRECATED_ALIAS
}

/// Warn once per request when a client used the deprecated spelling.
///
/// Logged rather than rejected: the alias keeps working, and the operator needs
/// to know a client is still sending it before it is eventually removed.
pub(crate) fn warn_if_deprecated_alias(model: Option<&str>) {
    if model == Some(DEPRECATED_ALIAS) {
        tracing::warn!(
            "request used deprecated model \"{DEPRECATED_ALIAS}\"; \
             send \"{DIRECTIVE}\" instead (\"{DEPRECATED_ALIAS}\" is an alias \
             and will be removed in a future release)"
        );
    }
}

/// A request the mesh has already established is automatic.
///
/// Named fields rather than positional arguments because `model` and `path`
/// are both strings and silently swapping them would change routing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AutomaticRequest<'a> {
    /// The `model` the client sent, or `None` when it sent none.
    pub(crate) model: Option<&'a str>,
    /// The **forwarded** request path, after `/v1/responses` normalisation.
    pub(crate) path: &'a str,
    /// The parsed request body.
    pub(crate) body: &'a Value,
}

/// Choose the serving mode for a request that is being routed automatically.
///
/// Callers must only reach here once they have established the request is
/// automatic; an explicitly named model never has a mode.
///
/// A request that named no model did not opt into committee serving, so it is
/// served by a single model rather than silently paying committee latency and
/// cost. Naming the directive is the opt-in.
///
/// The MoA gateway evaluates this in two stages, because its `messages`
/// contract check must stay reachable — see [`envelope_mode`] and
/// [`content_mode`]. This function is the whole decision for callers that have
/// no such ordering constraint.
pub(crate) fn serving_mode(request: AutomaticRequest<'_>) -> ServingMode {
    match envelope_mode(request) {
        ServingMode::Committee => content_mode(request.body),
        single => single,
    }
}

/// The part of the mode decision that depends only on the request's envelope:
/// which endpoint it arrived on and whether it named the directive at all.
///
/// Separated from [`content_mode`] so the MoA gateway can answer "could a
/// committee ever fan this out?" *before* validating the chat body. A
/// `/v1/completions` request has a `prompt` and no `messages`, so asserting the
/// chat contract on it would reject a shape `model=auto` used to serve from one
/// concrete model.
pub(crate) fn envelope_mode(request: AutomaticRequest<'_>) -> ServingMode {
    let AutomaticRequest { model, path, .. } = request;
    if model.is_none() {
        return ServingMode::SingleModel(SingleModelReason::ModelUnspecified);
    }
    if !is_chat_shaped_path(path) {
        return ServingMode::SingleModel(SingleModelReason::NonChatRequest);
    }
    ServingMode::Committee
}

/// The part of the mode decision that reads what the client declared in the
/// body of an otherwise committee-eligible chat request.
///
/// Runs *after* the chat contract check in the MoA gateway, so a malformed
/// chat body still gets the gateway's own rejection rather than being diverted
/// to single-model routing and failing later with a less specific error.
pub(crate) fn content_mode(body: &Value) -> ServingMode {
    // Media first: a media request that is also streaming still needs a
    // modality-capable model, and reporting the media reason is the more
    // useful diagnostic of the two.
    if router::media_requirements(body).has_media {
        return ServingMode::SingleModel(SingleModelReason::MediaInput);
    }
    if requests_streaming(body) {
        return ServingMode::SingleModel(SingleModelReason::StreamRequested);
    }
    ServingMode::Committee
}

/// True when the body asks for a streamed response.
///
/// Only a literal JSON `true` counts. OpenAI clients send a bool here, and
/// coercing strings or numbers would let an unrelated field silently change
/// routing.
fn requests_streaming(body: &Value) -> bool {
    body.get("stream") == Some(&Value::Bool(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text_request() -> Value {
        json!({
            "model": DIRECTIVE,
            "messages": [{ "role": "user", "content": "hello" }],
        })
    }

    /// An automatic chat-completions request. Endpoint shape is fixed here so
    /// each test varies only the field it is about.
    fn auto<'a>(model: Option<&'a str>, body: &'a Value) -> AutomaticRequest<'a> {
        AutomaticRequest {
            model,
            path: CHAT_COMPLETIONS_PATH,
            body,
        }
    }

    #[test]
    fn both_spellings_are_the_directive() {
        assert!(is_directive(DIRECTIVE));
        assert!(is_directive(DEPRECATED_ALIAS));
    }

    #[test]
    fn a_named_model_is_not_the_directive() {
        assert!(!is_directive("Qwen3-8B"));
        // Guard against prefix/substring matching: these are real model names.
        assert!(!is_directive("mesh-router-8B"));
        assert!(!is_directive("autocoder-3B"));
    }

    #[test]
    fn plain_text_request_convenes_a_committee() {
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &text_request())),
            ServingMode::Committee
        );
    }

    #[test]
    fn model_less_request_takes_a_single_model() {
        // No `model` field means the client never opted into committee
        // serving, so it must not silently pay committee latency and cost.
        let mut body = text_request();
        body.as_object_mut().unwrap().remove("model");
        assert_eq!(
            serving_mode(auto(None, &body)),
            ServingMode::SingleModel(SingleModelReason::ModelUnspecified)
        );
    }

    #[test]
    fn deprecated_alias_convenes_a_committee_too() {
        // `auto` and `mesh` are one directive; they must not diverge in mode.
        let body = text_request();
        assert_eq!(
            serving_mode(auto(Some(DEPRECATED_ALIAS), &body)),
            serving_mode(auto(Some(DIRECTIVE), &body))
        );
    }

    #[test]
    fn a_non_chat_endpoint_takes_a_single_model() {
        // `/v1/completions` carries `prompt`, not `messages`. A committee has
        // nothing to fan out, and letting one convene would reject with
        // "MoA requires a non-empty `messages` array" a request shape that
        // `model=auto` used to serve from one concrete model.
        let body = json!({ "model": DIRECTIVE, "prompt": "once upon a time" });
        assert_eq!(
            serving_mode(AutomaticRequest {
                model: Some(DIRECTIVE),
                path: "/v1/completions",
                body: &body,
            }),
            ServingMode::SingleModel(SingleModelReason::NonChatRequest)
        );
    }

    #[test]
    fn a_malformed_chat_body_stays_committee_eligible_at_the_envelope() {
        // The MoA gateway rejects a missing or non-array `messages` with a
        // precise 400. That check must stay reachable, so the envelope stage
        // must not divert a chat-shaped request on body grounds — even when the
        // body also declares something `content_mode` would divert on.
        for body in [
            json!({ "model": DIRECTIVE, "stream": true }),
            json!({ "model": DIRECTIVE, "stream": true, "messages": "hello" }),
            json!({ "model": DIRECTIVE }),
        ] {
            assert_eq!(
                envelope_mode(auto(Some(DIRECTIVE), &body)),
                ServingMode::Committee,
                "envelope stage must defer to the gateway's contract check: {body}"
            );
        }
    }

    #[test]
    fn the_two_stages_compose_into_the_whole_decision() {
        // `serving_mode` is the single-stage answer for callers with no
        // ordering constraint; it must not drift from the staged pair.
        let mut streaming = text_request();
        streaming["stream"] = json!(true);
        let completions = json!({ "model": DIRECTIVE, "prompt": "hi" });
        for (request, expected) in [
            (
                auto(Some(DIRECTIVE), &text_request()),
                ServingMode::Committee,
            ),
            (
                auto(Some(DIRECTIVE), &streaming),
                ServingMode::SingleModel(SingleModelReason::StreamRequested),
            ),
            (
                AutomaticRequest {
                    model: Some(DIRECTIVE),
                    path: "/v1/completions",
                    body: &completions,
                },
                ServingMode::SingleModel(SingleModelReason::NonChatRequest),
            ),
            (
                auto(None, &text_request()),
                ServingMode::SingleModel(SingleModelReason::ModelUnspecified),
            ),
        ] {
            assert_eq!(serving_mode(request), expected);
        }
    }

    #[test]
    fn a_normalised_responses_request_still_convenes_a_committee() {
        // `/v1/responses` is rewritten onto chat completions with a `messages`
        // array before routing, so gating on the *forwarded* path keeps it
        // committee-eligible. Gating on the client's path would exclude it.
        assert_eq!(
            serving_mode(AutomaticRequest {
                model: Some(DIRECTIVE),
                path: CHAT_COMPLETIONS_PATH,
                body: &text_request(),
            }),
            ServingMode::Committee
        );
    }

    #[test]
    fn a_query_string_does_not_hide_the_chat_endpoint() {
        assert_eq!(
            serving_mode(AutomaticRequest {
                model: Some(DIRECTIVE),
                path: "/v1/chat/completions?trace=1",
                body: &text_request(),
            }),
            ServingMode::Committee
        );
    }

    #[test]
    fn streaming_request_takes_a_single_model() {
        let mut body = text_request();
        body["stream"] = json!(true);
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &body)),
            ServingMode::SingleModel(SingleModelReason::StreamRequested)
        );
    }

    #[test]
    fn stream_false_still_convenes_a_committee() {
        let mut body = text_request();
        body["stream"] = json!(false);
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &body)),
            ServingMode::Committee
        );
    }

    #[test]
    fn non_bool_stream_does_not_change_routing() {
        // A stringly-typed `stream` is not a streaming declaration; treating it
        // as one would let a malformed field silently disable the committee.
        for value in [json!("true"), json!(1), json!(null)] {
            let mut body = text_request();
            body["stream"] = value;
            assert_eq!(
                serving_mode(auto(Some(DIRECTIVE), &body)),
                ServingMode::Committee
            );
        }
    }

    #[test]
    fn image_request_takes_a_single_model() {
        let body = json!({
            "model": DIRECTIVE,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "what is this?" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
                ],
            }],
        });
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &body)),
            ServingMode::SingleModel(SingleModelReason::MediaInput)
        );
    }

    #[test]
    fn audio_request_takes_a_single_model() {
        // Audio-only input has `needs_vision == false`, so gating on vision
        // alone would leave it in the committee and silently drop the audio.
        let body = json!({
            "model": DIRECTIVE,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "input_audio", "input_audio": { "data": "AAAA", "format": "wav" } },
                ],
            }],
        });
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &body)),
            ServingMode::SingleModel(SingleModelReason::MediaInput)
        );
    }

    #[test]
    fn media_outranks_streaming() {
        let body = json!({
            "model": DIRECTIVE,
            "stream": true,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
                ],
            }],
        });
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &body)),
            ServingMode::SingleModel(SingleModelReason::MediaInput)
        );
    }
}
