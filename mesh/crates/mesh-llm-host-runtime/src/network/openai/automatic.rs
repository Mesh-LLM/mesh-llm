//! The automatic routing directive and its serving modes.
//!
//! Mesh exposes exactly one automatic directive. A client that does not want
//! to name a model sends [`DIRECTIVE`] and the mesh decides how to serve the
//! request as well as it can. `auto` is accepted as a deprecated alias so
//! existing OpenAI-compatible clients keep working.
//!
//! Core always selects one capability-compatible model. An explicitly installed
//! plugin may override the directive for non-streaming, text-only chat requests.
//! Streaming, media, non-chat and model-less requests retain core routing.

use serde_json::Value;

use crate::network::router;

/// The one automatic routing directive clients should send.
pub(crate) const DIRECTIVE: &str = "mesh";

/// Accepted spelling of [`DIRECTIVE`] retained for compatibility.
///
/// Both spellings use the same single-model core routing and optional plugin override.
pub(crate) const DEPRECATED_ALIAS: &str = "auto";

/// How the mesh will serve a request that used the automatic directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServingMode {
    /// Allow an installed plugin to override the directive.
    PluginOverride,
    /// Serve from a single capability-selected model.
    SingleModel(SingleModelReason),
}

/// Why a request that asked for automatic routing is being served by one model.
///
/// Carried so logs, tests, and the management API can distinguish a deliberate
/// eligibility constraint from an unavailable optional override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SingleModelReason {
    /// The request carries image, audio, or file content.
    MediaInput,
    /// The client asked for a streamed response.
    StreamRequested,
    /// The request named no model, so it never opted into plugin override serving.
    ModelUnspecified,
    /// The endpoint does not supply the buffered chat body required by overrides.
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

/// The forwarded endpoint accepted by the buffered chat override seam.
///
/// Overrides receive chat-completion bodies with a `messages` array.
/// `/v1/completions` has a `prompt`, not `messages`, and stays on core routing.
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// True when `path` is the forwarded chat-completions endpoint.
///
/// Reads the forwarded path, not the client's: `/v1/responses` is normalised
/// onto chat completions before routing, so it is plugin override-eligible.
pub(crate) fn is_chat_shaped_path(path: &str) -> bool {
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
/// A request that named no model did not opt into plugin override serving, so it is
/// served by a single model rather than silently paying plugin override latency and
/// cost. Naming the directive is the opt-in.
///
/// The plugin handler evaluates this in two stages, because its `messages`
/// contract check must stay reachable — see [`envelope_mode`] and
/// [`content_mode`]. This function is the whole decision for callers that have
/// no such ordering constraint.
pub(crate) fn serving_mode(request: AutomaticRequest<'_>) -> ServingMode {
    match envelope_mode(request) {
        ServingMode::PluginOverride => content_mode(request.body),
        single => single,
    }
}

/// The part of the mode decision that depends only on the request's envelope:
/// which endpoint it arrived on and whether it named the directive at all.
///
/// Separated from [`content_mode`] so the plugin handler can answer "could a
/// plugin override ever fan this out?" *before* validating the chat body. A
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
    ServingMode::PluginOverride
}

/// The part of the mode decision that reads what the client declared in the
/// body of an otherwise plugin override-eligible chat request.
///
/// Runs *after* the chat contract check in the plugin handler, so a malformed
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
    ServingMode::PluginOverride
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
    fn plain_text_request_is_eligible_for_plugin_override() {
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &text_request())),
            ServingMode::PluginOverride
        );
    }

    #[test]
    fn model_less_request_takes_a_single_model() {
        // No `model` field means the client never opted into plugin override
        // serving, so it stays on the core routing path.
        let mut body = text_request();
        body.as_object_mut().unwrap().remove("model");
        assert_eq!(
            serving_mode(auto(None, &body)),
            ServingMode::SingleModel(SingleModelReason::ModelUnspecified)
        );
    }

    #[test]
    fn deprecated_alias_is_eligible_for_plugin_override_too() {
        // `auto` and `mesh` are one directive; they must not diverge in mode.
        let body = text_request();
        assert_eq!(
            serving_mode(auto(Some(DEPRECATED_ALIAS), &body)),
            serving_mode(auto(Some(DIRECTIVE), &body))
        );
    }

    #[test]
    fn a_non_chat_endpoint_takes_a_single_model() {
        // `/v1/completions` carries `prompt`, not the buffered `messages` body
        // accepted by the override seam. Keep it on core routing.
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
    fn a_malformed_chat_body_stays_plugin_override_eligible_at_the_envelope() {
        // The plugin handler rejects a missing or non-array `messages` with a
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
                ServingMode::PluginOverride,
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
                ServingMode::PluginOverride,
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
    fn a_normalised_responses_request_still_is_eligible_for_plugin_override() {
        // `/v1/responses` is rewritten onto chat completions with a `messages`
        // array before routing, so gating on the *forwarded* path keeps it
        // plugin override-eligible. Gating on the client's path would exclude it.
        assert_eq!(
            serving_mode(AutomaticRequest {
                model: Some(DIRECTIVE),
                path: CHAT_COMPLETIONS_PATH,
                body: &text_request(),
            }),
            ServingMode::PluginOverride
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
            ServingMode::PluginOverride
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
    fn stream_false_still_is_eligible_for_plugin_override() {
        let mut body = text_request();
        body["stream"] = json!(false);
        assert_eq!(
            serving_mode(auto(Some(DIRECTIVE), &body)),
            ServingMode::PluginOverride
        );
    }

    #[test]
    fn non_bool_stream_does_not_change_routing() {
        // A stringly-typed `stream` is not a streaming declaration; treating it
        // as one would let a malformed field silently disable the plugin override.
        for value in [json!("true"), json!(1), json!(null)] {
            let mut body = text_request();
            body["stream"] = value;
            assert_eq!(
                serving_mode(auto(Some(DIRECTIVE), &body)),
                ServingMode::PluginOverride
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
        // alone would leave it in the plugin override and silently drop the audio.
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
