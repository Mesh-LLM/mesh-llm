use mesh_llm_plugin::{
    HostInferenceRequest, MeshVisibility, PluginContext, PluginMetadata, PluginRuntime,
    SimplePlugin, VirtualModelInvocation, VirtualModelResponse, VirtualModelRouter,
    operation_with_schema, plugin_server_info, structured_tool_result, virtual_model,
};
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time::Instant;

pub const PLUGIN_ID: &str = "mesh-moa";
pub const HANDLER: &str = "chat";

/// Above this verified parameter count a worker is in the capable tier used
/// by the MoA engine. Unknown sizes are deliberately treated as small for
/// destructive pool decisions: a label is not evidence that a model is big.
const SMALL_TIER_MAX_B: f64 = moa::SMALL_TIER_MAX_B;

/// Measured quality is flat beyond four capable workers while fan-out cost is
/// roughly `2N+1` calls. Keep the shared mesh from being consumed by one turn.
const COMMITTEE_CAP: usize = 4;

/// Lines the host drips onto a streaming caller while the committee works.
/// Played once in order, then the last line repeats. See
/// `VirtualModelBuilder::progress_lines`.
pub const PROGRESS_LINES: [&str; 6] = [
    "Routing through mesh…",
    "Querying peer models…",
    "Comparing responses…",
    "Waiting on a slow peer…",
    "Still gathering responses…",
    "Hold on, this one's taking a moment…",
];

pub async fn run(stream: mesh_llm_plugin::LocalStream) -> anyhow::Result<()> {
    PluginRuntime::run_with_stream(plugin(), stream).await
}

/// How long a turn waits for better answers before shipping what it has.
struct PatienceProfile {
    first_answer_grace: Duration,
    strong_patience: Duration,
}

/// Timing profile for the turn, tightened on a public mesh.
///
/// A public mesh is a pathological *availability* case (unknown peers, wider
/// latency spread, more churn), not a trust case. Both knobs here are "how long
/// do we hold a usable answer hoping for a better one" — exactly the wait that
/// hurts most when the tail is long. The hard bounds are unchanged; only the
/// optional waiting shrinks, so quality paths still run when peers are prompt.
///
/// The host used to choose this per turn from `node.public_mesh`; it now hands
/// the plugin the same signal as `mesh_visibility` at init, so the policy stays
/// identical.
fn patience_profile(public_mesh: bool) -> PatienceProfile {
    if public_mesh {
        PatienceProfile {
            // Ship a good answer sooner rather than wait out a long tail.
            first_answer_grace: Duration::from_millis(1500),
            // Still give a strong peer a real chance, but don't hold a usable
            // small-tier answer for 20s against an unknown remote worker.
            strong_patience: Duration::from_secs(8),
        }
    } else {
        PatienceProfile {
            first_answer_grace: Duration::from_secs(10),
            strong_patience: Duration::from_secs(20),
        }
    }
}

pub fn plugin() -> SimplePlugin {
    let public_mesh = Arc::new(AtomicBool::new(false));
    let manifest = mesh_llm_plugin::plugin_manifest![
        virtual_model(moa::VIRTUAL_MODEL_NAME, HANDLER)
            .supports_tools(true)
            .supports_streaming(true)
            .requires_candidates(true)
            .input_modalities(["text", "image", "audio"])
            // The host drips these onto a streaming caller while the committee
            // works, so the client's thinking pane shows live activity instead
            // of a stalled spinner. Played once in order, then the last line
            // repeats. Short, factual, and grounded in what the mesh is
            // actually doing — not invented model "thoughts".
            .progress_lines(PROGRESS_LINES)
    ];
    let mut router = VirtualModelRouter::new();
    let turn_public_mesh = Arc::clone(&public_mesh);
    router.add_raw(
        operation_with_schema(HANDLER, "Run a Mesh MoA turn", serde_json::Map::new()),
        move |request, context| {
            let context = context.owned();
            let public_mesh = Arc::clone(&turn_public_mesh);
            Box::pin(async move {
                let invocation: VirtualModelInvocation = request.arguments()?;
                let response =
                    handle(invocation, context, public_mesh.load(Ordering::Relaxed)).await;
                structured_tool_result(response)
            })
        },
    );
    SimplePlugin::new(PluginMetadata::new(
        PLUGIN_ID,
        env!("CARGO_PKG_VERSION"),
        plugin_server_info(
            PLUGIN_ID,
            env!("CARGO_PKG_VERSION"),
            "Mesh MoA",
            "Built-in mixture-of-agents virtual model",
            None::<String>,
        ),
    ))
    .with_manifest(manifest)
    .with_virtual_model_router(router)
    .on_initialize(move |request, _context| {
        public_mesh.store(
            request.mesh_visibility == MeshVisibility::Public,
            Ordering::Relaxed,
        );
        Box::pin(async { Ok(()) })
    })
}

async fn handle(
    mut invocation: VirtualModelInvocation,
    context: PluginContext<'static>,
    public_mesh: bool,
) -> VirtualModelResponse {
    // The MoA admission contract applied to every chat request before the
    // plugin owned `mesh`: `messages` must be a present, non-empty array.
    // `handle_turn` builds an empty session from those shapes and still
    // returns a worker result or a 502, so a malformed request must be
    // rejected here — ahead of both the committee and the single-candidate
    // passthrough.
    if let Some(message) = chat_request_rejection(&invocation.request) {
        return error_response(400, message);
    }
    if invocation.candidates.is_empty() {
        return error_response(503, "no concrete models available in the mesh");
    }
    let requested_stream = invocation
        .request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(object) = invocation.request.as_object_mut() {
        object.remove("stream");
    }

    let (needs_vision, needs_audio) = match select_request_candidates(&mut invocation) {
        Ok(media) => media,
        Err(response) => return response,
    };
    // Media always takes the direct path, so its eligible standbys should not
    // be discarded by a cap that only exists for committee fan-out.
    let all_small = if needs_vision || needs_audio {
        invocation.candidates.sort_by(candidate_rank);
        false
    } else {
        apply_pool_policy(&mut invocation.candidates)
    };
    // A one-model pool has nothing to aggregate. Passing it through the MoA
    // engine would replace the caller's output budget with the Generalist
    // worker budget (1,024 tokens), which can make an otherwise valid request
    // ineligible for a small-context target before inference even starts.
    // Keep placement and admission host-governed, but preserve the original
    // request shape when there is no actual committee to convene.
    if all_small || invocation.candidates.len() == 1 || needs_vision || needs_audio {
        return direct_capability_response(invocation, context, requested_stream).await;
    }

    let mut backends: Vec<Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for candidate in invocation.candidates {
        let index = backends.len();
        backends.push(Arc::new(HostBackend {
            context: context.clone(),
            target_node_id: candidate.target_node_id.clone(),
        }));
        models.push(
            moa::ModelEntry::new(candidate.model_id, index)
                .with_parameter_count_b(candidate.parameter_count_b),
        );
    }
    let actor_candidates = actor_candidates(&models);
    let patience = patience_profile(public_mesh);
    let config = moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(60),
        reducer_timeout: Duration::from_secs(60),
        hedge_delay: Duration::from_secs(5),
        first_answer_grace: patience.first_answer_grace,
        strong_patience: patience.strong_patience,
        enable_thinking: Some(false),
        actor_candidates,
        reference_policy: moa::ReferencePolicy::Auto,
        refinement_policy: moa::RefinementPolicy::Auto,
    };
    let mut result = moa::handle_turn(&config, &invocation.request).await;
    strip_response_thinking(&mut result.response_body);
    let workers_ok = result
        .worker_summaries
        .iter()
        .filter(|w| w.succeeded)
        .count();
    let headers = vec![
        ("x-moa-elapsed-ms".into(), result.elapsed_ms.to_string()),
        ("x-moa-turn".into(), result.turn_kind.label().to_string()),
        (
            "x-moa-workers".into(),
            result.worker_summaries.len().to_string(),
        ),
        ("x-moa-workers-ok".into(), workers_ok.to_string()),
        ("x-moa-reducer".into(), result.reducer_used.to_string()),
        (
            "x-moa-reducer-attempts".into(),
            result.reducer_attempts.to_string(),
        ),
    ];
    VirtualModelResponse {
        status_code: if is_failure_body(&result.response_body) {
            502
        } else {
            200
        },
        body: result.response_body,
        headers,
        event_stream: requested_stream,
    }
}

/// Media capabilities are runtime requirements; missing tool evidence is not
/// proof that a model cannot call tools. Prefer advertised tool support within
/// the media-eligible pool, but keep that pool when no member advertises it.
fn select_request_candidates(
    invocation: &mut VirtualModelInvocation,
) -> Result<(bool, bool), VirtualModelResponse> {
    let needs_vision = contains_any_key(&invocation.request, &["image_url", "input_image"]);
    let needs_audio = contains_any_key(
        &invocation.request,
        &["audio_url", "input_audio", "input_audio_buffer"],
    );
    let needs_tools = invocation.request.get("tools").is_some_and(|tools| {
        !tools.is_null() && tools.as_array().is_none_or(|items| !items.is_empty())
    });
    invocation.candidates.retain(|candidate| {
        (!needs_vision || candidate.supports_vision) && (!needs_audio || candidate.supports_audio)
    });
    if invocation.candidates.is_empty() {
        return Err(error_response(
            422,
            "no concrete model satisfies the request capabilities",
        ));
    }
    if needs_tools
        && invocation
            .candidates
            .iter()
            .any(|candidate| candidate.supports_tools)
    {
        invocation
            .candidates
            .retain(|candidate| candidate.supports_tools);
    }
    Ok((needs_vision, needs_audio))
}

async fn direct_capability_response(
    invocation: VirtualModelInvocation,
    context: PluginContext<'static>,
    requested_stream: bool,
) -> VirtualModelResponse {
    direct_with_fallback(
        invocation,
        requested_stream,
        Duration::from_secs(60),
        |request| context.infer(request),
    )
    .await
}

async fn direct_with_fallback<F, Fut>(
    mut invocation: VirtualModelInvocation,
    requested_stream: bool,
    budget: Duration,
    mut infer: F,
) -> VirtualModelResponse
where
    F: FnMut(HostInferenceRequest) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<mesh_llm_plugin::HostInferenceResponse>>,
{
    invocation.candidates.sort_by(candidate_rank);
    let deadline = Instant::now() + budget;
    let mut last_failure = None;
    for candidate in &invocation.candidates {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let request = HostInferenceRequest {
            model_id: candidate.model_id.clone(),
            target_node_id: candidate.target_node_id.clone(),
            request: invocation.request.clone(),
            timeout_ms: Some(remaining.as_millis().min(u64::MAX as u128) as u64),
        };
        let result = tokio::time::timeout(remaining, infer(request)).await;
        let response = match result {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                last_failure = Some(error_response(
                    502,
                    &format!("capability route failed: {error}"),
                ));
                continue;
            }
            Err(_) => {
                last_failure = Some(error_response(504, "capability route timed out"));
                break;
            }
        };
        if matches!(response.status_code, 502..=504) {
            last_failure = Some(direct_response(response, requested_stream));
            continue;
        }
        return direct_response(response, requested_stream);
    }
    last_failure.unwrap_or_else(|| error_response(504, "capability route timed out"))
}

fn direct_response(
    mut response: mesh_llm_plugin::HostInferenceResponse,
    requested_stream: bool,
) -> VirtualModelResponse {
    normalize_direct_response_body(&mut response.body);
    VirtualModelResponse {
        status_code: response.status_code,
        body: response.body,
        headers: response
            .served_by
            .map(|served_by| vec![("x-mesh-served-by".into(), served_by)])
            .unwrap_or_default(),
        event_stream: requested_stream,
    }
}

/// The chat contract MoA shares with host ingress: `messages` must be a
/// present, non-empty array. Returns the client-facing message when it is not.
fn chat_request_rejection(request: &Value) -> Option<&'static str> {
    match request.get("messages") {
        Some(Value::Array(messages)) if !messages.is_empty() => None,
        _ => Some("MoA requires a non-empty `messages` array"),
    }
}

fn contains_any_key(value: &Value, keys: &[&str]) -> bool {
    match value {
        Value::Object(object) => object
            .iter()
            .any(|(key, value)| keys.contains(&key.as_str()) || contains_any_key(value, keys)),
        Value::Array(values) => values.iter().any(|value| contains_any_key(value, keys)),
        _ => false,
    }
}

fn is_failure_body(body: &Value) -> bool {
    body.get("error").is_some()
        || body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            == Some("error")
}

fn strip_response_thinking(body: &mut Value) {
    let Some(content) = body
        .pointer_mut("/choices/0/message/content")
        .and_then(|value| value.as_str())
        .map(moa::strip_thinking)
    else {
        return;
    };
    body["choices"][0]["message"]["content"] = Value::String(content);
}

fn normalize_direct_response_body(body: &mut Value) {
    strip_response_thinking(body);
    if body.is_object() {
        body["model"] = Value::String(moa::VIRTUAL_MODEL_NAME.into());
    }
}

fn actor_candidates(models: &[moa::ModelEntry]) -> Vec<usize> {
    let mut indices = (0..models.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        models[*right]
            .parameter_count_b
            .partial_cmp(&models[*left].parameter_count_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// Restore the measured gateway admission rules before the engine assigns
/// roles. An all-small pool regressed against its best member, so it becomes a
/// direct route; any real committee is capped to bound shared-mesh cost.
fn apply_pool_policy(candidates: &mut Vec<mesh_llm_plugin::VirtualModelCandidate>) -> bool {
    candidates.sort_by(candidate_rank);
    let all_small = candidates.iter().all(|candidate| {
        candidate
            .parameter_count_b
            .is_none_or(|size| size < SMALL_TIER_MAX_B)
    });
    if all_small {
        // One answer, but keep eligible physical standbys for retryable
        // placement failures after the candidate snapshot.
    } else {
        candidates.truncate(COMMITTEE_CAP);
    }
    all_small
}

fn candidate_rank(
    left: &mesh_llm_plugin::VirtualModelCandidate,
    right: &mesh_llm_plugin::VirtualModelCandidate,
) -> std::cmp::Ordering {
    left.deprioritized
        .cmp(&right.deprioritized)
        .then_with(|| {
            right
                .parameter_count_b
                .partial_cmp(&left.parameter_count_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| left.model_id.cmp(&right.model_id))
        .then_with(|| left.target_node_id.cmp(&right.target_node_id))
}

fn error_response(status_code: u16, message: &str) -> VirtualModelResponse {
    VirtualModelResponse {
        status_code,
        body: json!({"error": {"message": message, "type": "virtual_model_error"}}),
        headers: Vec::new(),
        event_stream: false,
    }
}

struct HostBackend {
    context: PluginContext<'static>,
    target_node_id: Option<String>,
}

#[async_trait::async_trait]
impl moa::ModelBackend for HostBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        let mut request = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
        });
        if let Some(tools) = tools {
            request["tools"] = tools.clone();
        }
        moa::apply_enable_thinking(&mut request, sampling.enable_thinking);
        let response = self
            .context
            .infer(HostInferenceRequest {
                model_id: model.to_string(),
                target_node_id: self.target_node_id.clone(),
                request,
                timeout_ms: Some(timeout.as_millis().min(u64::MAX as u128) as u64),
            })
            .await
            .map_err(|error| error.to_string())?;
        if !(200..300).contains(&response.status_code) {
            return Err(format!("HTTP {}: {}", response.status_code, response.body));
        }
        Ok(response.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::Plugin;

    fn candidate(
        model_id: &str,
        parameter_count_b: Option<f64>,
        target_node_id: &str,
    ) -> mesh_llm_plugin::VirtualModelCandidate {
        mesh_llm_plugin::VirtualModelCandidate {
            model_id: model_id.into(),
            target_node_id: Some(target_node_id.into()),
            parameter_count_b,
            context_length: Some(32_768),
            deprioritized: false,
            supports_tools: true,
            supports_vision: false,
            supports_audio: false,
        }
    }

    #[test]
    fn plugin_declares_mesh_virtual_model() {
        let manifest = plugin().manifest().expect("manifest");
        assert_eq!(manifest.virtual_models.len(), 1);
        assert_eq!(manifest.virtual_models[0].model_id, "mesh");
        assert_eq!(manifest.virtual_models[0].handler, HANDLER);
        assert_eq!(
            manifest.virtual_models[0].input_modalities,
            ["text", "image", "audio"]
        );
        assert!(manifest.virtual_models[0].requires_candidates);
    }

    #[test]
    fn rejects_malformed_messages_before_dispatch() {
        for request in [
            json!({}),
            json!({"messages": []}),
            json!({"messages": "hi"}),
        ] {
            assert_eq!(
                chat_request_rejection(&request),
                Some("MoA requires a non-empty `messages` array"),
                "unexpected contract decision for {request}"
            );
        }
        assert_eq!(
            chat_request_rejection(&json!({
                "messages": [{"role": "user", "content": "hi"}]
            })),
            None
        );
    }

    #[test]
    fn public_mesh_waits_less_for_a_better_answer() {
        let public = patience_profile(true);
        let private = patience_profile(false);
        assert!(
            public.first_answer_grace < private.first_answer_grace,
            "a public mesh must not hold a usable answer for the trusted-mesh grace"
        );
        assert!(
            public.strong_patience < private.strong_patience,
            "a public mesh must not wait out a long tail for a strong peer"
        );
        assert_eq!(public.first_answer_grace, Duration::from_millis(1500));
        assert_eq!(public.strong_patience, Duration::from_secs(8));
        assert_eq!(private.first_answer_grace, Duration::from_secs(10));
        assert_eq!(private.strong_patience, Duration::from_secs(20));
    }

    #[test]
    fn detects_nested_media_inputs() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]
            }]
        });
        assert!(contains_any_key(&request, &["image_url", "input_image"]));
        assert!(!contains_any_key(&request, &["audio_url", "input_audio"]));
    }

    #[test]
    fn strips_thinking_and_detects_error_shapes() {
        let mut response = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "<think>private</think>answer"}
            }]
        });
        strip_response_thinking(&mut response);
        assert_eq!(response["choices"][0]["message"]["content"], "answer");
        assert!(!is_failure_body(&response));

        response["choices"][0]["finish_reason"] = Value::String("error".into());
        assert!(is_failure_body(&response));
    }

    #[test]
    fn direct_response_uses_virtual_identity_after_thinking_is_removed() {
        let mut response = json!({
            "model": "worker-a",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "<think>private</think>answer"}
            }]
        });

        normalize_direct_response_body(&mut response);

        assert_eq!(response["model"], moa::VIRTUAL_MODEL_NAME);
        assert_eq!(response["choices"][0]["message"]["content"], "answer");
    }

    #[test]
    fn all_small_pool_routes_to_its_best_member() {
        let mut candidates = vec![
            candidate("small-a", Some(8.0), "a"),
            candidate("small-b", Some(9.0), "b"),
            candidate("unknown", None, "c"),
        ];

        assert!(apply_pool_policy(&mut candidates));

        assert_eq!(
            candidates.len(),
            3,
            "physical standbys must survive selection"
        );
        assert_eq!(candidates[0].model_id, "small-b");
    }

    fn invocation(
        candidates: Vec<mesh_llm_plugin::VirtualModelCandidate>,
    ) -> VirtualModelInvocation {
        VirtualModelInvocation {
            request: json!({"messages": [{"role": "user", "content": "hi"}]}),
            candidates,
            response_adapter: String::new(),
        }
    }

    fn tools_invocation(
        candidates: Vec<mesh_llm_plugin::VirtualModelCandidate>,
    ) -> VirtualModelInvocation {
        let mut invocation = invocation(candidates);
        invocation.request["tools"] = json!([{
            "type": "function",
            "function": {"name": "report_status", "parameters": {"type": "object"}}
        }]);
        invocation
    }

    #[tokio::test]
    async fn tools_without_advertised_support_reach_inference() {
        let mut worker = candidate("qwen", Some(27.0), "local");
        worker.supports_tools = false;
        let mut invocation = tools_invocation(vec![worker]);
        assert_eq!(
            select_request_candidates(&mut invocation).unwrap(),
            (false, false)
        );
        let original_request = invocation.request.clone();
        let response =
            direct_with_fallback(invocation, false, Duration::from_secs(1), move |request| {
                assert_eq!(request.model_id, "qwen");
                assert_eq!(request.request, original_request);
                async {
                    Ok(mesh_llm_plugin::HostInferenceResponse {
                        status_code: 200,
                        body: json!({"choices": [{"message": {"tool_calls": [{
                            "type": "function",
                            "function": {"name": "report_status", "arguments": "{}"}
                        }]}}]}),
                        served_by: Some("local".into()),
                    })
                }
            })
            .await;
        assert_eq!(response.status_code, 200);
        assert_eq!(
            response.body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "report_status"
        );
    }

    #[test]
    fn tools_prefer_advertised_support_over_larger_unknown_models() {
        let mut unknown = candidate("unknown", Some(70.0), "a");
        unknown.supports_tools = false;
        let mut invocation = tools_invocation(vec![unknown, candidate("tools", Some(8.0), "b")]);
        select_request_candidates(&mut invocation).unwrap();
        apply_pool_policy(&mut invocation.candidates);
        assert_eq!(invocation.candidates.len(), 1);
        assert_eq!(invocation.candidates[0].model_id, "tools");
    }

    #[test]
    fn absent_null_or_empty_tools_do_not_filter_candidates() {
        for tools in [None, Some(Value::Null), Some(json!([]))] {
            let mut unknown = candidate("unknown", Some(70.0), "a");
            unknown.supports_tools = false;
            let mut invocation = invocation(vec![unknown, candidate("tools", Some(8.0), "b")]);
            if let Some(tools) = tools {
                invocation.request["tools"] = tools;
            }
            select_request_candidates(&mut invocation).unwrap();
            assert_eq!(invocation.candidates.len(), 2);
        }
    }

    #[test]
    fn tools_fallback_never_relaxes_media_requirements() {
        for key in ["image_url", "input_audio"] {
            let mut worker = candidate("text", Some(27.0), "a");
            worker.supports_tools = false;
            let mut invocation = tools_invocation(vec![worker]);
            invocation.request["messages"][0]["content"] = json!([{key: {"url": "fixture"}}]);
            let response = select_request_candidates(&mut invocation).unwrap_err();
            assert_eq!(response.status_code, 422);
        }
    }

    #[test]
    fn tools_preference_is_computed_after_media_filtering() {
        for key in ["image_url", "input_audio"] {
            let text = candidate("text-tools", Some(70.0), "a");
            let mut media = candidate("media", Some(27.0), "b");
            media.supports_tools = false;
            media.supports_vision = true;
            media.supports_audio = true;
            let mut invocation = tools_invocation(vec![text, media]);
            invocation.request["messages"][0]["content"] = json!([{key: {"url": "fixture"}}]);
            select_request_candidates(&mut invocation).unwrap();
            assert_eq!(invocation.candidates.len(), 1);
            assert_eq!(invocation.candidates[0].model_id, "media");
        }
    }

    #[tokio::test]
    async fn direct_route_retries_a_failed_or_disappeared_replica() {
        for first_fails_with_response in [true, false] {
            let attempts = Arc::new(std::sync::Mutex::new(Vec::new()));
            let seen = Arc::clone(&attempts);
            let response = direct_with_fallback(
                invocation(vec![
                    candidate("small", Some(8.0), "a"),
                    candidate("small", Some(8.0), "b"),
                ]),
                false,
                Duration::from_secs(1),
                move |request| {
                    let seen = Arc::clone(&seen);
                    async move {
                        seen.lock().unwrap().push(request.target_node_id.clone());
                        assert!(request.timeout_ms.is_some_and(|ms| ms <= 1000));
                        if request.target_node_id.as_deref() == Some("a") {
                            if first_fails_with_response {
                                return Ok(mesh_llm_plugin::HostInferenceResponse {
                                    status_code: 503,
                                    body: json!({"error": "unavailable"}),
                                    served_by: None,
                                });
                            }
                            anyhow::bail!("target disappeared after discovery");
                        }
                        Ok(mesh_llm_plugin::HostInferenceResponse {
                            status_code: 200,
                            body: json!({"model": "small", "choices": [{"message": {"content": "healthy"}}]}),
                            served_by: Some("b".into()),
                        })
                    }
                },
            )
            .await;
            assert_eq!(response.status_code, 200);
            assert_eq!(response.body["choices"][0]["message"]["content"], "healthy");
            assert_eq!(response.headers, [("x-mesh-served-by".into(), "b".into())]);
            assert_eq!(
                *attempts.lock().unwrap(),
                [Some("a".into()), Some("b".into())]
            );
        }
    }

    #[tokio::test]
    async fn direct_route_does_not_retry_a_non_retryable_response() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::clone(&attempts);
        let response = direct_with_fallback(
            invocation(vec![
                candidate("small", Some(8.0), "a"),
                candidate("small", Some(8.0), "b"),
            ]),
            false,
            Duration::from_secs(1),
            move |_| {
                let seen = Arc::clone(&seen);
                async move {
                    seen.fetch_add(1, Ordering::Relaxed);
                    Ok(mesh_llm_plugin::HostInferenceResponse {
                        status_code: 402,
                        body: json!({"error": "payment required"}),
                        served_by: None,
                    })
                }
            },
        )
        .await;
        assert_eq!(response.status_code, 402);
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn direct_route_has_one_shared_deadline() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::clone(&attempts);
        let response = direct_with_fallback(
            invocation(vec![
                candidate("small", Some(8.0), "a"),
                candidate("small", Some(8.0), "b"),
            ]),
            false,
            Duration::from_millis(20),
            move |_| {
                let seen = Arc::clone(&seen);
                async move {
                    seen.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    anyhow::bail!("late")
                }
            },
        )
        .await;
        assert_eq!(response.status_code, 504);
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn capable_committee_is_capped_and_keeps_same_model_replicas() {
        let mut candidates = vec![
            candidate("big", Some(70.0), "node-a"),
            candidate("big", Some(70.0), "node-b"),
            candidate("medium-a", Some(32.0), "node-c"),
            candidate("medium-b", Some(24.0), "node-d"),
            candidate("small", Some(8.0), "node-e"),
        ];

        apply_pool_policy(&mut candidates);

        assert_eq!(candidates.len(), COMMITTEE_CAP);
        assert_eq!(candidates[0].model_id, "big");
        assert_eq!(candidates[1].model_id, "big");
        assert_ne!(
            candidates[0].target_node_id, candidates[1].target_node_id,
            "physical replicas must remain distinct committee members"
        );
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.model_id != "small")
        );
    }

    #[test]
    fn committee_cap_keeps_ready_replica_ahead_of_deprioritized_large_ones() {
        let mut candidates = (0..COMMITTEE_CAP)
            .map(|index| candidate("big", Some(70.0), &format!("deprioritized-{index}")))
            .collect::<Vec<_>>();
        for candidate in &mut candidates {
            candidate.deprioritized = true;
        }
        candidates.push(candidate("ready", Some(32.0), "ready"));
        assert!(!apply_pool_policy(&mut candidates));
        assert_eq!(candidates.len(), COMMITTEE_CAP);
        assert_eq!(candidates[0].model_id, "ready");
    }
}
