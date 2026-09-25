use mesh_llm_plugin::{
    HostInferenceRequest, PluginContext, PluginMetadata, PluginRuntime, SimplePlugin,
    VirtualModelInvocation, VirtualModelResponse, VirtualModelRouter, operation_with_schema,
    plugin_server_info, structured_tool_result, virtual_model,
};
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

pub const PLUGIN_ID: &str = "mesh-moa";
pub const HANDLER: &str = "chat";

pub async fn run(stream: mesh_llm_plugin::LocalStream) -> anyhow::Result<()> {
    PluginRuntime::run_with_stream(plugin(), stream).await
}

pub fn plugin() -> SimplePlugin {
    let manifest = mesh_llm_plugin::plugin_manifest![
        virtual_model(moa::VIRTUAL_MODEL_NAME, HANDLER)
            .supports_tools(true)
            .supports_streaming(true)
            .input_modalities(["text", "image", "audio"])
    ];
    let mut router = VirtualModelRouter::new();
    router.add_raw(
        operation_with_schema(HANDLER, "Run a Mesh MoA turn", serde_json::Map::new()),
        |request, context| {
            let context = context.owned();
            Box::pin(async move {
                let invocation: VirtualModelInvocation = request.arguments()?;
                let response = handle(invocation, context).await;
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
}

async fn handle(
    mut invocation: VirtualModelInvocation,
    context: PluginContext<'static>,
) -> VirtualModelResponse {
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

    let needs_vision = contains_any_key(&invocation.request, &["image_url", "input_image"]);
    let needs_audio = contains_any_key(
        &invocation.request,
        &["audio_url", "input_audio", "input_audio_buffer"],
    );
    let needs_tools = invocation.request.get("tools").is_some_and(|tools| {
        !tools.is_null() && tools.as_array().is_none_or(|items| !items.is_empty())
    });
    invocation.candidates.retain(|candidate| {
        (!needs_vision || candidate.supports_vision)
            && (!needs_audio || candidate.supports_audio)
            && (!needs_tools || candidate.supports_tools)
    });
    if invocation.candidates.is_empty() {
        return error_response(422, "no concrete model satisfies the request capabilities");
    }
    // A one-model pool has nothing to aggregate. Passing it through the MoA
    // engine would replace the caller's output budget with the Generalist
    // worker budget (1,024 tokens), which can make an otherwise valid request
    // ineligible for a small-context target before inference even starts.
    // Keep placement and admission host-governed, but preserve the original
    // request shape when there is no actual committee to convene.
    if invocation.candidates.len() == 1 || needs_vision || needs_audio {
        return direct_capability_response(invocation, context, requested_stream).await;
    }

    let mut backends: Vec<Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for candidate in invocation.candidates {
        let index = backends.len();
        backends.push(Arc::new(HostBackend {
            context: context.clone(),
        }));
        models.push(
            moa::ModelEntry::new(candidate.model_id, index)
                .with_parameter_count_b(candidate.parameter_count_b),
        );
    }
    let actor_candidates = actor_candidates(&models);
    let config = moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(60),
        reducer_timeout: Duration::from_secs(60),
        hedge_delay: Duration::from_secs(5),
        first_answer_grace: Duration::from_secs(10),
        strong_patience: Duration::from_secs(20),
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

async fn direct_capability_response(
    mut invocation: VirtualModelInvocation,
    context: PluginContext<'static>,
    requested_stream: bool,
) -> VirtualModelResponse {
    invocation.candidates.sort_by(|left, right| {
        right
            .parameter_count_b
            .partial_cmp(&left.parameter_count_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.model_id.cmp(&right.model_id))
    });
    let candidate = &invocation.candidates[0];
    match context
        .infer(HostInferenceRequest {
            model_id: candidate.model_id.clone(),
            request: invocation.request,
            timeout_ms: Some(60_000),
        })
        .await
    {
        Ok(response) => VirtualModelResponse {
            status_code: response.status_code,
            body: response.body,
            headers: response
                .served_by
                .map(|served_by| vec![("x-mesh-served-by".into(), served_by)])
                .unwrap_or_default(),
            event_stream: requested_stream,
        },
        Err(error) => error_response(502, &format!("capability route failed: {error}")),
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
}
