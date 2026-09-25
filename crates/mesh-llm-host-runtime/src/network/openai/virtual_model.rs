use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::transport as proxy;
use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_plugin::{HostInferenceRequest, HostInferenceResponse};
use mesh_llm_plugin::{VirtualModelCandidate, VirtualModelInvocation};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

pub(crate) fn advertisable_routes(
    routes: Vec<crate::plugin::VirtualModelRoute>,
    concrete_models: &[String],
) -> Vec<crate::plugin::VirtualModelRoute> {
    let has_candidates = !concrete_models.is_empty();
    routes
        .into_iter()
        .filter(|route| !route.requires_candidates || has_candidates)
        .collect()
}

#[derive(Clone)]
pub(crate) struct InferenceRpcBridge {
    api_port: u16,
    plugins: PluginManager,
    client: reqwest::Client,
}

impl InferenceRpcBridge {
    pub(crate) fn new(api_port: u16, plugins: PluginManager) -> Self {
        Self {
            api_port,
            plugins,
            client: reqwest::Client::new(),
        }
    }

    async fn infer(&self, params_json: &str) -> Result<RpcResult, proto::ErrorResponse> {
        let mut request: HostInferenceRequest = serde_json::from_str(params_json)
            .map_err(|error| invalid_params(format!("invalid inference request: {error}")))?;
        let model_id = request.model_id.trim().to_string();
        if model_id.is_empty() {
            return Err(invalid_params("model_id is required"));
        }
        if super::automatic::is_directive(&model_id)
            || self
                .plugins
                .virtual_model_for_model(&model_id)
                .await
                .map_err(internal)?
                .is_some()
        {
            return Err(invalid_params(format!(
                "nested inference cannot target virtual model '{}'",
                model_id
            )));
        }
        let object = request
            .request
            .as_object_mut()
            .ok_or_else(|| invalid_params("request must be a JSON object"))?;
        object.insert("model".into(), model_id.into());
        object.insert("stream".into(), false.into());
        let timeout = std::time::Duration::from_millis(
            request.timeout_ms.unwrap_or(60_000).clamp(1, 300_000),
        );
        let response = self
            .client
            .post(format!(
                "http://127.0.0.1:{}/v1/chat/completions",
                self.api_port
            ))
            .timeout(timeout)
            .json(&request.request)
            .send()
            .await
            .map_err(internal)?;
        let status_code = response.status().as_u16();
        let served_by = response
            .headers()
            .get("x-mesh-served-by")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = response.json().await.map_err(internal)?;
        let result = HostInferenceResponse {
            status_code,
            body,
            served_by,
        };
        Ok(RpcResult {
            result_json: serde_json::to_string(&result).map_err(internal)?,
        })
    }
}

impl PluginRpcBridge for InferenceRpcBridge {
    fn handle_request(
        &self,
        _plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        let this = self.clone();
        Box::pin(async move {
            match method.as_str() {
                "inference/chat_completions" => this.infer(&params_json).await,
                _ => Err(proto::ErrorResponse {
                    code: -32601,
                    message: format!("Unsupported host RPC method '{method}'"),
                    data_json: String::new(),
                }),
            }
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

fn invalid_params(message: impl Into<String>) -> proto::ErrorResponse {
    proto::ErrorResponse {
        code: -32602,
        message: message.into(),
        data_json: String::new(),
    }
}

fn internal(error: impl std::fmt::Display) -> proto::ErrorResponse {
    proto::ErrorResponse {
        code: -32603,
        message: error.to_string(),
        data_json: String::new(),
    }
}

pub(crate) async fn install_inference_bridge(plugins: &PluginManager, api_port: u16) {
    plugins
        .set_rpc_bridge(Some(Arc::new(InferenceRpcBridge::new(
            api_port,
            plugins.clone(),
        ))))
        .await;
}

pub(crate) enum VirtualModelDispatchResult {
    NotVirtual(ClientStream),
    Responded(proxy::RouteDispatchOutcome),
}

// `RouteDispatchOutcome` is deliberately `Copy`; its usage-plus-output-digests variant
// (three optional 32-byte digests inline) exceeds clippy's 128-byte `Err` threshold.
#[allow(clippy::result_large_err)]
pub(crate) async fn route_virtual_model_or_passthrough(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<ClientStream, proxy::RouteDispatchOutcome> {
    if request.is_tokenize_request() {
        return Ok(tcp_stream);
    }
    let Some(mut model_id) = request.model_name.clone() else {
        return Ok(tcp_stream);
    };
    if super::automatic::is_directive(&model_id) {
        super::automatic::warn_if_deprecated_alias(Some(&model_id));
        request.ensure_body_json();
        let Some(body) = request.body_json.as_ref() else {
            return Ok(tcp_stream);
        };
        if matches!(
            super::automatic::serving_mode(super::automatic::AutomaticRequest {
                model: Some(&model_id),
                path: &request.path,
                body,
            }),
            super::automatic::ServingMode::SingleModel(_)
        ) {
            return Ok(tcp_stream);
        }
        model_id = super::automatic::DIRECTIVE.to_string();
    }
    let Some(plugin_manager) = node.plugin_manager().await else {
        return Ok(tcp_stream);
    };
    if plugin_manager
        .virtual_model_for_model(&model_id)
        .await
        .ok()
        .flatten()
        .is_none()
    {
        return Ok(tcp_stream);
    }
    if super::ingress::mesh_routing_headers_requested(request) {
        let write = proxy::send_error_observed(
            tcp_stream,
            409,
            "x-mesh-target/x-mesh-exclude are not supported for virtual models",
            route_observer,
        )
        .await;
        return Err(if write.is_ok() {
            proxy::RouteDispatchOutcome::Responded(409)
        } else {
            proxy::RouteDispatchOutcome::Dropped("virtual_model_response_write_failed")
        });
    }
    request.ensure_body_json();
    let Some(body) = request.body_json.clone() else {
        let write = proxy::send_400_observed(
            tcp_stream,
            "virtual models require a JSON body",
            route_observer,
        )
        .await;
        return Err(if write.is_ok() {
            proxy::RouteDispatchOutcome::Responded(400)
        } else {
            proxy::RouteDispatchOutcome::Dropped("virtual_model_response_write_failed")
        });
    };
    let mut candidates = node.models_being_served().await;
    candidates.extend(node.serving_models().await);
    if let Ok(inference_models) = plugin_manager.inference_models().await {
        candidates.extend(inference_models);
    }
    candidates.extend(
        node.all_served_model_descriptors()
            .await
            .into_iter()
            .map(|descriptor| descriptor.identity.model_name),
    );
    match try_handle_virtual_model(
        &plugin_manager,
        node,
        tcp_stream,
        &request.path,
        &model_id,
        body,
        candidates,
        request.response_adapter,
        route_observer,
    )
    .await
    {
        VirtualModelDispatchResult::NotVirtual(stream) => Ok(stream),
        VirtualModelDispatchResult::Responded(outcome) => Err(outcome),
    }
}

fn validate_virtual_request(
    forwarded_path: &str,
    model_id: &str,
    route: &crate::plugin::VirtualModelRoute,
    request_body: &serde_json::Value,
    candidate_models: &[String],
) -> Option<(u16, String)> {
    if !super::automatic::is_chat_shaped_path(forwarded_path) {
        return Some((
            422,
            format!("virtual model '{model_id}' supports chat-shaped requests only"),
        ));
    }
    if candidate_models
        .iter()
        .any(|candidate| candidate == model_id)
    {
        return Some((
            409,
            format!("virtual model '{model_id}' collides with a concrete model"),
        ));
    }
    let requests_tools = request_body.get("tools").is_some_and(|tools| {
        !tools.is_null() && tools.as_array().is_none_or(|items| !items.is_empty())
    });
    if requests_tools && !route.supports_tools {
        return Some((
            422,
            format!("virtual model '{model_id}' does not support tools"),
        ));
    }
    let requests_stream = request_body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if requests_stream && !route.supports_streaming {
        return Some((
            422,
            format!("virtual model '{model_id}' does not support streaming"),
        ));
    }
    None
}

#[allow(clippy::cognitive_complexity, clippy::too_many_arguments)]
pub(crate) async fn try_handle_virtual_model(
    plugins: &PluginManager,
    node: &mesh::Node,
    tcp_stream: ClientStream,
    forwarded_path: &str,
    model_id: &str,
    request_body: serde_json::Value,
    candidate_models: Vec<String>,
    response_adapter: proxy::ResponseAdapter,
    route_observer: OpenAiRouteObserver<'_>,
) -> VirtualModelDispatchResult {
    let route = match plugins.virtual_model_for_model(model_id).await {
        Ok(Some(route)) => route,
        Ok(None) => return VirtualModelDispatchResult::NotVirtual(tcp_stream),
        Err(error) => {
            let result = proxy::send_error_observed(
                tcp_stream,
                503,
                &format!("virtual model registry error: {error}"),
                route_observer,
            )
            .await;
            return VirtualModelDispatchResult::Responded(response_outcome(503, result));
        }
    };
    if let Some((status, message)) = validate_virtual_request(
        forwarded_path,
        model_id,
        &route,
        &request_body,
        &candidate_models,
    ) {
        let result = proxy::send_error_observed(tcp_stream, status, &message, route_observer).await;
        return VirtualModelDispatchResult::Responded(response_outcome(status, result));
    }
    let virtual_ids = plugins
        .virtual_models()
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|route| route.model_id)
        .collect::<std::collections::BTreeSet<_>>();
    let requests_stream = request_body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let descriptors = node.all_served_model_descriptors().await;
    let runtimes = node.all_model_runtime_descriptors().await;
    let mut candidates = candidate_models
        .into_iter()
        .filter(|candidate| !virtual_ids.contains(candidate))
        .map(|model_id| {
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.identity.model_name == model_id);
            let runtime = runtimes
                .iter()
                .find(|runtime| runtime.model_name == model_id);
            VirtualModelCandidate {
                model_id,
                parameter_count_b: descriptor
                    .and_then(|descriptor| descriptor.metadata.as_ref())
                    .and_then(|metadata| metadata.parameter_count_b),
                context_length: runtime.and_then(|runtime| runtime.advertised_context_length()),
                supports_tools: descriptor.is_some_and(|descriptor| {
                    descriptor.capabilities.tool_use != crate::models::CapabilityLevel::None
                }),
                supports_vision: descriptor
                    .is_some_and(|descriptor| descriptor.capabilities.supports_vision_runtime()),
                supports_audio: descriptor
                    .is_some_and(|descriptor| descriptor.capabilities.supports_audio_runtime()),
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.model_id.cmp(&right.model_id));
    candidates.dedup_by(|left, right| left.model_id == right.model_id);
    let invocation = VirtualModelInvocation {
        request: request_body,
        candidates,
        response_adapter: format!("{response_adapter:?}"),
    };
    let input_json = match serde_json::to_string(&invocation) {
        Ok(value) => value,
        Err(error) => {
            let result = proxy::send_error_observed(
                tcp_stream,
                500,
                &format!("failed to encode virtual model request: {error}"),
                route_observer,
            )
            .await;
            return VirtualModelDispatchResult::Responded(response_outcome(500, result));
        }
    };
    let response = match plugins
        .invoke_virtual_model(
            &route,
            &input_json,
            Some(std::time::Duration::from_secs(300)),
        )
        .await
    {
        Ok(response) => response,
        Err(error) => {
            let result = proxy::send_error_observed(
                tcp_stream,
                502,
                &format!("virtual model '{}' failed: {error}", route.model_id),
                route_observer,
            )
            .await;
            return VirtualModelDispatchResult::Responded(response_outcome(502, result));
        }
    };
    if !(200..=599).contains(&response.status_code) {
        let result = proxy::send_error_observed(
            tcp_stream,
            502,
            &format!(
                "virtual model '{}' returned invalid HTTP status {}",
                route.model_id, response.status_code
            ),
            route_observer,
        )
        .await;
        return VirtualModelDispatchResult::Responded(response_outcome(502, result));
    }
    let headers = response
        .headers
        .iter()
        .take(32)
        .filter(|(name, value)| virtual_response_header_allowed(name, value))
        .map(|(name, value)| (name.as_str(), value.clone()))
        .collect::<Vec<_>>();
    let status = response.status_code;
    let write = if requests_stream && (200..300).contains(&status) {
        match response_adapter {
            proxy::ResponseAdapter::OpenAiResponsesStream => {
                send_responses_sse(tcp_stream, &response.body, &headers).await
            }
            _ => send_chat_sse(tcp_stream, &response.body, &headers).await,
        }
    } else {
        let body = if response_adapter == proxy::ResponseAdapter::OpenAiResponsesJson
            && (200..300).contains(&status)
        {
            chat_completion_to_responses_json(&response.body)
        } else {
            response.body.clone()
        };
        proxy::send_json_with_status_and_headers_observed(
            tcp_stream,
            status,
            &body,
            &headers,
            route_observer,
        )
        .await
    };
    if write.is_err() {
        return VirtualModelDispatchResult::Responded(proxy::RouteDispatchOutcome::Dropped(
            "virtual_model_response_write_failed",
        ));
    }
    let usage = parse_usage(&response.body);
    let outcome = if !(200..300).contains(&status) {
        proxy::RouteDispatchOutcome::FailedWithStatus {
            status_code: status,
            reason: "virtual_model_failed",
        }
    } else if let Some(usage) = usage {
        proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: status,
            usage,
            output_digests: Default::default(),
        }
    } else {
        proxy::RouteDispatchOutcome::Responded(status)
    };
    VirtualModelDispatchResult::Responded(outcome)
}

fn virtual_response_header_allowed(name: &str, value: &str) -> bool {
    if value.len() > 8 * 1024 || !proxy::is_valid_header_name(name) {
        return false;
    }
    !matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "content-length"
            | "content-type"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn response_outcome(status: u16, result: std::io::Result<()>) -> proxy::RouteDispatchOutcome {
    if result.is_ok() {
        proxy::RouteDispatchOutcome::Responded(status)
    } else {
        proxy::RouteDispatchOutcome::Dropped("virtual_model_response_write_failed")
    }
}

fn parse_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    super::response::parse_token_usage_from_json_body(body.to_string().as_bytes())
}

async fn write_sse_headers(
    stream: &mut ClientStream,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Transfer-Encoding: chunked\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n",
    );
    for (name, value) in extra_headers {
        proxy::append_safe_header(&mut header, name, value);
    }
    header.push_str("\r\n");
    stream.write_all(header.as_bytes()).await
}

async fn write_sse_event(
    stream: &mut ClientStream,
    event: &serde_json::Value,
) -> std::io::Result<()> {
    write_sse_data(stream, &event.to_string()).await
}

async fn write_sse_data(stream: &mut ClientStream, data: &str) -> std::io::Result<()> {
    let payload = format!("data: {data}\n\n");
    let framed = format!("{:x}\r\n{}\r\n", payload.len(), payload);
    stream.write_all(framed.as_bytes()).await
}

async fn finish_sse(stream: &mut ClientStream) -> std::io::Result<()> {
    write_sse_data(stream, "[DONE]").await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await
}

async fn send_chat_sse(
    mut stream: ClientStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    write_sse_headers(&mut stream, extra_headers).await?;
    let id = response
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("chatcmpl-virtual");
    let model = response
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("virtual-model");
    let message = response
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let tool_calls = message.get("tool_calls").cloned();
    let delta = match tool_calls.as_ref() {
        Some(tool_calls) => serde_json::json!({
            "role": "assistant",
            "tool_calls": tool_calls,
        }),
        None => serde_json::json!({
            "role": "assistant",
            "content": message.get("content").cloned().unwrap_or(serde_json::Value::String(String::new())),
        }),
    };
    write_sse_event(
        &mut stream,
        &serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
        }),
    )
    .await?;
    write_sse_event(
        &mut stream,
        &serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": if tool_calls.is_some() { "tool_calls" } else { "stop" },
            }],
        }),
    )
    .await?;
    finish_sse(&mut stream).await
}

async fn send_responses_sse(
    mut stream: ClientStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    write_sse_headers(&mut stream, extra_headers).await?;
    let response_id = response
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("resp_virtual");
    let model = response
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("virtual-model");
    let content = response
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let usage = response
        .get("usage")
        .map(openai_frontend::responses::chat_usage_to_responses_usage);
    let item_id = format!("msg_{response_id}");
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    use openai_frontend::responses as resp;
    let mut created = resp::responses_stream_created_event_with_sequence(model, created_at, 0);
    if let Some(object) = created
        .get_mut("response")
        .and_then(serde_json::Value::as_object_mut)
    {
        object.insert("id".into(), response_id.into());
    }
    let events = [
        created,
        resp::responses_stream_delta_event_with_logprobs_and_sequence(&item_id, content, None, 1),
        resp::responses_stream_text_done_event_with_sequence(&item_id, content, 2),
        resp::responses_stream_completed_event_with_sequence(
            response_id,
            created_at,
            model,
            &item_id,
            content,
            usage,
            3,
        ),
    ];
    for event in &events {
        write_sse_event(&mut stream, event).await?;
    }
    finish_sse(&mut stream).await
}

fn chat_completion_to_responses_json(chat: &serde_json::Value) -> serde_json::Value {
    let bytes = serde_json::to_vec(chat).unwrap_or_default();
    match super::response_adapter::translate_chat_completion_to_responses(&bytes) {
        Ok(translated) => serde_json::from_slice(&translated).unwrap_or_else(|_| chat.clone()),
        Err(error) => {
            tracing::warn!("virtual-model response translation failed: {error}");
            chat.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(model_id: &str, requires_candidates: bool) -> crate::plugin::VirtualModelRoute {
        crate::plugin::VirtualModelRoute {
            plugin_name: "test-plugin".into(),
            model_id: model_id.into(),
            handler: "handle".into(),
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            supports_tools: false,
            supports_streaming: false,
            requires_candidates,
        }
    }

    #[test]
    fn candidate_dependent_routes_are_advertised_only_when_candidates_exist() {
        let routes = vec![route("standalone", false), route("dependent", true)];

        assert_eq!(
            advertisable_routes(routes.clone(), &[]),
            vec![route("standalone", false)]
        );
        assert_eq!(
            advertisable_routes(routes.clone(), &["concrete".into()]),
            routes
        );
    }

    #[test]
    fn virtual_response_headers_cannot_override_http_framing() {
        assert!(virtual_response_header_allowed(
            "x-plugin-route",
            "worker-a"
        ));
        assert!(!virtual_response_header_allowed("content-length", "0"));
        assert!(!virtual_response_header_allowed(
            "Transfer-Encoding",
            "chunked"
        ));
        assert!(!virtual_response_header_allowed("bad\r\nname", "value"));
        assert!(!virtual_response_header_allowed(
            "x-too-large",
            &"x".repeat(8 * 1024 + 1)
        ));
    }
}
