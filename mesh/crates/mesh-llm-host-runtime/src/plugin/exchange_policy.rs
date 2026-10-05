//! Compose core hooks and plugin observation on the real typed serving backend.
use super::ExchangeSession;
use async_trait::async_trait;
use serde_json::json;
use skippy_inference_api::http_exchange::{HttpExchangeAdmission, HttpExchangePolicy};
use skippy_inference_api::wire_bytes::{WireBytesCommitment, WireBytesObserver};
use skippy_inference_api::{
    CapsuleMarker, ChatCompletionOutcome, ChatCompletionRequest, ChatCompletionResponse,
    ChatExchangeRoute, OpenAiHookPolicy, OpenAiResult,
};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) struct NodeExchangePolicy {
    node: crate::mesh::Node,
}
impl NodeExchangePolicy {
    pub(crate) fn new(node: crate::mesh::Node) -> Arc<Self> {
        Arc::new(Self { node })
    }
    async fn legacy_bridge(&self) -> Option<super::openai_exchange::OpenAiExchangeHookBridge> {
        self.node
            .plugin_manager()
            .await
            .map(|manager| super::openai_exchange::OpenAiExchangeHookBridge::new(Arc::new(manager)))
    }
    async fn admit_typed_request(
        &self,
        request: &(impl serde::Serialize + Sync),
        model: &str,
        endpoint: &str,
        path: &str,
        exchange_id: &str,
    ) -> OpenAiResult<()> {
        let Some(manager) = self.node.plugin_manager().await else {
            return Ok(());
        };
        if !manager.has_exchange_hooks().await {
            return Ok(());
        }
        let bytes = serde_json::to_vec(request).map_err(|_| {
            skippy_inference_api::OpenAiError::backend("cannot encode effective request")
        })?;
        let mut event = super::request_event(
            exchange_id.to_owned(),
            endpoint,
            "POST",
            path,
            &bytes,
            Default::default(),
            false,
        );
        event["phase"] = json!("backend_selected");
        event["observation_point"] = json!("backend_dispatch");
        event["dispatch_path"] = json!("typed_frontend");
        event["backend_exchange_id"] = json!(exchange_id);
        event["model"] = json!(model);
        event["provider"] = json!("native_runtime");
        event["target"] = json!("local");
        event["attempt"] = json!(0);
        event["effective_request_wire_digest"] = event["request_wire_digest"].take();
        event["effective_request_encoding"] = json!("typed_json_serialization");
        event.as_object_mut().unwrap().remove("request_wire_digest");
        let result =
            match skippy_inference_api::http_exchange::current_http_exchange_observation_id() {
                Some(id) => manager.selected_exchange_phase(&id, event).await,
                None => manager.exchange_phase(event).await,
            };
        match result.error_status() {
            Some(403) => Err(permission_error(
                "OpenAI plugin policy denied the selected backend",
            )),
            Some(_) => Err(hook_unavailable(
                "required OpenAI plugin admission unavailable",
            )),
            None => Ok(()),
        }
    }
}
#[async_trait]
impl OpenAiHookPolicy for NodeExchangePolicy {
    fn requires_exchange_lifecycle(&self) -> bool {
        true
    }
    fn http_exchange_policy(&self) -> Option<Arc<dyn HttpExchangePolicy>> {
        Some(Arc::new(self.clone()))
    }
    async fn admit_effective_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) -> OpenAiResult<()> {
        self.admit_typed_request(
            request,
            &request.model,
            "chat_completions",
            "/v1/chat/completions",
            &route.exchange_id,
        )
        .await
    }
    async fn admit_effective_completion(
        &self,
        request: &skippy_inference_api::CompletionRequest,
        exchange_id: &str,
    ) -> OpenAiResult<()> {
        self.admit_typed_request(
            request,
            &request.model,
            "completions",
            "/v1/completions",
            exchange_id,
        )
        .await
    }
    async fn on_effective_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) {
        if let Some(bridge) = self.legacy_bridge().await {
            bridge.on_effective_chat_completion(request, route).await;
        }
    }
    async fn on_chat_completion_terminal(
        &self,
        request: &ChatCompletionRequest,
        id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        if let Some(manager) = self.node.plugin_manager().await {
            if let ChatCompletionOutcome::Success { response } = outcome {
                manager.record_typed_usage(id, json!(response.usage));
            }
            let execution_outcome = match outcome {
                ChatCompletionOutcome::Success { .. } | ChatCompletionOutcome::StreamCompleted => {
                    "completed"
                }
                ChatCompletionOutcome::Error { status: 504, .. } => "timed_out",
                ChatCompletionOutcome::Error { .. } => "backend_error",
                ChatCompletionOutcome::Denied { status: 403, .. } => "policy_denied",
                ChatCompletionOutcome::Denied { .. } => "internal_hook_failure",
                ChatCompletionOutcome::Cancelled => "client_cancelled",
                _ => "evidence_unavailable",
            };
            manager.record_typed_execution_outcome(id, execution_outcome);
        }
        if let Some(bridge) = self.legacy_bridge().await {
            bridge
                .on_chat_completion_terminal(request, id, outcome)
                .await;
        }
    }
    async fn on_completion_terminal(&self, exchange_id: &str, outcome: &str) {
        if let Some(manager) = self.node.plugin_manager().await {
            manager.record_typed_execution_outcome(exchange_id, outcome);
        }
    }
    async fn capsule_marker_for_response(
        &self,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        match self.legacy_bridge().await {
            Some(bridge) => bridge.capsule_marker_for_response(request, response).await,
            None => None,
        }
    }
}

#[async_trait]
impl HttpExchangePolicy for NodeExchangePolicy {
    async fn is_enabled(&self) -> bool {
        match self.node.plugin_manager().await {
            Some(manager) => manager.has_exchange_hooks().await,
            None => false,
        }
    }
    async fn received(
        &self,
        method: &str,
        path: &str,
        headers: &axum::http::HeaderMap,
        body: &[u8],
        request_id: skippy_inference_api::RequestId,
    ) -> HttpExchangeAdmission {
        let Some(manager) = self.node.plugin_manager().await else {
            return HttpExchangeAdmission::default();
        };
        if !manager.has_exchange_hooks().await {
            return HttpExchangeAdmission::default();
        }
        let endpoint = match path.split('?').next() {
            Some("/v1/chat/completions") => "chat_completions",
            Some("/v1/completions") => "completions",
            Some("/v1/responses") => "responses",
            _ => return HttpExchangeAdmission::default(),
        };
        let sanitized = headers
            .iter()
            .filter_map(|(name, value)| {
                (mesh_llm_config::safe_exchange_header(name.as_str()))
                    .then(|| {
                        value
                            .to_str()
                            .ok()
                            .map(|v| (name.to_string(), v.to_string()))
                    })
                    .flatten()
            })
            .collect();
        let mut event = super::request_event(
            request_id.as_uuid().to_string(),
            endpoint,
            method,
            path,
            body,
            sanitized,
            false,
        );
        event["node_endpoint_id"] = json!(self.node.id().to_string());
        event["emission_boundary"] = json!("http_body_poll");
        event["dispatch_path"] = json!("typed_frontend");
        let (session, result) = ExchangeSession::begin(&manager, event).await;
        let denial = result.error_status().map(|status| {
            if status == 403 {
                permission_error("OpenAI plugin policy denied the exchange")
            } else {
                hook_unavailable("required OpenAI plugin admission unavailable")
            }
        });
        let terminal_override = result.error_status().map(|status| {
            if status == 403 {
                "policy_denied"
            } else {
                "internal_hook_failure"
            }
        });
        HttpExchangeAdmission {
            observation_id: Some(session.observation_id().to_owned()),
            observer: Some(Arc::new(TypedEmission {
                inner: session.observer(),
                session: Mutex::new(Some(session)),
                terminal_override,
            })),
            denial,
            response_headers: result.headers,
        }
    }
}
struct TypedEmission {
    inner: Arc<dyn WireBytesObserver>,
    session: Mutex<Option<ExchangeSession>>,
    terminal_override: Option<&'static str>,
}
impl WireBytesObserver for TypedEmission {
    fn response_headers(&self) -> Vec<(String, String)> {
        self.inner.response_headers()
    }
    fn response_status(&self, status: u16) {
        self.inner.response_status(status);
    }
    fn try_chunk(&self, offset: u64, bytes: &[u8]) -> bool {
        self.inner.try_chunk(offset, bytes)
    }
    fn finish(&self, commitment: WireBytesCommitment) {
        self.inner.finish(commitment.clone());
        if let Some(mut session) = self.session.lock().unwrap().take() {
            let outcome = self
                .terminal_override
                .unwrap_or(match commitment.incomplete {
                    Some(skippy_inference_api::wire_bytes::WireBytesIncomplete::Cancelled) => {
                        "client_cancelled"
                    }
                    Some(skippy_inference_api::wire_bytes::WireBytesIncomplete::Timeout) => {
                        "timed_out"
                    }
                    Some(
                        skippy_inference_api::wire_bytes::WireBytesIncomplete::TransportError
                        | skippy_inference_api::wire_bytes::WireBytesIncomplete::InvalidFraming,
                    ) => "transport_error",
                    _ => session.outcome_from_status(),
                });
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    session.finish(outcome).await;
                });
            }
        }
    }
}

pub(crate) fn compose_node_hooks(node: crate::mesh::Node) -> Arc<dyn OpenAiHookPolicy> {
    skippy_inference_api::composite_hooks::CompositeOpenAiHookPolicy::new(vec![
        crate::inference::skippy::MeshAutoHookPolicy::new(node.clone()),
        NodeExchangePolicy::new(node),
    ])
}

fn permission_error(message: &str) -> skippy_inference_api::OpenAiError {
    skippy_inference_api::OpenAiError::from_kind(
        axum::http::StatusCode::FORBIDDEN,
        skippy_inference_api::OpenAiErrorKind::Permission,
        message,
    )
    .with_code("plugin_policy_denied")
}

fn hook_unavailable(message: &str) -> skippy_inference_api::OpenAiError {
    skippy_inference_api::OpenAiError::from_kind(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        skippy_inference_api::OpenAiErrorKind::ServiceUnavailable,
        message,
    )
    .with_code("plugin_hook_unavailable")
}
