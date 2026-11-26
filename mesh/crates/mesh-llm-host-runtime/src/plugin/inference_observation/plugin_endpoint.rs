//! Observation bridge for plugin-hosted OpenAI inference endpoints.
//!
//! Native Skippy frontends report to [`ObservationHub`] through
//! [`super::adapter::ObservationAdapter`]. Plugin-hosted HTTP inference
//! endpoints (such as `openai_compatible` and `transcription_v1` plugins) run
//! locally on this node through `route_http_endpoint_request` and publish
//! their healthy models through `PluginManager::refresh_plugin_endpoints`.
//! This module bridges both boundaries into the same [`ObservationHub`].

use super::hub::ObservationHub;
use super::proto;
use crate::network::openai::request_parse::BufferedHttpRequest;
use crate::plugin::PluginEndpointSummary;
use mesh_llm_events::logging::events::TokenUsage;
use skippy_inference_api::{RequestId, generate_request_id};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use url::{Host, Url};

/// Whether a plugin inference endpoint protocol maps to `InferenceOperation`
/// events in `local_inference_observation.v1`.
pub(crate) fn is_observable_inference_protocol(protocol: Option<&str>) -> bool {
    matches!(protocol, None | Some("transcription_v1"))
        || protocol.is_some_and(|p| p.eq_ignore_ascii_case("openai_compatible"))
}

/// Whether `address` parses as an HTTP(S) URL targeting loopback on this host.
pub(crate) fn is_loopback_http_address(address: &str) -> bool {
    let Ok(url) = Url::parse(address) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    match url.host() {
        Some(Host::Ipv4(ip)) => ip.is_loopback(),
        Some(Host::Ipv6(ip)) => {
            ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
        Some(Host::Domain(domain)) => domain
            .trim_end_matches('.')
            .eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

/// Shared locality rule used for both readiness (`model_ready` / `model_unloaded`)
/// and execution observation (`local_inference_observation.v1`).
///
/// An explicit operator setting (`plugin.<name>.local_inference_endpoints`)
/// wins for ambiguous proxies or unmanaged local servers. When unset, only
/// plugin-managed loopback endpoints are trusted as executing on this node so
/// remote relay endpoints never skew node-local measurements (such as energy
/// per token).
pub(crate) fn is_local_inference_endpoint(
    address: &str,
    managed_by_plugin: bool,
    operator_local_override: Option<bool>,
) -> bool {
    if let Some(local) = operator_local_override {
        return local;
    }
    managed_by_plugin && is_loopback_http_address(address)
}

/// Collect the ready observable local inference model set from one plugin's
/// endpoint summaries.
pub(crate) fn ready_models_for_endpoints(
    endpoints: &[PluginEndpointSummary],
    operator_local_override: Option<bool>,
) -> BTreeSet<String> {
    let mut ready_models = BTreeSet::new();
    for endpoint in endpoints {
        if endpoint.kind == "inference"
            && endpoint.available
            && endpoint.address.as_deref().is_some_and(|address| {
                is_local_inference_endpoint(
                    address,
                    endpoint.managed_by_plugin,
                    operator_local_override,
                )
            })
            && is_observable_inference_protocol(endpoint.protocol.as_deref())
        {
            for model in &endpoint.models {
                if !model.trim().is_empty() {
                    ready_models.insert(model.clone());
                }
            }
        }
    }
    ready_models
}

/// Map an OpenAI ingress client path + streaming flag to an observed
/// `InferenceOperation`, or `None` when the path executes no model (model
/// listing, token counting, or task contracts without a proto operation yet).
fn executed_operation_for_path(
    path: &str,
    streaming: bool,
) -> Option<(proto::InferenceOperation, bool)> {
    use proto::InferenceOperation as Op;
    let path_only = path.split('?').next().unwrap_or(path);
    if crate::network::openai::workload_routing::is_system_one_path(path_only) {
        return Some((Op::SystemOne, false));
    }
    Some(match path_only {
        "/v1/chat/completions" => (Op::ChatCompletion, streaming),
        "/v1/completions" => (Op::Completion, streaming),
        "/v1/responses" => (Op::Responses, streaming),
        "/v1/messages" => (Op::Messages, streaming),
        "/v1/embeddings" => (Op::Embeddings, false),
        "/v1/rerank" => (Op::Rerank, false),
        "/v1/audio/speech" => (Op::AudioSpeech, false),
        "/v1/audio/transcriptions" => (Op::AudioTranscription, false),
        "/v1/audio/translations" => (Op::AudioTranslation, false),
        _ => return None,
    })
}

#[derive(Default)]
struct TrackerState {
    closed: bool,
    models_by_plugin: BTreeMap<String, BTreeSet<String>>,
}

/// Tracks the ready model set published by each plugin into [`ObservationHub`]
/// and unloads every still-tracked model on `close` or `Drop`.
pub(crate) struct PluginModelReadinessTracker {
    hub: Arc<ObservationHub>,
    state: Mutex<TrackerState>,
}

impl PluginModelReadinessTracker {
    pub(crate) fn new(hub: Arc<ObservationHub>) -> Self {
        Self {
            hub,
            state: Mutex::new(TrackerState::default()),
        }
    }

    /// Reconcile `plugin_name`'s currently ready models with `ObservationHub`.
    /// Steady refreshes emit no events; only added/removed models transition.
    /// Ignored once [`Self::close`] has been called during manager shutdown.
    pub(crate) fn sync_plugin_models(&self, plugin_name: &str, ready_models: BTreeSet<String>) {
        let mut state = self
            .state
            .lock()
            .expect("plugin model readiness tracker poisoned");
        if state.closed {
            return;
        }
        let previous = state
            .models_by_plugin
            .remove(plugin_name)
            .unwrap_or_default();
        for added in ready_models.difference(&previous) {
            self.hub.model_ready(added);
        }
        for removed in previous.difference(&ready_models) {
            self.hub.model_unloaded(removed);
        }
        if !ready_models.is_empty() {
            state
                .models_by_plugin
                .insert(plugin_name.to_string(), ready_models);
        }
    }

    pub(crate) fn clear_plugin(&self, plugin_name: &str) {
        self.sync_plugin_models(plugin_name, BTreeSet::new());
    }

    /// Atomically mark the tracker closed and unload every still-tracked model
    /// so in-flight health probes cannot resurrect `ready_models` after shutdown.
    pub(crate) fn close(&self) {
        let mut state = self
            .state
            .lock()
            .expect("plugin model readiness tracker poisoned");
        state.closed = true;
        for models in state.models_by_plugin.values() {
            for model in models {
                self.hub.model_unloaded(model);
            }
        }
        state.models_by_plugin.clear();
    }
}

impl Drop for PluginModelReadinessTracker {
    fn drop(&mut self) {
        self.close();
    }
}

fn proto_usage_from_token_usage(usage: &TokenUsage) -> Option<proto::InferenceOutputUsage> {
    if usage.prompt_tokens.is_none()
        && usage.completion_tokens.is_none()
        && usage.cached_prompt_tokens.is_none()
    {
        return None;
    }
    let to_u32 = |value: Option<u64>| value.map(|v| u32::try_from(v).unwrap_or(u32::MAX));
    Some(proto::InferenceOutputUsage {
        prompt_tokens: to_u32(usage.prompt_tokens).unwrap_or_default(),
        cached_prompt_tokens: to_u32(usage.cached_prompt_tokens),
        completion_tokens: to_u32(usage.completion_tokens).unwrap_or_default(),
    })
}

/// Deferred handle prepared before connecting to a plugin HTTP endpoint so
/// `Started` is emitted only once forwarding the request succeeds.
pub(crate) struct PendingPluginExecutionObservation {
    hub: Arc<ObservationHub>,
    model: String,
    operation: proto::InferenceOperation,
    streaming: bool,
}

impl PendingPluginExecutionObservation {
    pub(crate) fn prepare_for_request(
        model: Option<&str>,
        request: &BufferedHttpRequest,
        observe_local_inference: bool,
    ) -> Option<Self> {
        if !observe_local_inference || !request.method.eq_ignore_ascii_case("POST") {
            return None;
        }
        let model = model.filter(|m| !m.trim().is_empty())?;
        let (operation, streaming) =
            executed_operation_for_path(&request.client_path, request.is_streaming_request())?;
        Some(Self {
            hub: super::hub(),
            model: model.to_owned(),
            operation,
            streaming,
        })
    }

    pub(crate) fn start(self) -> PluginExecutionObservation {
        let observation_id = generate_request_id();
        self.hub
            .started(observation_id, &self.model, self.operation, self.streaming);
        PluginExecutionObservation {
            hub: self.hub,
            observation_id,
            finished: false,
        }
    }
}

/// RAII guard for one plugin-hosted HTTP inference execution.
///
/// Uses a freshly minted internal `RequestId` for `ObservationHub` keying so
/// concurrent requests sharing a client-supplied `x-request-id` can never
/// collide or terminate each other's execution. Emits `started` on
/// construction, `first_output` on the first streamed output frame, and
/// `usage` / `compute_finished` / `ended` on completion. If dropped before
/// finishing, `Drop` ends the execution as `Cancelled`.
pub(crate) struct PluginExecutionObservation {
    hub: Arc<ObservationHub>,
    observation_id: RequestId,
    finished: bool,
}

impl PluginExecutionObservation {
    #[cfg(test)]
    pub(crate) fn start(
        hub: Arc<ObservationHub>,
        model: Option<&str>,
        client_path: &str,
        streaming: bool,
    ) -> Option<Self> {
        let model = model.filter(|m| !m.trim().is_empty())?;
        let (operation, streaming) = executed_operation_for_path(client_path, streaming)?;
        let observation_id = generate_request_id();
        hub.started(observation_id, model, operation, streaming);
        Some(Self {
            hub,
            observation_id,
            finished: false,
        })
    }

    pub(crate) fn first_output(&self) {
        self.hub.first_output(&self.observation_id);
    }

    pub(crate) fn complete(mut self, usage: Option<&TokenUsage>) {
        self.finished = true;
        if let Some(usage) = usage.and_then(proto_usage_from_token_usage) {
            self.hub.usage(&self.observation_id, usage);
        }
        self.hub.compute_finished(&self.observation_id);
        self.hub.ended(
            &self.observation_id,
            proto::InferenceExecutionOutcome::Completed,
        );
    }

    pub(crate) fn fail(mut self) {
        self.finished = true;
        self.hub.compute_finished(&self.observation_id);
        self.hub.ended(
            &self.observation_id,
            proto::InferenceExecutionOutcome::Failed,
        );
    }

    pub(crate) fn dropped(mut self) {
        self.finished = true;
        self.hub.compute_finished(&self.observation_id);
        self.hub.ended(
            &self.observation_id,
            proto::InferenceExecutionOutcome::Dropped,
        );
    }
}

impl Drop for PluginExecutionObservation {
    fn drop(&mut self) {
        if !self.finished {
            self.hub.compute_finished(&self.observation_id);
            self.hub.ended(
                &self.observation_id,
                proto::InferenceExecutionOutcome::Cancelled,
            );
        }
    }
}

#[cfg(test)]
#[path = "plugin_endpoint_tests.rs"]
mod tests;
