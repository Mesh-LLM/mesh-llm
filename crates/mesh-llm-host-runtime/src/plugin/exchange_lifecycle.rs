//! Permissioned lifecycle delivery over the authenticated plugin connection.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use futures_util::stream::{FuturesUnordered, StreamExt};
use mesh_llm_config::{OpenAiExchangeFailurePolicy, OpenAiExchangeGrant};
use mesh_llm_plugin::openai_exchange::{OpenAiAdmissionDecision, OpenAiExchangeDecision};
use openai_frontend::wire_bytes::{WireBytesCommitment, WireBytesObserver, commit_wire_bytes};
use serde_json::{Value, json};
use tokio::time::Instant;

use super::{PluginManager, proto};
#[path = "exchange_metadata.rs"]
mod exchange_metadata;
#[path = "exchange_terminal_completion.rs"]
mod exchange_terminal_completion;

#[derive(Default)]
pub(super) struct ExchangeHealth {
    failures: u32,
    in_flight: u32,
    open_until: Option<Instant>,
    pub(super) response_in_flight: u32,
    pub(super) response_failures: u32,
    pub(super) response_open_until: Option<Instant>,
    pub(super) permissions_unavailable: bool,
}

#[derive(Debug, Default)]
pub(crate) struct PhaseResult {
    pub denied: bool,
    pub required_failure: bool,
    pub incomplete: bool,
    pub evidence_unavailable: bool,
    pub internal_hook_failure: bool,
    pub headers: Vec<(String, String)>,
    pub annotations: BTreeMap<String, String>,
}

impl PhaseResult {
    fn merge(&mut self, other: Self) {
        self.denied |= other.denied;
        self.required_failure |= other.required_failure;
        self.incomplete |= other.incomplete;
        self.evidence_unavailable |= other.evidence_unavailable;
        self.internal_hook_failure |= other.internal_hook_failure;
        let headers_complete = exchange_metadata::merge_headers(&mut self.headers, other.headers);
        let annotations_complete =
            exchange_metadata::merge_annotations(&mut self.annotations, other.annotations);
        if !headers_complete || !annotations_complete {
            self.incomplete = true;
            self.evidence_unavailable = true;
        }
    }
    pub(crate) fn error_status(&self) -> Option<u16> {
        if self.denied {
            Some(403)
        } else if self.required_failure {
            Some(503)
        } else {
            None
        }
    }
}

fn project_event(mut event: Value, grant: &OpenAiExchangeGrant, name: &str) -> Value {
    if event["phase"] == "exchange_finished" {
        let complete = !grant.response_body || event["observer_response_delivery"][name] == true;
        if let Some(commitment) = event["response_wire_commitment"].as_object_mut() {
            commitment.insert("side_stream_complete".into(), json!(complete));
        }
        event["observer_evidence_complete"] = json!(
            complete
                && event["response_wire_commitment"]["incomplete"].is_null()
                && !event["response_wire_commitment"].is_null()
        );
    }
    if let Some(fields) = event.as_object_mut() {
        fields.remove("observer_response_delivery");
    }
    if let Some(annotations) = event["annotations"].as_object_mut() {
        let prefix = annotation_prefix(name);
        annotations.retain(|key, _| key.starts_with(&prefix));
    }
    let request_phase = event["phase"] == "request_received";
    let body_permitted = if request_phase {
        grant.request_body
    } else {
        grant.effective_request_body
    };
    let Some(fields) = event.as_object_mut() else {
        return Value::Null;
    };
    // Small parsed bodies can accompany the event; the exact entity always uses
    // a negotiated side stream. Large prompts never use service envelopes.
    if !body_permitted
        || fields
            .get("body")
            .is_some_and(|body| body.to_string().len() > 64 * 1024)
    {
        fields.remove("body");
    }
    if !body_permitted {
        fields.remove("body_hex");
    }
    if let Some(headers) = event["headers"].as_object_mut() {
        headers.retain(|name, _| {
            grant.headers.iter().any(|h| h.eq_ignore_ascii_case(name))
                && mesh_llm_config::safe_exchange_header(name)
        });
    }
    event
}

impl PluginManager {
    pub(crate) fn record_typed_execution_outcome(&self, backend_exchange_id: &str, outcome: &str) {
        let state = self
            .inner
            .exchange_observations
            .1
            .lock()
            .unwrap()
            .get(backend_exchange_id)
            .and_then(Weak::upgrade);
        if let Some(state) = state {
            state.lock().unwrap().execution_outcome = Some(outcome.to_owned());
        }
    }
    pub(crate) fn record_typed_usage(&self, backend_exchange_id: &str, usage: Value) {
        let state = self
            .inner
            .exchange_observations
            .1
            .lock()
            .unwrap()
            .get(backend_exchange_id)
            .and_then(Weak::upgrade);
        if let Some(state) = state {
            state.lock().unwrap().usage = Some(usage);
        }
    }
    /// Operator-visible callback health; never includes request identifiers.
    pub(crate) fn exchange_health_status(&self, name: &str) -> &'static str {
        if self.effective_exchange_grant(name).is_none() {
            return "disabled";
        }
        let health = self.inner.exchange_health.lock().unwrap();
        match health.get(name) {
            Some(state) if state.permissions_unavailable => "permissions_unavailable",
            Some(state)
                if state.open_until.is_some_and(|until| until > Instant::now())
                    || state
                        .response_open_until
                        .is_some_and(|until| until > Instant::now()) =>
            {
                "circuit_open"
            }
            Some(state) if state.failures > 0 || state.response_failures > 0 => "degraded",
            _ => "healthy",
        }
    }

    pub(crate) async fn selected_exchange_phase(
        &self,
        observation_id: &str,
        mut event: Value,
    ) -> PhaseResult {
        let state = self
            .inner
            .exchange_observations
            .0
            .lock()
            .unwrap()
            .get(observation_id)
            .and_then(Weak::upgrade);
        let Some(state) = state else {
            return PhaseResult {
                required_failure: true,
                incomplete: true,
                ..Default::default()
            };
        };
        {
            let state = state.lock().unwrap();
            event["exchange_id"] = json!(state.exchange_id);
            event["endpoint"] = json!(state.endpoint);
        }
        event["observation_id"] = json!(observation_id);
        if let Some(id) = event["backend_exchange_id"].as_str() {
            let mut emission = state.lock().unwrap();
            if !emission
                .backend_exchange_ids
                .iter()
                .any(|known| known == id)
                && emission.backend_exchange_ids.len() < 16
            {
                self.inner
                    .exchange_observations
                    .1
                    .lock()
                    .unwrap()
                    .insert(id.into(), Arc::downgrade(&state));
                emission.backend_exchange_ids.push(id.into());
            }
        }
        let mut result = self
            .exchange_permissions_preflight(event["endpoint"].as_str().unwrap_or_default())
            .await;
        result.merge(self.exchange_phase(event.clone()).await);
        let mut state = state.lock().unwrap();
        state.incomplete |= result.incomplete;
        state.required_failure |= result.required_failure;
        state.denied |= result.denied;
        state.evidence_unavailable |= result.evidence_unavailable;
        state.internal_hook_failure |= result.internal_hook_failure;
        let headers_complete =
            exchange_metadata::merge_headers(&mut state.headers, result.headers.clone());
        let annotations_complete = exchange_metadata::merge_annotations(
            &mut state.annotations,
            result.annotations.clone(),
        );
        if !headers_complete || !annotations_complete {
            state.incomplete = true;
            state.evidence_unavailable = true;
        }
        if state.attempt_history.len() < 16 {
            let mut attempt = serde_json::Map::new();
            for key in [
                "model",
                "provider",
                "target",
                "attempt",
                "effective_request_wire_digest",
                "effective_request_encoding",
            ] {
                if let Some(value) = event.get(key) {
                    attempt.insert(key.into(), value.clone());
                }
            }
            state.attempt_history.push(Value::Object(attempt));
        } else {
            state.incomplete = true;
            state.evidence_unavailable = true;
        }
        state.selected = Some(event);
        result
    }
    pub(crate) async fn plugin_exchange_callback_active(&self, name: &str) -> bool {
        self.inner
            .exchange_health
            .lock()
            .unwrap()
            .get(name)
            .is_some_and(|s| s.in_flight > 0)
    }
    #[cfg(test)]
    pub(crate) fn set_test_exchange_callback_active(&self, name: &str, active: bool) {
        self.inner
            .exchange_health
            .lock()
            .unwrap()
            .entry(name.into())
            .or_default()
            .in_flight = u32::from(active);
    }
    pub(crate) async fn has_exchange_hooks(&self) -> bool {
        self.exchange_grant_snapshot()
            .values()
            .any(|grant| !grant.endpoints.is_empty() && !grant.phases.is_empty())
    }
    pub(crate) async fn exchange_phase(&self, event: Value) -> PhaseResult {
        let mut result = self
            .exchange_permissions_preflight(event["endpoint"].as_str().unwrap_or_default())
            .await;
        result.merge(self.exchange_phase_before(event, None).await);
        result
    }
    async fn exchange_phase_before(
        &self,
        event: Value,
        outer_deadline: Option<Instant>,
    ) -> PhaseResult {
        let endpoint = event["endpoint"].as_str().unwrap_or_default();
        let phase = event["phase"].as_str().unwrap_or_default();
        let mut subscriptions = Vec::new();
        let mut result = PhaseResult::default();
        for (name, plugin) in &self.inner.plugins {
            let Some(manifest) = plugin.manifest_snapshot().await else {
                continue;
            };
            let revision = self.exchange_grant_revision(name);
            let negotiation = self.granted_exchange_subscription(
                name,
                manifest.openai_exchange_hook.as_deref(),
                endpoint,
                phase,
            );
            let grant = match negotiation {
                Ok(Some(grant)) => grant,
                Ok(None) => continue,
                Err(_) => {
                    result.merge(PhaseResult {
                        required_failure: true,
                        incomplete: true,
                        evidence_unavailable: true,
                        ..Default::default()
                    });
                    continue;
                }
            };
            if grant.endpoints.iter().any(|v| v == endpoint)
                && grant.phases.iter().any(|v| v == phase)
            {
                let handler = manifest.openai_exchange_hook.unwrap().handler;
                subscriptions.push((name, plugin, handler, grant, revision));
            }
        }
        // All policies share one absolute host deadline. No serial timeout multiplication.
        let mut deadline = Instant::now()
            + Duration::from_millis(
                subscriptions
                    .iter()
                    .map(|(_, _, _, g, _)| g.deadline_ms)
                    .min()
                    .unwrap_or(1),
            );
        if let Some(outer) = outer_deadline {
            deadline = deadline.min(outer);
        }
        let mut tasks = FuturesUnordered::new();
        for (name, plugin, handler, grant, mut revision) in subscriptions {
            let projected = project_event(event.clone(), &grant, name);
            tasks.push(async move {
                let unavailable = {
                    let mut health = self.inner.exchange_health.lock().unwrap();
                    let state = health.entry(name.clone()).or_default();
                    let unavailable = state.open_until.is_some_and(|until| until > Instant::now())
                        || state.in_flight >= grant.max_in_flight;
                    if !unavailable { state.in_flight += 1; }
                    unavailable
                };
                if unavailable { return failed_policy(&grant); }
                let permit = ExchangePermit::new(self.inner.exchange_health.clone(), name.clone());
                let decision = tokio::select! {
                    biased;
                    _ = revision.changed() => Err(HookFailure::EvidenceUnavailable),
                    decision = invoke_phase(self, plugin, name, &handler, projected, &grant, (phase, deadline)) => decision,
                };
                permit.complete(decision.is_ok());
                decision.map_or_else(|failure| failed_policy_reason(&grant,failure), |decision| phase_result(name, decision))
            });
        }
        while let Some(policy) = tasks.next().await {
            result.merge(policy);
        }
        result
    }
}

fn failed_policy(grant: &OpenAiExchangeGrant) -> PhaseResult {
    failed_policy_reason(grant, HookFailure::Internal)
}
fn failed_policy_reason(grant: &OpenAiExchangeGrant, failure: HookFailure) -> PhaseResult {
    PhaseResult {
        required_failure: grant.failure_policy == OpenAiExchangeFailurePolicy::Required,
        incomplete: true,
        evidence_unavailable: failure == HookFailure::EvidenceUnavailable,
        internal_hook_failure: failure == HookFailure::Internal,
        ..Default::default()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HookFailure {
    EvidenceUnavailable,
    Internal,
}

pub(super) type HealthStates = Arc<Mutex<BTreeMap<String, ExchangeHealth>>>;

/// Releases the capacity slot on every return, panic, deadline, or cancellation.
struct ExchangePermit {
    health: HealthStates,
    name: String,
    completed: bool,
}

impl ExchangePermit {
    fn new(health: HealthStates, name: String) -> Self {
        Self {
            health,
            name,
            completed: false,
        }
    }
    fn complete(mut self, success: bool) {
        self.record(success);
        self.completed = true;
    }
    fn record(&self, success: bool) {
        let mut health = self.health.lock().unwrap();
        let state = health.entry(self.name.clone()).or_default();
        state.in_flight = state.in_flight.saturating_sub(1);
        if success {
            state.failures = 0;
            state.open_until = None;
        } else {
            state.failures = state.failures.saturating_add(1);
            if state.failures >= 3 {
                state.open_until = Some(Instant::now() + Duration::from_secs(30));
            }
            tracing::warn!(plugin = %self.name, failures = state.failures, "OpenAI observer degraded");
        }
    }
}
impl Drop for ExchangePermit {
    fn drop(&mut self) {
        if !self.completed {
            self.record(false);
        }
    }
}

async fn invoke_phase(
    manager: &PluginManager,
    plugin: &super::ExternalPlugin,
    name: &str,
    handler: &str,
    mut event: Value,
    grant: &OpenAiExchangeGrant,
    phase_deadline: (&str, Instant),
) -> Result<OpenAiExchangeDecision, HookFailure> {
    let (phase, deadline) = phase_deadline;
    if let Some(encoded) = event["body_hex"].as_str() {
        let bytes = hex::decode(encoded).map_err(|_| HookFailure::Internal)?;
        if bytes.len() as u64 > grant.max_body_bytes {
            return Err(HookFailure::EvidenceUnavailable);
        }
        let kind = if phase == "request_received" {
            "openai_exchange_original"
        } else {
            "openai_exchange_effective"
        };
        super::exchange_streams::send_request_body(
            manager,
            name,
            event["exchange_id"].as_str().ok_or(HookFailure::Internal)?,
            kind,
            &bytes,
            deadline,
        )
        .await
        .map_err(|_| HookFailure::EvidenceUnavailable)?;
        event
            .as_object_mut()
            .ok_or(HookFailure::Internal)?
            .remove("body_hex");
        event["body_stream_kind"] = json!(kind);
    }
    let input = serde_json::to_string(&event).map_err(|_| HookFailure::Internal)?;
    let reply = tokio::time::timeout_at(
        deadline,
        plugin.invoke_service(
            proto::ServiceKind::OpenaiExchange,
            handler,
            &input,
            Some(deadline.saturating_duration_since(Instant::now())),
        ),
    )
    .await
    .map_err(|_| HookFailure::Internal)?
    .map_err(|_| HookFailure::Internal)?;
    if reply.is_error || reply.output_json.len() > 32 * 1024 {
        return Err(HookFailure::Internal);
    }
    let decision: OpenAiExchangeDecision =
        serde_json::from_str(&reply.output_json).map_err(|_| HookFailure::Internal)?;
    if valid_decision(&decision, phase, name, grant) {
        Ok(decision)
    } else {
        Err(HookFailure::Internal)
    }
}

fn valid_decision(
    decision: &OpenAiExchangeDecision,
    phase: &str,
    name: &str,
    grant: &OpenAiExchangeGrant,
) -> bool {
    if decision
        .reason
        .as_ref()
        .is_some_and(|reason| reason.len() > 1024)
        || (decision.decision == OpenAiAdmissionDecision::Deny
            && (!grant.admission || !matches!(phase, "request_received" | "backend_selected")))
        || ((!decision.annotations.is_empty() || !decision.response_headers.is_empty())
            && !grant.metadata)
        || decision.response_headers.len() > 16
    {
        return false;
    }
    let prefix = exchange_metadata::response_header_prefix(name);
    let valid_headers = decision.response_headers.iter().all(|(name, value)| {
        name.len() <= 128
            && value.len() <= 1024
            && name.to_ascii_lowercase().starts_with(&prefix)
            && mesh_llm_config::safe_exchange_header(name)
            && grant
                .headers
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(name))
            && value
                .bytes()
                .all(|byte| byte == b'\t' || (32..127).contains(&byte))
    });
    let mut annotation_bytes = 0usize;
    let valid_annotations = decision.annotations.iter().all(|(key, value)| {
        annotation_bytes =
            annotation_bytes.saturating_add(name.len() + 1 + key.len() + value.len());
        !key.is_empty()
            && key.len() <= 128
            && value.len() <= 1024
            && key
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
            && annotation_bytes <= 4096
    });
    valid_headers && valid_annotations
}

fn phase_result(name: &str, decision: OpenAiExchangeDecision) -> PhaseResult {
    let prefix = annotation_prefix(name);
    PhaseResult {
        denied: decision.decision == OpenAiAdmissionDecision::Deny,
        headers: decision.response_headers,
        annotations: decision
            .annotations
            .into_iter()
            .map(|(key, value)| (format!("{prefix}{key}"), value))
            .collect(),
        ..Default::default()
    }
}

fn annotation_prefix(name: &str) -> String {
    // Plugin names may contain dots, whereas the dot separates author and key.
    format!("{}.", name.replace('%', "%25").replace('.', "%2e"))
}

#[derive(Default)]
struct EmissionState {
    usage: Option<Value>,
    attempt_history: Vec<Value>,
    backend_exchange_ids: Vec<String>,
    execution_outcome: Option<String>,
    status: Option<u16>,
    commitment: Option<WireBytesCommitment>,
    copies: super::exchange_streams::ResponseCopies,
    exchange_id: String,
    endpoint: String,
    headers: Vec<(String, String)>,
    annotations: BTreeMap<String, String>,
    incomplete: bool,
    required_failure: bool,
    denied: bool,
    selected: Option<Value>,
    evidence_unavailable: bool,
    internal_hook_failure: bool,
}

#[derive(Default)]
pub(super) struct ObservationRegistry(
    Mutex<BTreeMap<String, Weak<Mutex<EmissionState>>>>,
    Mutex<BTreeMap<String, Weak<Mutex<EmissionState>>>>,
);

struct EmissionObserver(Arc<Mutex<EmissionState>>);
impl WireBytesObserver for EmissionObserver {
    fn execution_outcome(&self, outcome: &str) {
        self.0.lock().unwrap().execution_outcome = Some(outcome.to_owned());
    }
    fn response_headers(&self) -> Vec<(String, String)> {
        self.0.lock().unwrap().headers.clone()
    }
    fn response_status(&self, status: u16) {
        self.0.lock().unwrap().status = Some(status);
    }
    fn try_chunk(&self, offset: u64, bytes: &[u8]) -> bool {
        self.0.lock().unwrap().copies.enqueue(offset, bytes)
    }
    fn finish(&self, commitment: WireBytesCommitment) {
        self.0.lock().unwrap().commitment = Some(commitment);
    }
}

/// One terminal owner per accepted external HTTP exchange.
pub(crate) struct ExchangeSession {
    started: Instant,
    manager: PluginManager,
    event: Value,
    emission: Arc<Mutex<EmissionState>>,
    incomplete: bool,
    finished: bool,
    response_delivery: BTreeMap<String, bool>,
}

impl ExchangeSession {
    pub(crate) fn outcome_from_status(&self) -> &'static str {
        match self.emission.lock().unwrap().status {
            Some(200..=299) => "completed",
            Some(400..=499) => "request_invalid",
            Some(504) => "timed_out",
            _ => "backend_error",
        }
    }
    pub(crate) async fn begin(manager: &PluginManager, event: Value) -> (Self, PhaseResult) {
        let started = Instant::now();
        let deadline = Instant::now()
            + Duration::from_millis(
                manager
                    .inner
                    .plugins
                    .keys()
                    .filter_map(|name| manager.effective_exchange_grant(name))
                    .filter(|grant| {
                        grant
                            .endpoints
                            .iter()
                            .any(|endpoint| event["endpoint"] == endpoint.as_str())
                    })
                    .map(|grant| grant.deadline_ms)
                    .min()
                    .unwrap_or(1),
            );
        let mut result = manager
            .exchange_permissions_preflight(event["endpoint"].as_str().unwrap_or_default())
            .await;
        result.merge(
            manager
                .exchange_phase_before(event.clone(), Some(deadline))
                .await,
        );
        let copies = super::exchange_streams::ResponseCopies::open(manager, &event, deadline).await;
        let (unavailable, required) = copies.admission_failure();
        result.incomplete |= unavailable;
        result.evidence_unavailable |= unavailable;
        result.required_failure |= required;
        let emission = Arc::new(Mutex::new(EmissionState {
            copies,
            exchange_id: event["exchange_id"].as_str().unwrap_or_default().into(),
            endpoint: event["endpoint"].as_str().unwrap_or_default().into(),
            headers: result.headers.clone(),
            annotations: result.annotations.clone(),
            incomplete: result.incomplete,
            required_failure: result.required_failure,
            denied: result.denied,
            evidence_unavailable: result.evidence_unavailable,
            internal_hook_failure: result.internal_hook_failure,
            ..Default::default()
        }));
        manager
            .inner
            .exchange_observations
            .0
            .lock()
            .unwrap()
            .insert(
                event["observation_id"].as_str().unwrap().into(),
                Arc::downgrade(&emission),
            );
        (
            Self {
                started,
                manager: manager.clone(),
                event,
                emission,
                incomplete: result.incomplete,
                finished: false,
                response_delivery: BTreeMap::new(),
            },
            result,
        )
    }
    pub(crate) fn observer(&self) -> Arc<dyn WireBytesObserver> {
        Arc::new(EmissionObserver(self.emission.clone()))
    }
    pub(crate) fn observation_id(&self) -> &str {
        self.event["observation_id"].as_str().unwrap()
    }
    pub(crate) fn record_usage(&self, usage: Value) {
        self.emission.lock().unwrap().usage = Some(usage);
    }
    pub(crate) async fn finish(&mut self, outcome: &str) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.remove_backend_observations();
        self.manager
            .inner
            .exchange_observations
            .0
            .lock()
            .unwrap()
            .remove(self.observation_id());
        let mut owner = self.terminal_owner();
        let outcome = outcome.to_owned();
        exchange_terminal_completion::complete(async move {
            let terminal = owner.final_terminal_event(&outcome).await;
            owner.manager.exchange_phase(terminal).await;
        })
        .await;
    }
    fn terminal_owner(&self) -> Self {
        Self {
            started: self.started,
            manager: self.manager.clone(),
            event: self.event.clone(),
            emission: self.emission.clone(),
            incomplete: self.incomplete,
            finished: true,
            response_delivery: BTreeMap::new(),
        }
    }
    async fn final_terminal_event(&mut self, outcome: &str) -> Value {
        let mut copies = std::mem::take(&mut self.emission.lock().unwrap().copies);
        self.response_delivery = copies.close_with_receipts().await;
        self.terminal_event(outcome)
    }
    fn terminal_event(&self, outcome: &str) -> Value {
        let emission = self.emission.lock().unwrap();
        let mut event = self.event.clone();
        event.as_object_mut().unwrap().remove("body");
        event.as_object_mut().unwrap().remove("body_hex");
        event["phase"] = json!("exchange_finished");
        event["ingress_observation_point"] = event["observation_point"].clone();
        event["observation_point"] = json!("client_egress");
        event["execution_outcome"] = json!(if emission.denied {
            "policy_denied"
        } else if emission.required_failure {
            "internal_hook_failure"
        } else {
            emission.execution_outcome.as_deref().unwrap_or(outcome)
        });
        event["status"] = json!(emission.status);
        event["usage"] = json!(emission.usage);
        event["elapsed_ms"] = json!(self.started.elapsed().as_millis());
        event["attempt_history"] = json!(emission.attempt_history);
        event["any_response_bytes_emitted"] = json!(
            emission
                .commitment
                .as_ref()
                .is_some_and(|commitment| commitment.byte_count > 0)
        );
        event["response_wire_commitment"] = json!(emission.commitment);
        if let Some(selected) = &emission.selected {
            for key in [
                "effective_request_wire_digest",
                "effective_request_encoding",
                "model",
                "provider",
                "target",
                "attempt",
            ] {
                if let Some(value) = selected.get(key) {
                    event[key] = value.clone();
                }
            }
        }
        event["annotations"] = json!(emission.annotations);
        event["evidence_unavailable"] = json!(
            emission.evidence_unavailable
                || self.response_delivery.values().any(|complete| !*complete)
        );
        event["internal_hook_failure"] = json!(emission.internal_hook_failure);
        event["observer_response_delivery"] = json!(self.response_delivery);
        event["evidence_complete"] = json!(
            !self.incomplete
                && !emission.incomplete
                && self.response_delivery.values().all(|complete| *complete)
                && emission
                    .commitment
                    .as_ref()
                    .is_some_and(|c| c.incomplete.is_none() && c.side_stream_complete)
        );
        event
    }
    fn remove_backend_observations(&self) {
        let emission = self.emission.lock().unwrap();
        let mut registry = self.manager.inner.exchange_observations.1.lock().unwrap();
        for id in &emission.backend_exchange_ids {
            registry.remove(id);
        }
    }
}

impl Drop for ExchangeSession {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.remove_backend_observations();
        self.manager
            .inner
            .exchange_observations
            .0
            .lock()
            .unwrap()
            .remove(self.observation_id());
        let mut owner = self.terminal_owner();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let terminal = owner.final_terminal_event("client_cancelled").await;
                owner.manager.exchange_phase(terminal).await;
            });
        }
    }
}

#[cfg(test)]
#[path = "exchange_terminal_tests.rs"]
mod terminal_tests;

pub(crate) fn request_event(
    id: String,
    endpoint: &str,
    method: &str,
    path: &str,
    body: &[u8],
    headers: BTreeMap<String, String>,
    serving_ingress: bool,
) -> Value {
    let parsed = serde_json::from_slice::<Value>(body).ok();
    let content_type = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.as_str())
        .unwrap_or("application/json");
    json!({"exchange_id":id,"request_id":id,"observation_id":uuid::Uuid::new_v4().to_string(),"phase":"request_received","endpoint":endpoint,
        "api_version":"v1","parse_status":if parsed.is_some() { "valid_json" } else { "invalid_json" },
        "content_type":content_type,"origin_class":if serving_ingress { "tunneled_ingress" } else { "local_ingress" },
        "recorder_role":"host_api_boundary","evidence_status":"api_boundary","assurance_rung":"api_boundary",
        "method":method,"path":path.split('?').next().unwrap_or(path),"model":parsed.as_ref().and_then(|body|body.get("model")).and_then(Value::as_str),"headers":headers,"body_hex":hex::encode(body),
        "body":parsed,"request_wire_digest":commit_wire_bytes(body),
        "observation_point":if serving_ingress {"serving_host_ingress"} else {"gateway_ingress"}})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_is_released_on_cancel_and_circuit_opens_after_three_failures() {
        let health: HealthStates = Arc::default();
        for _ in 0..3 {
            health
                .lock()
                .unwrap()
                .entry("observer".into())
                .or_default()
                .in_flight += 1;
            drop(ExchangePermit::new(health.clone(), "observer".into()));
            assert_eq!(health.lock().unwrap()["observer"].in_flight, 0);
        }
        assert_eq!(health.lock().unwrap()["observer"].failures, 3);
        assert!(health.lock().unwrap()["observer"].open_until.unwrap() > Instant::now());
        health
            .lock()
            .unwrap()
            .get_mut("observer")
            .unwrap()
            .in_flight = 1;
        ExchangePermit::new(health.clone(), "observer".into()).complete(true);
        let health = health.lock().unwrap();
        assert_eq!(health["observer"].in_flight, 0);
        assert_eq!(health["observer"].failures, 0);
        assert!(health["observer"].open_until.is_none());
    }

    #[test]
    fn selected_admission_can_deny_but_terminal_cannot() {
        let grant = OpenAiExchangeGrant {
            admission: true,
            ..Default::default()
        };
        let decision = OpenAiExchangeDecision {
            decision: OpenAiAdmissionDecision::Deny,
            ..Default::default()
        };
        assert!(valid_decision(
            &decision,
            "request_received",
            "observer",
            &grant
        ));
        assert!(valid_decision(
            &decision,
            "backend_selected",
            "observer",
            &grant
        ));
        assert!(!valid_decision(
            &decision,
            "exchange_finished",
            "observer",
            &grant
        ));
    }

    #[test]
    fn terminal_delivery_receipts_are_projected_per_recipient() {
        let event = json!({"phase":"exchange_finished", "model":"tiny", "response_wire_commitment":{"incomplete":null,"side_stream_complete":false}, "observer_response_delivery":{"healthy":true,"failed":false}});
        let grant = OpenAiExchangeGrant {
            response_body: true,
            ..Default::default()
        };
        let healthy = project_event(event.clone(), &grant, "healthy");
        let failed = project_event(event, &grant, "failed");
        assert_eq!(
            healthy["response_wire_commitment"]["side_stream_complete"],
            true
        );
        assert_eq!(healthy["observer_evidence_complete"], true);
        assert_eq!(failed["observer_evidence_complete"], false);
        assert_eq!(healthy["model"], "tiny");
        assert!(healthy.get("observer_response_delivery").is_none());
    }
    #[test]
    fn annotations_cannot_relay_another_plugins_body_observations() {
        let event = json!({"phase":"exchange_finished","annotations":{"body-reader.secret":"prompt","observer.receipt":"safe"}});
        let projected = project_event(event, &OpenAiExchangeGrant::default(), "observer");
        assert_eq!(projected["annotations"], json!({"observer.receipt":"safe"}));
        let decision = OpenAiExchangeDecision {
            annotations: vec![("secret".into(), "prompt".into())],
            ..Default::default()
        };
        let authored = phase_result("observer.child", decision);
        let event = json!({"phase":"exchange_finished","annotations":authored.annotations});
        assert_eq!(
            project_event(event.clone(), &OpenAiExchangeGrant::default(), "observer")["annotations"],
            json!({})
        );
        assert_eq!(
            project_event(event, &OpenAiExchangeGrant::default(), "observer.child")["annotations"]
                ["observer%2echild.secret"],
            "prompt"
        );
    }
    #[test]
    fn deny_wins_over_allow_and_required_failure() {
        let mut result = PhaseResult {
            denied: true,
            ..Default::default()
        };
        result.merge(PhaseResult {
            required_failure: true,
            incomplete: true,
            ..Default::default()
        });
        result.merge(PhaseResult::default());
        assert_eq!(result.error_status(), Some(403));
    }
    #[test]
    fn ungranted_body_and_sensitive_headers_are_removed() {
        let event = json!({"phase":"request_received","body":{"secret":"prompt"},"body_hex":"aa",
            "headers":{"authorization":"secret","x-capsule-client-nonce":"public"}});
        let grant = OpenAiExchangeGrant {
            headers: vec!["authorization".into(), "x-capsule-client-nonce".into()],
            ..Default::default()
        };
        let event = project_event(event, &grant, "observer");
        assert!(event.get("body").is_none());
        assert!(event.get("body_hex").is_none());
        assert!(event["headers"].get("authorization").is_none());
        assert_eq!(event["headers"]["x-capsule-client-nonce"], "public");
    }
}
