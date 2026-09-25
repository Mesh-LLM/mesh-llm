//! `runtime_state` domain projection.
//!
//! Defect D6 (`.omo/plans/event-system-fixes.md` task 6): the six category
//! arrays used to be structurally present but always empty, because the
//! reducer's `ReducerSnapshot` discarded every domain fact after folding it
//! into generic operation health. Task 6 landed
//! `runtime_events::reducer::DomainState` (bounded per-category state); this
//! module projects it into EXPLICIT `Serialize` structs -- no
//! `serde_json::Value` bags -- so every field this module emits is real,
//! reducer-backed data. `node` was already populated from genuinely
//! available reducer/health data and is unchanged.
//!
//! Inner-key naming here is Rust-side only (task 6). Pinning these keys
//! into `fixtures/runtime_events_v1/frames.json` with sample frames is
//! task 7's job, not this module's.

use serde::Serialize;

use super::node_projection::{NodeCounters, NodeProjection};
#[cfg(test)]
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::reducer::{
    CacheDomainState, DeviceDomainState, DomainState, ModelDomainState, RequestDomainState,
    RequestGenerationState, RequestPrefillState, SessionRecentEntry, StageDomainState,
};

#[derive(Debug, Serialize)]
pub(crate) struct ModelProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) availability: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) load_phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_outcome: Option<String>,
}

impl From<ModelDomainState> for ModelProjection {
    fn from(model: ModelDomainState) -> Self {
        Self {
            id: model.id,
            availability: model.availability,
            load_phase: model.load_phase,
            last_outcome: model.last_outcome,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct StageProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_outcome: Option<String>,
}

impl From<StageDomainState> for StageProjection {
    fn from(stage: StageDomainState) -> Self {
        Self {
            id: stage.id,
            index: stage.index,
            state: stage.state,
            last_outcome: stage.last_outcome,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionRecentProjection {
    pub(crate) id: String,
    pub(crate) state: String,
}

impl From<SessionRecentEntry> for SessionRecentProjection {
    fn from(entry: SessionRecentEntry) -> Self {
        Self {
            id: entry.id,
            state: entry.state,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionsProjection {
    pub(crate) active_count: usize,
    pub(crate) recent: Vec<SessionRecentProjection>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestPrefillProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) phase: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) computed_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_restore: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) outcome: Option<&'static str>,
}

impl From<RequestPrefillState> for RequestPrefillProjection {
    fn from(prefill: RequestPrefillState) -> Self {
        Self {
            phase: prefill.phase,
            cached_tokens: prefill.cached_tokens,
            computed_tokens: prefill.computed_tokens,
            cache_restore: prefill.cache_restore,
            outcome: prefill.outcome,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestGenerationProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) phase: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) generated_tokens: Option<u64>,
    pub(crate) first_token: bool,
    pub(crate) stop_condition_reached: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) outcome: Option<&'static str>,
}

impl From<RequestGenerationState> for RequestGenerationProjection {
    fn from(generation: RequestGenerationState) -> Self {
        Self {
            phase: generation.phase,
            generated_tokens: generation.generated_tokens,
            first_token: generation.first_token,
            stop_condition_reached: generation.stop_condition_reached,
            outcome: generation.outcome,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill: Option<RequestPrefillProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) generation: Option<RequestGenerationProjection>,
}

impl From<RequestDomainState> for RequestProjection {
    fn from(request: RequestDomainState) -> Self {
        Self {
            id: request.id,
            state: request.state,
            prefill: request.prefill.map(RequestPrefillProjection::from),
            generation: request.generation.map(RequestGenerationProjection::from),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DeviceProjection {
    pub(crate) id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) state: Option<String>,
}

impl From<DeviceDomainState> for DeviceProjection {
    fn from(device: DeviceDomainState) -> Self {
        Self {
            id: device.id,
            state: device.state,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct CacheProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pressure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) capacity_state: Option<String>,
}

impl From<CacheDomainState> for CacheProjection {
    fn from(cache: CacheDomainState) -> Self {
        Self {
            pressure: cache.pressure,
            capacity_state: cache.capacity_state,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct StateProjection {
    pub(crate) node: NodeProjection,
    pub(crate) models: Vec<ModelProjection>,
    pub(crate) stages: Vec<StageProjection>,
    pub(crate) sessions: SessionsProjection,
    pub(crate) requests: Vec<RequestProjection>,
    pub(crate) devices: Vec<DeviceProjection>,
    pub(crate) cache: CacheProjection,
}

fn build_from_domain(node: NodeProjection, domain: &DomainState) -> StateProjection {
    StateProjection {
        node,
        models: domain
            .models()
            .into_iter()
            .map(ModelProjection::from)
            .collect(),
        stages: domain
            .stages()
            .into_iter()
            .map(StageProjection::from)
            .collect(),
        sessions: SessionsProjection {
            active_count: domain.sessions_active_count(),
            recent: domain
                .sessions_recent()
                .into_iter()
                .map(SessionRecentProjection::from)
                .collect(),
        },
        requests: domain
            .requests()
            .into_iter()
            .map(RequestProjection::from)
            .collect(),
        devices: domain
            .devices()
            .into_iter()
            .map(DeviceProjection::from)
            .collect(),
        cache: CacheProjection::from(domain.cache()),
    }
}

#[cfg(test)]
pub(crate) fn build(engine: &RuntimeEventEngine) -> StateProjection {
    let snapshot = engine.reducer_snapshot();
    build_from_snapshot(&snapshot)
}

/// Project an already-captured reducer snapshot. Runtime-event stream
/// attachment uses this form so `runtime_state` is built from the same
/// publication boundary as its cursor and initial health frame.
pub(crate) fn build_from_snapshot(
    snapshot: &crate::runtime_events::reducer::ReducerSnapshot,
) -> StateProjection {
    let counters = NodeCounters {
        rebuild_generation: snapshot.rebuild_generation,
        tracked_operation_count: snapshot.operation_count(),
    };
    let node = NodeProjection::build(counters, snapshot.domain());
    build_from_domain(node, snapshot.domain())
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{
        ChildOperationId, DiagnosticEventKind, EventSystemHealthEventKind, FactData, FamilyFact,
        GenerationEventKind, LogicalModelId, ModelAvailabilityEventKind, NativeRuntimeEventKind,
        NodeAvailabilityEventKind, OperationId, OperationScope, PrefillEventKind, RequestEventKind,
        RequestId, RuntimeEventIngress, RuntimeFact, ScopeIdentities,
    };

    use super::*;

    fn model_available_fact(model_id: &str) -> mesh_llm_runtime_event_contracts::RuntimeFact {
        mesh_llm_runtime_event_contracts::RuntimeFact::ModelAvailability(FamilyFact::with_data(
            ModelAvailabilityEventKind::ModelAvailable,
            FactData {
                scope: ScopeIdentities {
                    model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                    ..ScopeIdentities::default()
                },
                ..FactData::default()
            },
        ))
    }

    #[test]
    fn a_model_available_fact_populates_the_models_category() {
        let engine = RuntimeEventEngine::new();
        let reservation = engine
            .reserve_root(mesh_llm_runtime_event_contracts::OperationId::new(), || {
                mesh_llm_runtime_event_contracts::RuntimeFact::NativeRuntime(FamilyFact::new(
                    mesh_llm_runtime_event_contracts::NativeRuntimeEventKind::RuntimeStopped,
                ))
            })
            .expect("reserve");
        reservation
            .ingress()
            .try_submit(model_available_fact("qa-model"));
        engine.drain();

        let projection = build(&engine);
        let model = projection
            .models
            .iter()
            .find(|model| model.id == "qa-model")
            .expect("model must be projected onto state.models");
        assert_eq!(model.availability.as_deref(), Some("available"));

        let value = serde_json::to_value(model).expect("serializable");
        let object = value
            .as_object()
            .expect("model projection is a JSON object");
        let keys: std::collections::BTreeSet<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["id", "availability"].into_iter().collect(),
            "unset load_phase/last_outcome must be omitted, not null"
        );
    }

    #[test]
    fn sessions_projection_has_exactly_active_count_and_recent() {
        let engine = RuntimeEventEngine::new();
        let projection = build(&engine);
        let value = serde_json::to_value(&projection.sessions).expect("serializable");
        let object = value.as_object().expect("sessions is a JSON object");
        let keys: std::collections::BTreeSet<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, ["active_count", "recent"].into_iter().collect());
    }

    fn key_set(value: &serde_json::Value) -> std::collections::BTreeSet<String> {
        value
            .as_object()
            .expect("JSON object")
            .keys()
            .cloned()
            .collect()
    }

    fn reserve_and_submit(
        engine: &std::sync::Arc<RuntimeEventEngine>,
        scope: OperationScope,
        facts: Vec<RuntimeFact>,
    ) {
        let reservation = match scope {
            OperationScope::Root(root) => engine.reserve_root(root, stopped),
            OperationScope::Child { root, child } => engine.reserve_child(root, child, stopped),
        }
        .expect("reserve");
        for fact in facts {
            reservation.ingress().try_submit(fact);
        }
        engine.drain();
    }

    fn stopped() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
    }

    #[test]
    fn a_fresh_node_projection_has_only_its_counters() {
        let engine = RuntimeEventEngine::new();
        let value = serde_json::to_value(&build(&engine).node).expect("serializable");
        assert_eq!(
            key_set(&value),
            ["rebuild_generation", "tracked_operation_count"]
                .map(String::from)
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn node_family_facts_add_their_sub_objects_to_node() {
        let engine = RuntimeEventEngine::new();
        reserve_and_submit(
            &engine,
            OperationScope::root_only(OperationId::new()),
            vec![
                RuntimeFact::NativeRuntime(FamilyFact::new(
                    NativeRuntimeEventKind::RuntimeInitialized,
                )),
                RuntimeFact::NodeAvailability(FamilyFact::new(
                    NodeAvailabilityEventKind::NodeAcceptingRequests,
                )),
                RuntimeFact::Diagnostic(FamilyFact::new(
                    DiagnosticEventKind::DegradedOperationEntered,
                )),
                RuntimeFact::EventSystemHealth(FamilyFact::new(
                    EventSystemHealthEventKind::EventsSampled,
                )),
            ],
        );

        let value = serde_json::to_value(&build(&engine).node).expect("serializable");
        assert_eq!(value["runtime"]["status"], "initialized");
        assert_eq!(value["availability"]["state"], "accepting_requests");
        assert_eq!(value["diagnostics"]["degraded"], true);
        assert_eq!(value["event_system"]["counts_by_kind"]["events_sampled"], 1);
    }

    #[test]
    fn request_rows_carry_prefill_and_generation_from_child_facts() {
        let engine = RuntimeEventEngine::new();
        let root = OperationId::new();
        let request_id = root.to_string();
        reserve_and_submit(
            &engine,
            OperationScope::root_only(root),
            vec![RuntimeFact::Request(FamilyFact::with_data(
                RequestEventKind::RequestReceived,
                FactData {
                    scope: ScopeIdentities {
                        request_id: Some(RequestId::new(&request_id).expect("valid request id")),
                        ..ScopeIdentities::default()
                    },
                    ..FactData::default()
                },
            ))],
        );
        reserve_and_submit(
            &engine,
            OperationScope::with_child(root, ChildOperationId::new()),
            vec![
                RuntimeFact::Prefill(FamilyFact::new(PrefillEventKind::PrefillStarted)),
                RuntimeFact::Generation(FamilyFact::new(GenerationEventKind::FirstTokenProduced)),
            ],
        );

        let projection = build(&engine);
        let request = projection
            .requests
            .iter()
            .find(|request| request.id == request_id)
            .expect("request row");
        let value = serde_json::to_value(request).expect("serializable");
        assert_eq!(value["prefill"]["phase"], "started");
        assert_eq!(value["generation"]["phase"], "streaming");
        assert_eq!(value["generation"]["first_token"], true);
    }

    #[test]
    fn a_request_row_without_execution_facts_omits_prefill_and_generation() {
        let request = RequestProjection::from(RequestDomainState {
            id: "plain".to_string(),
            state: Some("received".to_string()),
            ..RequestDomainState::default()
        });
        let value = serde_json::to_value(&request).expect("serializable");
        assert_eq!(
            key_set(&value),
            ["id", "state"].map(String::from).into_iter().collect()
        );
    }

    #[test]
    fn cache_projection_omits_unset_fields() {
        let engine = RuntimeEventEngine::new();
        let projection = build(&engine);
        let value = serde_json::to_value(&projection.cache).expect("serializable");
        let object = value.as_object().expect("cache is a JSON object");
        assert!(object.is_empty(), "a fresh engine has no cache signal yet");
    }
}
