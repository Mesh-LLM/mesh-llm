use super::{
    ManagedModelController, ModelTargetReconciliationAction, ModelTargetReconciliationCandidate,
    ModelTargetReconciliationCapacityState, ModelTargetReconciliationInput,
    ModelTargetReconciliationPolicy, ModelTargetReconciliationState, RuntimeEvent,
    RuntimeModelHandleEntry, plan_model_target_reconciliation,
};
use crate::api;
use crate::mesh;
use crate::models;
use crate::plugin;
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_llm_node::serving::{UnloadOptions, UnloadTarget};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn model_target_reconciliation_policy(
    config: &plugin::MeshConfig,
) -> ModelTargetReconciliationPolicy {
    ModelTargetReconciliationPolicy {
        enabled: config.runtime.reconcile_model_targets,
        demand_upgrades_enabled: config.runtime.reconcile_model_target_demand_upgrades,
        demand_upgrade_min_request_count: config.runtime.model_target_demand_upgrade_min_requests,
        demand_upgrade_max_age_secs: config.runtime.model_target_demand_upgrade_max_age_secs,
        ..ModelTargetReconciliationPolicy::default()
    }
}

pub(crate) struct ReconcileModelTargetsContext<'a> {
    pub(super) policy: &'a ModelTargetReconciliationPolicy,
    pub(super) state: &'a mut ModelTargetReconciliationState,
    pub(super) node: &'a mesh::Node,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) runtime_models: &'a HashMap<String, RuntimeModelHandleEntry>,
    pub(super) managed_models: &'a HashMap<String, ManagedModelController>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
}

pub(crate) async fn reconcile_model_targets_once(ctx: ReconcileModelTargetsContext<'_>) {
    let ReconcileModelTargetsContext {
        policy,
        state,
        node,
        console_state,
        runtime_models,
        managed_models,
        control_tx,
        runtime_event_tx,
    } = ctx;
    if !policy.enabled {
        return;
    }
    let Some(console_state) = console_state else {
        return;
    };
    let local_interest_model_refs = node
        .explicit_model_interests()
        .await
        .into_iter()
        .collect::<BTreeSet<_>>();
    let loaded_model_refs = runtime_loaded_model_refs(runtime_models, managed_models);
    if local_interest_model_refs.is_empty() && loaded_model_refs.is_empty() {
        state.prune_expired(runtime_unix_secs());
        return;
    }

    let target_lookup = console_state.model_target_lookup().await;
    let local_vram_bytes = node.local_runtime_capacity_bytes();
    let targets = target_lookup
        .targets
        .into_iter()
        .map(|target| {
            let demand_upgrade_target = model_target_reconciliation_demand_upgrade_candidate(
                policy,
                &loaded_model_refs,
                &target,
            );
            let local_path = if target.wanted
                && target.serving_node_count == 0
                && (local_interest_model_refs.contains(&target.model_ref) || demand_upgrade_target)
                && target.capacity_advice.state
                    == api::status::ModelTargetCapacityAdviceState::SingleNodeFit
                && model_target_reconciliation_local_fit(&target, local_vram_bytes)
            {
                local_model_path_for_reconciliation_target(&target)
            } else {
                None
            };
            ModelTargetReconciliationCandidate {
                rank: target.rank,
                model_ref: target.model_ref,
                profile: target.profile,
                model_name: target.model_name,
                wanted: target.wanted,
                wanted_reason: target.wanted_reason,
                request_count: target.request_count,
                last_active_secs_ago: target.last_active_secs_ago,
                serving_node_count: target.serving_node_count,
                capacity_state: ModelTargetReconciliationCapacityState::from(
                    target.capacity_advice.state,
                ),
                local_path,
            }
        })
        .collect::<Vec<_>>();

    let now_secs = runtime_unix_secs();
    let actions = plan_model_target_reconciliation(
        policy,
        state,
        ModelTargetReconciliationInput {
            now_secs,
            local_role: node.role().await,
            local_interest_model_refs: &local_interest_model_refs,
            loaded_model_refs: &loaded_model_refs,
            targets: &targets,
        },
    );

    for action in actions {
        let load_spec = action.load_spec.to_string_lossy().to_string();
        let profile = action.profile.clone();
        state.mark_load_started(&action.model_ref, &profile);
        let event_tx = runtime_event_tx.clone();
        let model_ref = action.model_ref.clone();
        let control_tx = control_tx.clone();
        let replace_model_ref = action.replace_model_ref.clone();
        let event_profile = action.profile.clone();
        tokio::spawn(async move {
            let result = run_model_target_reconciliation_action(
                control_tx,
                model_ref.clone(),
                load_spec,
                replace_model_ref,
                profile,
            )
            .await;
            let _ = event_tx.send(RuntimeEvent::ModelTargetReconciliationLoadFinished {
                model_ref,
                profile: event_profile,
                result,
            });
        });
        emit_model_target_reconciliation_queued(&action);
    }
}

pub(crate) async fn run_model_target_reconciliation_action(
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    config_model_id: String,
    load_spec: String,
    replace_model_ref: Option<String>,
    profile: String,
) -> std::result::Result<api::RuntimeLoadResponse, String> {
    if let Some(replace_model_ref) = replace_model_ref {
        run_model_target_reconciliation_unload(control_tx.clone(), replace_model_ref).await?;
    }
    run_model_target_reconciliation_load(control_tx, config_model_id, load_spec, profile).await
}

pub(crate) async fn run_model_target_reconciliation_unload(
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    model_ref: String,
) -> std::result::Result<api::RuntimeUnloadResponse, String> {
    let (resp, response) = tokio::sync::oneshot::channel();
    control_tx
        .send(api::RuntimeControlRequest::Unload {
            target: UnloadTarget::Model(model_ref.clone()),
            options: UnloadOptions::default(),
            resp,
        })
        .map_err(|_| format!("runtime unload queue closed for replacement target '{model_ref}'"))?;
    response
        .await
        .map_err(|err| format!("runtime unload response channel closed: {err}"))?
        .map_err(|err| err.to_string())
}

pub(crate) async fn run_model_target_reconciliation_load(
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    config_model_id: String,
    load_spec: String,
    profile: String,
) -> std::result::Result<api::RuntimeLoadResponse, String> {
    let (resp, response) = tokio::sync::oneshot::channel();
    control_tx
        .send(api::RuntimeControlRequest::Load {
            spec: load_spec.clone(),
            config_model_id: Some(config_model_id),
            profile: profile.clone(),
            resp,
        })
        .map_err(|_| format!("runtime load queue closed for '{load_spec}'"))?;
    response
        .await
        .map_err(|err| format!("runtime unload response channel closed: {err}"))?
        .map_err(|err| err.to_string())
}

pub(crate) fn emit_model_target_reconciliation_queued(action: &ModelTargetReconciliationAction) {
    let context = match action.replace_model_ref.as_deref() {
        Some(replace_model_ref) => Some(format!("replace={replace_model_ref}")),
        None => Some(format!("path={}", action.load_spec.display())),
    };
    let verb = if action.replace_model_ref.is_some() {
        "upgrading to"
    } else {
        "loading"
    };
    let _ = emit_event(OutputEvent::Info {
        message: format!("Model target reconciliation {verb} '{}'", action.model_ref),
        context,
    });
}

pub(crate) fn runtime_loaded_model_refs(
    runtime_models: &HashMap<String, RuntimeModelHandleEntry>,
    managed_models: &HashMap<String, ManagedModelController>,
) -> BTreeSet<String> {
    runtime_models
        .values()
        .map(|entry| entry.model_name.clone())
        .chain(
            managed_models
                .values()
                .map(|controller| controller.model_name.clone()),
        )
        .collect()
}

pub(crate) fn local_model_path_for_reconciliation_target(
    target: &api::status::ModelTargetPayload,
) -> Option<PathBuf> {
    [
        Some(target.model_ref.as_str()),
        target.model_name.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(models::find_model_path)
    .find(|path| path.exists())
}

pub(crate) fn model_target_reconciliation_local_fit(
    target: &api::status::ModelTargetPayload,
    local_vram_bytes: u64,
) -> bool {
    target
        .capacity_advice
        .required_bytes
        .is_some_and(|required| local_vram_bytes >= required)
}

pub(crate) fn model_target_reconciliation_demand_upgrade_candidate(
    policy: &ModelTargetReconciliationPolicy,
    loaded_model_refs: &BTreeSet<String>,
    target: &api::status::ModelTargetPayload,
) -> bool {
    policy.demand_upgrades_enabled
        && !loaded_model_refs.is_empty()
        && target.wanted_reason == Some("active_demand")
        && target.request_count >= policy.demand_upgrade_min_request_count
        && target
            .last_active_secs_ago
            .is_some_and(|age| age <= policy.demand_upgrade_max_age_secs)
}

pub(crate) fn runtime_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
