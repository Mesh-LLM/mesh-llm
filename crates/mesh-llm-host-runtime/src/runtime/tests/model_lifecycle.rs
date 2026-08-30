use super::*;
use crate::runtime::model_reconciliation::intent::DesiredModelState;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn reconciliation_target_with_required_bytes(
    required_bytes: Option<u64>,
) -> api::status::ModelTargetPayload {
    api::status::ModelTargetPayload {
        rank: 1,
        model_ref: "org/model@main:model.gguf".to_string(),
        display_name: "Model".to_string(),
        profile: String::new(),
        model_name: Some("Model".to_string()),
        explicit_interest_count: 1,
        request_count: 0,
        last_active_secs_ago: None,
        serving_node_count: 0,
        requested: false,
        wanted: true,
        wanted_reason: Some("explicit_interest"),
        capacity_advice: api::status::ModelTargetCapacityAdvicePayload {
            state: api::status::ModelTargetCapacityAdviceState::SingleNodeFit,
            reason: "single_node_capacity_available",
            required_bytes,
            best_single_node_capacity_bytes: required_bytes,
            aggregate_capacity_bytes: required_bytes.unwrap_or_default(),
            shortfall_bytes: None,
            eligible_node_count: 1,
            missing_capacity_node_count: 0,
            excluded_client_node_count: 0,
            split_capable: false,
        },
    }
}

#[test]
fn model_target_reconciliation_local_fit_requires_current_node_capacity() {
    let target = reconciliation_target_with_required_bytes(Some(10));

    assert!(model_target_reconciliation_local_fit(&target, 10));
    assert!(!model_target_reconciliation_local_fit(&target, 9));
}

#[test]
fn model_target_reconciliation_local_fit_rejects_unknown_required_bytes() {
    let target = reconciliation_target_with_required_bytes(None);

    assert!(!model_target_reconciliation_local_fit(&target, u64::MAX));
}

#[tokio::test]
async fn model_target_reconciliation_replacement_unloads_before_loading() {
    let (control_tx, mut control_rx) =
        tokio::sync::mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    let profile = "low-ctx".to_string();
    let task = tokio::spawn(run_model_target_reconciliation_action(
        control_tx,
        "org/large-model".to_string(),
        "/models/large.gguf".to_string(),
        Some("Small".to_string()),
        profile.clone(),
    ));

    match control_rx.recv().await {
        Some(api::RuntimeControlRequest::Unload { target, resp, .. }) => {
            assert_eq!(target.as_runtime_target(), "Small");
            resp.send(Ok(api::RuntimeUnloadResponse {
                model: "Small".to_string(),
                instance_id: "runtime-1".to_string(),
                unloaded: true,
            }))
            .expect("replacement unload response should be received");
        }
        _ => panic!("expected unload request before load"),
    }
    match control_rx.recv().await {
        Some(api::RuntimeControlRequest::Load {
            spec,
            config_model_id,
            profile,
            resp,
        }) => {
            assert_eq!(spec, "/models/large.gguf");
            assert_eq!(config_model_id.as_deref(), Some("org/large-model"));
            assert_eq!(profile, "low-ctx");
            resp.send(Ok(api::RuntimeLoadResponse {
                model_ref: spec,
                model: "Large".to_string(),
                instance_id: "runtime-2".to_string(),
                profile,
                backend: Some("skippy".to_string()),
                context_length: Some(4096),
            }))
            .expect("replacement load response should be received");
        }
        _ => panic!("expected load request after unload"),
    }

    let result = task
        .await
        .expect("replacement task should join")
        .expect("replacement action should finish");
    assert_eq!(result.model, "Large");
    assert!(control_rx.try_recv().is_err());
}

#[test]
fn startup_load_finished_notifies_stacked_load_callers() {
    // Given: two same-key callers are stacked behind a startup load already in flight.
    let mut state = ModelTargetReconciliationState::default();
    let model_ref = "org/model@main:model.gguf";
    let profile = "low-ctx";
    state.mark_load_started(model_ref, profile);
    let (first_tx, first_rx) = tokio::sync::oneshot::channel();
    let (second_tx, second_rx) = tokio::sync::oneshot::channel();
    state.stack_load_completion(model_ref, profile, first_tx);
    state.stack_load_completion(model_ref, profile, second_tx);
    let response = api::RuntimeLoadResponse {
        model_ref: model_ref.to_string(),
        model: "Model".to_string(),
        instance_id: "runtime-1".to_string(),
        profile: profile.to_string(),
        backend: Some("skippy".to_string()),
        context_length: Some(4096),
    };

    // When: startup reports the load as finished through the reconciliation seam.
    crate::runtime::model_lifecycle::apply_startup_model_load_finished(
        &mut state,
        &ModelTargetReconciliationPolicy::default(),
        model_ref,
        profile,
        Ok(response),
        1,
    );

    // Then: every stacked same-key caller receives the typed runtime load response.
    let first = first_rx
        .blocking_recv()
        .expect("first stacked load caller should receive completion")
        .expect("startup load should succeed");
    let second = second_rx
        .blocking_recv()
        .expect("second stacked load caller should receive completion")
        .expect("startup load should succeed");
    assert_eq!(first.instance_id, "runtime-1");
    assert_eq!(second.profile, "low-ctx");
    assert!(!state.is_load_pending(model_ref, profile));
}

#[test]
fn reconciliation_local_path_load_notifies_stacked_canonical_caller() {
    // Given: reconciliation tracks the canonical catalog identity while executing a local path.
    let mut state = ModelTargetReconciliationState::default();
    let model_ref = "org/model@main:model.gguf";
    let load_spec = "/models/model.gguf";
    let profile = "low-ctx";
    state.mark_load_started(model_ref, profile);
    let (caller_tx, mut caller_rx) = tokio::sync::oneshot::channel();
    state.stack_load_completion(model_ref, profile, caller_tx);
    let response = api::RuntimeLoadResponse {
        model_ref: load_spec.to_string(),
        model: "Model".to_string(),
        instance_id: "runtime-1".to_string(),
        profile: profile.to_string(),
        backend: Some("skippy".to_string()),
        context_length: Some(4096),
    };

    // When: the internal path-keyed load finishes before the canonical reconciliation event.
    state.record_load_success(load_spec, profile);
    state.notify_load_success(load_spec, profile, response.clone());

    // Then: the canonical API/owner caller remains stacked until that stable identity finishes.
    assert!(matches!(
        caller_rx.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    crate::runtime::model_lifecycle::apply_model_target_reconciliation_load_finished(
        &mut state,
        &ModelTargetReconciliationPolicy::default(),
        model_ref,
        profile,
        Ok(response),
        1,
    )
    .expect("canonical reconciliation result should succeed");
    let caller = caller_rx
        .blocking_recv()
        .expect("stacked canonical caller should receive completion")
        .expect("reconciliation load should succeed");
    assert_eq!(caller.instance_id, "runtime-1");
    assert!(!state.is_load_pending(model_ref, profile));
}

#[test]
fn resolved_unload_of_profiled_load_prevents_reconciliation_reload() {
    // Given: default and low-context intents share a model name, and the low profile has loaded.
    let policy = ModelTargetReconciliationPolicy {
        enabled: true,
        ..ModelTargetReconciliationPolicy::default()
    };
    let mut state = ModelTargetReconciliationState::default();
    state.add_desired("Qwen", "", IntentSource::StartupConfig);
    state.add_desired("Qwen", "low-ctx", IntentSource::StartupConfig);
    state.mark_load_started("Qwen", "low-ctx");
    crate::runtime::model_lifecycle::apply_startup_model_load_finished(
        &mut state,
        &policy,
        "Qwen",
        "low-ctx",
        Ok(api::RuntimeLoadResponse {
            model_ref: "Qwen".to_string(),
            model: "Qwen".to_string(),
            instance_id: "runtime-low".to_string(),
            profile: "low-ctx".to_string(),
            backend: Some("skippy".to_string()),
            context_length: Some(4096),
        }),
        1,
    );
    let resolved = resolve_runtime_unload_target(
        "runtime-low",
        vec![
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Runtime,
                instance_id: "runtime-default".to_string(),
                model_name: "Qwen".to_string(),
                profile: String::new(),
            },
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Runtime,
                instance_id: "runtime-low".to_string(),
                model_name: "Qwen".to_string(),
                profile: "low-ctx".to_string(),
            },
        ],
    )
    .expect("exact loaded profile instance should resolve");

    // When: the resolved loaded profile is unloaded through the desired-state seam.
    crate::runtime::model_lifecycle::suppress_desired_for_resolved_unload_candidate(
        &mut state,
        &resolved,
        IntentSource::ApiUnload,
        false,
        None,
    );

    // Then: the low profile is absent from effective desired state, so it is not reloaded.
    let targets = reconciliation_targets_from_desired(&state);
    let local_interests = local_interests_from_desired(&state);
    let loaded = BTreeSet::new();
    let actions = plan_model_target_reconciliation(
        &policy,
        &mut state,
        ModelTargetReconciliationInput {
            now_secs: 2,
            local_role: mesh::NodeRole::Host { http_port: 9337 },
            local_interest_model_refs: &local_interests,
            loaded_model_refs: &loaded,
            targets: &targets,
        },
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].model_ref, "Qwen");
    assert_eq!(actions[0].profile, "");
    assert!(state.is_desired("Qwen", ""));
    assert!(!state.is_desired("Qwen", "low-ctx"));
}

fn local_interests_from_desired(state: &ModelTargetReconciliationState) -> BTreeSet<String> {
    state
        .intent_history()
        .iter()
        .filter(|intent| {
            intent.desired_state == DesiredModelState::Present
                && state.is_effective_intent(
                    &intent.intent_id,
                    &intent.canonical_model_ref,
                    &intent.profile,
                )
        })
        .map(|intent| intent.canonical_model_ref.clone())
        .collect()
}

fn reconciliation_targets_from_desired(
    state: &ModelTargetReconciliationState,
) -> Vec<ModelTargetReconciliationCandidate> {
    state
        .intent_history()
        .iter()
        .filter(|intent| {
            intent.desired_state == DesiredModelState::Present
                && state.is_effective_intent(
                    &intent.intent_id,
                    &intent.canonical_model_ref,
                    &intent.profile,
                )
        })
        .map(|intent| ModelTargetReconciliationCandidate {
            rank: 1,
            model_ref: intent.canonical_model_ref.clone(),
            profile: intent.profile.clone(),
            model_name: Some(intent.canonical_model_ref.clone()),
            wanted: true,
            wanted_reason: Some("explicit_interest"),
            request_count: 0,
            last_active_secs_ago: None,
            serving_node_count: 0,
            capacity_state: ModelTargetReconciliationCapacityState::SingleNodeFit,
            local_path: Some(PathBuf::from(format!(
                "/models/{}.gguf",
                intent.canonical_model_ref
            ))),
        })
        .collect()
}

#[test]
fn runtime_unload_target_requires_instance_id_for_duplicate_models() {
    let err = resolve_runtime_unload_target(
        "Qwen",
        vec![
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Runtime,
                instance_id: "runtime-1".to_string(),
                model_name: "Qwen".to_string(),
                profile: String::new(),
            },
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Managed,
                instance_id: "runtime-2".to_string(),
                model_name: "Qwen".to_string(),
                profile: String::new(),
            },
        ],
    )
    .expect_err("duplicate model-name unload should be ambiguous");

    assert!(err.to_string().contains("multiple loaded instances"));
}

#[test]
fn runtime_unload_target_resolves_exact_instance_before_model_name() {
    let target = resolve_runtime_unload_target(
        "runtime-2",
        vec![
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Runtime,
                instance_id: "runtime-1".to_string(),
                model_name: "runtime-2".to_string(),
                profile: String::new(),
            },
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Managed,
                instance_id: "runtime-2".to_string(),
                model_name: "Qwen".to_string(),
                profile: String::new(),
            },
        ],
    )
    .expect("exact instance id should resolve");

    assert_eq!(target.instance_id, "runtime-2");
    assert_eq!(target.model_name, "Qwen");
    assert_eq!(target.owner, RuntimeUnloadOwner::Managed);
}

#[tokio::test]
async fn register_runtime_instance_preserves_existing_known_descriptor_capabilities() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should initialize");
    let registry = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
    let vision_model = "Qwen3VL-2B-Instruct-Q4_K_M";
    let text_model = "Qwen3-8B-Q4_K_M";
    let vision_capabilities = models::ModelCapabilities {
        multimodal: true,
        vision: models::CapabilityLevel::Supported,
        ..Default::default()
    };

    register_runtime_instance(
        &registry,
        &node,
        vision_model,
        vision_model,
        "runtime-vision",
        Some(8192),
        vision_capabilities,
    )
    .await;
    register_runtime_instance(
        &registry,
        &node,
        vision_model,
        text_model,
        "runtime-text",
        Some(8192),
        models::ModelCapabilities::default(),
    )
    .await;

    let descriptors = node.served_model_descriptors().await;
    let vision = descriptors
        .iter()
        .find(|descriptor| descriptor.identity.model_name == vision_model)
        .expect("vision descriptor should remain registered");
    assert!(vision.capabilities_known);
    assert_eq!(vision.capabilities, vision_capabilities);

    let text = descriptors
        .iter()
        .find(|descriptor| descriptor.identity.model_name == text_model)
        .expect("text descriptor should be registered");
    assert!(text.capabilities_known);
    assert_eq!(text.capabilities, models::ModelCapabilities::default());
}

#[tokio::test]
async fn test_runtime_load_unload_regossips_across_nodes() {
    let host = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let observer = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();

    host.set_role(mesh::NodeRole::Host { http_port: 9337 })
        .await;
    host.set_serving_models(vec!["Primary".into()]).await;
    host.set_hosted_models(vec!["Primary".into()]).await;

    observer.sync_from_peer_for_tests(&host).await;

    wait_for_condition(Duration::from_secs(5), || {
        let observer = observer.clone();
        let host_id = host.id();
        async move {
            observer.peers().await.iter().any(|peer| {
                peer.id == host_id && peer.routes_model("Primary") && !peer.routes_model("Runtime")
            })
        }
    })
    .await;

    add_serving_assignment(&host, "Primary", "Runtime").await;
    advertise_model_ready(&host, "Primary", "Runtime", "").await;
    observer.sync_from_peer_for_tests(&host).await;

    wait_for_condition(Duration::from_secs(5), || {
        let observer = observer.clone();
        let host_id = host.id();
        async move {
            observer.peers().await.iter().any(|peer| {
                peer.id == host_id
                    && peer.is_assigned_model("Runtime")
                    && peer.routes_model("Runtime")
                    && peer.routable_models() == vec!["Primary".to_string(), "Runtime".to_string()]
            })
        }
    })
    .await;

    remove_serving_assignment(&host, "Runtime").await;
    withdraw_advertised_model(&host, "Runtime", "").await;
    observer.sync_from_peer_for_tests(&host).await;

    wait_for_condition(Duration::from_secs(5), || {
        let observer = observer.clone();
        let host_id = host.id();
        async move {
            observer.peers().await.iter().any(|peer| {
                peer.id == host_id
                    && peer.routes_model("Primary")
                    && !peer.is_assigned_model("Runtime")
                    && !peer.routes_model("Runtime")
                    && peer.routable_models() == vec!["Primary".to_string()]
            })
        }
    })
    .await;
}

#[tokio::test]
async fn test_benchmark_result_bandwidth_still_works() {
    let mem_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
    let fp32_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
    let fp16_arc = std::sync::Arc::new(tokio::sync::Mutex::new(None));
    let result = benchmark::BenchmarkResult {
        mem_bandwidth_gbps: vec![10.5, 20.0],
        compute_tflops_fp32: None,
        compute_tflops_fp16: None,
    };

    store_benchmark_metrics(
        mem_arc.clone(),
        fp32_arc.clone(),
        fp16_arc.clone(),
        Some(&result),
    )
    .await;

    assert_eq!(*mem_arc.lock().await, Some(vec![10.5, 20.0]));
    assert!(fp32_arc.lock().await.is_none());
    assert!(fp16_arc.lock().await.is_none());
}

#[test]
fn runtime_load_ctx_size_uses_model_override_when_cli_is_unset() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let model = plugin::ModelConfigEntry {
        model: "runtime/model".to_string(),
        ctx_size: Some(16_384),
        ..Default::default()
    };

    assert_eq!(
        runtime_model_ctx_size_override(&options, Some(&model)),
        Some(16_384)
    );
}

#[test]
fn runtime_load_ctx_size_prefers_cli_override_over_model_override() {
    let options = runtime_options_for_test(&["mesh-llm", "--ctx-size", "8192"]);
    let model = plugin::ModelConfigEntry {
        model: "runtime/model".to_string(),
        ctx_size: Some(16_384),
        ..Default::default()
    };

    assert_eq!(
        runtime_model_ctx_size_override(&options, Some(&model)),
        Some(8192)
    );
}
