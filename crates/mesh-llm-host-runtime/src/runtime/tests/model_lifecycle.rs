use super::*;

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
            profile,
            resp,
        }) => {
            assert_eq!(spec, "/models/large.gguf");
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
fn runtime_unload_target_requires_instance_id_for_duplicate_models() {
    let err = resolve_runtime_unload_target(
        "Qwen",
        vec![
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Runtime,
                instance_id: "runtime-1".to_string(),
                model_name: "Qwen".to_string(),
            },
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Managed,
                instance_id: "runtime-2".to_string(),
                model_name: "Qwen".to_string(),
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
            },
            RuntimeUnloadCandidate {
                owner: RuntimeUnloadOwner::Managed,
                instance_id: "runtime-2".to_string(),
                model_name: "Qwen".to_string(),
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
