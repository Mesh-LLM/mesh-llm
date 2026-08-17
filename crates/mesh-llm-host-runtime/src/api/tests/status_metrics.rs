#[tokio::test]
async fn test_api_status_includes_local_gpu_benchmark_metrics() {
    let state = build_test_mesh_api().await;
    let node = {
        let mut inner = state.inner.lock().await;
        inner.node.gpu_name = Some("NVIDIA A100".into());
        inner.node.gpu_vram = Some("85899345920".into());
        inner.node.gpu_reserved_bytes = Some("1073741824".into());
        inner.node.hostname = Some("worker-01".into());
        inner.node.is_soc = Some(false);
        inner.node.clone()
    };
    *node.gpu_mem_bandwidth_gbps.lock().await = Some(vec![1948.7]);
    *node.gpu_compute_tflops_fp32.lock().await = Some(vec![19.5]);
    *node.gpu_compute_tflops_fp16.lock().await = Some(vec![312.0]);

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let gpu = &payload["gpus"][0];
    assert_eq!(gpu["name"], json!("NVIDIA A100"));
    assert_eq!(gpu["vram_bytes"], json!(85899345920_u64));
    assert_eq!(gpu["reserved_bytes"], json!(1073741824_u64));
    assert_eq!(gpu["mem_bandwidth_gbps"], json!(1948.7));
    assert_eq!(gpu["compute_tflops_fp32"], json!(19.5));
    assert_eq!(gpu["compute_tflops_fp16"], json!(312.0));

    handle.abort();
}

#[tokio::test]
async fn test_api_status_includes_routing_metrics_summary() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());

    node.record_inference_attempt(
        Some("test-model"),
        &election::InferenceTarget::Local(9338),
        Duration::from_millis(4),
        Duration::from_millis(16),
        crate::network::metrics::AttemptOutcome::Timeout,
        None,
    );
    node.record_inference_attempt(
        Some("test-model"),
        &election::InferenceTarget::Remote(peer_id),
        Duration::from_millis(18),
        Duration::from_millis(48),
        crate::network::metrics::AttemptOutcome::Success,
        Some(12),
    );
    node.record_routed_request(
        Some("test-model"),
        2,
        crate::network::metrics::RequestOutcome::Success(
            crate::network::metrics::RequestService::Remote,
        ),
    );

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["routing_metrics"]["request_count"], json!(1));
    assert_eq!(payload["routing_metrics"]["successful_requests"], json!(1));
    assert_eq!(payload["routing_metrics"]["retry_count"], json!(1));
    assert_eq!(payload["routing_metrics"]["failover_count"], json!(1));
    assert_eq!(
        payload["routing_metrics"]["attempt_timeout_count"],
        json!(1)
    );
    assert_eq!(
        payload["routing_metrics"]["pressure"]["remotely_served_request_count"],
        json!(1)
    );
    assert_eq!(
        payload["routing_metrics"]["local_node"]["remote_attempt_count"],
        json!(1)
    );
    assert_eq!(
        payload["routing_metrics"]["local_node"]["local_attempt_count"],
        json!(1)
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_models_include_model_routing_metrics() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let model_ref = qwen_coder_remote_catalog_ref();
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    node.set_requested_models(vec![model_ref.clone()]).await;

    node.record_inference_attempt(
        Some(&model_ref),
        &election::InferenceTarget::Remote(peer_id),
        Duration::from_millis(6),
        Duration::from_millis(24),
        crate::network::metrics::AttemptOutcome::Success,
        Some(9),
    );
    node.record_routed_request(
        Some(&model_ref),
        1,
        crate::network::metrics::RequestOutcome::Success(
            crate::network::metrics::RequestService::Remote,
        ),
    );

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let models = payload["mesh_models"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let model = models
        .into_iter()
        .find(|entry| entry["name"] == model_ref)
        .expect("catalog model present");
    assert_eq!(model["routing_metrics"]["request_count"], json!(1));
    assert_eq!(model["routing_metrics"]["successful_requests"], json!(1));
    assert_eq!(
        model["routing_metrics"]["targets"][0]["kind"],
        json!("remote")
    );
    assert_eq!(
        model["routing_metrics"]["targets"][0]["success_count"],
        json!(1)
    );

    handle.abort();
}
