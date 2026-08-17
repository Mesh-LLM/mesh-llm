#[tokio::test]
async fn test_api_model_interests_post_and_get_round_trip() {
    let state = build_test_mesh_api().await;
    let (post_addr, post_handle) = spawn_management_test_server(state.clone()).await;
    let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M","source":"ui"}"#;

    let post_response = send_management_request(
            post_addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

    assert!(post_response.starts_with("HTTP/1.1 201"));
    let post_payload = json_body(&post_response);
    assert_eq!(post_payload["created"], json!(true));
    assert_eq!(
        post_payload["interest"]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(post_payload["interest"]["submission_source"], json!("ui"));
    assert_eq!(post_payload["model_interests"].as_array().unwrap().len(), 1);
    post_handle.abort();

    let (get_addr, get_handle) = spawn_management_test_server(state).await;
    let get_response = send_management_request(
        get_addr,
        "GET /api/model-interests HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(get_response.starts_with("HTTP/1.1 200"));
    let get_payload = json_body(&get_response);
    let interests = get_payload["model_interests"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert_eq!(interests.len(), 1);
    assert_eq!(
        interests[0]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(interests[0]["submission_source"], json!("ui"));

    get_handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_post_is_idempotent() {
    let state = build_test_mesh_api().await;
    let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M","source":"ui"}"#;
    let request = format!(
        "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let (first_addr, first_handle) = spawn_management_test_server(state.clone()).await;
    let first_response = send_management_request(first_addr, request.clone()).await;
    assert!(first_response.starts_with("HTTP/1.1 201"));
    let first_payload = json_body(&first_response);
    let created_at = first_payload["interest"]["created_at_unix"]
        .as_u64()
        .expect("created_at_unix");
    first_handle.abort();

    let (second_addr, second_handle) = spawn_management_test_server(state).await;
    let second_response = send_management_request(second_addr, request).await;
    assert!(second_response.starts_with("HTTP/1.1 200"));
    let second_payload = json_body(&second_response);
    assert_eq!(second_payload["created"], json!(false));
    assert_eq!(
        second_payload["interest"]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(
        second_payload["interest"]["created_at_unix"],
        json!(created_at)
    );
    assert_eq!(
        second_payload["model_interests"].as_array().unwrap().len(),
        1
    );

    second_handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_delete_decodes_percent_encoded_model_ref() {
    let state = build_test_mesh_api().await;
    state
        .upsert_model_interest(
            crate::models::canonicalize_interest_model_ref(
                "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M",
            )
            .unwrap(),
            Some("ui".to_string()),
        )
        .await;

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
            addr,
            "DELETE /api/model-interests/Qwen%2FQwen3-Coder-Next-GGUF%40main%3AQ4_K_M HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["removed"], json!(true));
    assert_eq!(
        payload["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );
    assert_eq!(payload["model_interests"], json!([]));

    handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_delete_rejects_empty_model_ref_path() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "DELETE /api/model-interests/ HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(payload["error"], json!("Missing model interest path"));

    handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_delete_rejects_malformed_model_ref_path() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "DELETE /api/model-interests/Qwen%2 HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(payload["error"], json!("Missing model interest path"));

    handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_reject_direct_urls() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = r#"{"model_ref":"https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"}"#;

    let response = send_management_request(
            addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(
        payload["error"],
        json!("Invalid 'model_ref'. Use a canonical ref returned by /api/search, not a direct URL")
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_model_interests_normalize_legacy_selector_revision_order() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M@main","source":"ui"}"#;

    let response = send_management_request(
            addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 201"));
    let payload = json_body(&response);
    assert_eq!(
        payload["interest"]["model_ref"],
        json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_model_targets_combine_interest_demand_and_serving_visibility() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let model_ref = qwen_coder_remote_catalog_ref();
    let (interest, _) = state
        .upsert_model_interest(model_ref.clone(), Some("ui".to_string()))
        .await;
    assert_eq!(
        node.explicit_model_interests().await,
        vec![model_ref.clone()]
    );

    node.record_request(&model_ref);

    let mut peer = make_test_peer(
        0x44,
        mesh::NodeRole::Host { http_port: 9337 },
        vec![model_ref.as_str()],
        vec![model_ref.as_str()],
        true,
    );
    peer.explicit_model_interests = vec![interest.model_ref.clone()];
    node.insert_test_peer(peer).await;

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/model-targets HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let targets = payload["model_targets"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let target = targets
        .into_iter()
        .find(|entry| entry["model_ref"] == interest.model_ref)
        .expect("target for explicit interest present");
    assert_eq!(target["derived"]["target_rank"], json!(1));
    assert_eq!(target["signals"]["explicit_interest_count"], json!(2));
    assert_eq!(target["signals"]["request_count"], json!(1));
    assert_eq!(target["signals"]["serving_node_count"], json!(1));
    assert_eq!(target["signals"]["requested"], json!(false));
    assert_eq!(target["derived"]["wanted"], json!(false));
    assert!(target.get("rank").is_none());
    assert!(target.get("explicit_interest_count").is_none());
    assert!(target.get("wanted").is_none());

    handle.abort();
}

#[tokio::test]
#[serial]
async fn test_api_model_targets_surface_capacity_advice_under_derived() {
    let _catalog_guard = crate::models::remote_catalog::set_catalog_entries_for_test(vec![
        qwen_coder_remote_catalog_entry(),
    ]);
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    node.set_role(mesh::NodeRole::Client).await;
    let model_ref = qwen_coder_remote_catalog_ref();
    let (interest, _) = state
        .upsert_model_interest(model_ref.clone(), Some("ui".to_string()))
        .await;

    node.insert_test_peer(make_test_peer(
        0x45,
        mesh::NodeRole::Worker,
        Vec::new(),
        Vec::new(),
        true,
    ))
    .await;

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/model-targets HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let target = payload["model_targets"]
        .as_array()
        .and_then(|targets| {
            targets
                .iter()
                .find(|entry| entry["model_ref"] == interest.model_ref)
        })
        .expect("target for explicit interest present");
    assert_eq!(target["derived"]["target_rank"], json!(1));
    assert_eq!(target["derived"]["wanted"], json!(true));
    assert!(target.get("capacity_advice").is_none());

    let advice = &target["derived"]["capacity_advice"];
    assert_eq!(advice["state"], json!("single_node_fit"));
    assert_eq!(advice["reason"], json!("single_node_capacity_available"));
    assert_eq!(advice["required_bytes"], json!(22_000_000_000_u64));
    assert_eq!(
        advice["best_single_node_capacity_bytes"],
        json!(24_000_000_000_u64)
    );
    assert_eq!(
        advice["aggregate_capacity_bytes"],
        json!(24_000_000_000_u64)
    );
    assert_eq!(advice["eligible_node_count"], json!(1));
    assert_eq!(advice["missing_capacity_node_count"], json!(0));
    assert_eq!(advice["excluded_client_node_count"], json!(1));

    handle.abort();
}

#[tokio::test]
#[serial]
async fn test_api_model_targets_capacity_advice_stays_unknown_with_partial_capacity() {
    let _catalog_guard = crate::models::remote_catalog::set_catalog_entries_for_test(vec![
        qwen_coder_remote_catalog_entry(),
    ]);
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let model_ref = qwen_coder_remote_catalog_ref();
    let (interest, _) = state
        .upsert_model_interest(model_ref.clone(), Some("ui".to_string()))
        .await;

    node.insert_test_peer(make_test_peer(
        0x48,
        mesh::NodeRole::Worker,
        Vec::new(),
        Vec::new(),
        true,
    ))
    .await;

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/model-targets HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let target = payload["model_targets"]
        .as_array()
        .and_then(|targets| {
            targets
                .iter()
                .find(|entry| entry["model_ref"] == interest.model_ref)
        })
        .expect("target for explicit interest present");

    let advice = &target["derived"]["capacity_advice"];
    assert_eq!(advice["state"], json!("unknown_capacity"));
    assert_eq!(advice["reason"], json!("eligible_nodes_missing_capacity"));
    assert_eq!(advice["required_bytes"], json!(22_000_000_000_u64));
    assert_eq!(
        advice["best_single_node_capacity_bytes"],
        json!(24_000_000_000_u64)
    );
    assert_eq!(
        advice["aggregate_capacity_bytes"],
        json!(24_000_000_000_u64)
    );
    assert!(advice.get("shortfall_bytes").is_none());
    assert_eq!(advice["eligible_node_count"], json!(1));
    assert_eq!(advice["missing_capacity_node_count"], json!(1));

    handle.abort();
}

#[tokio::test]
#[serial]
async fn test_api_model_targets_capacity_advice_separates_clients_from_missing_vram() {
    let _catalog_guard = crate::models::remote_catalog::set_catalog_entries_for_test(vec![
        qwen_coder_remote_catalog_entry(),
    ]);
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let model_ref = qwen_coder_remote_catalog_ref();
    let (interest, _) = state
        .upsert_model_interest(model_ref.clone(), Some("ui".to_string()))
        .await;

    let mut client_with_vram =
        make_test_peer(0x46, mesh::NodeRole::Client, Vec::new(), Vec::new(), true);
    client_with_vram.vram_bytes = 128_000_000_000;
    node.insert_test_peer(client_with_vram).await;

    let mut worker_missing_vram =
        make_test_peer(0x47, mesh::NodeRole::Worker, Vec::new(), Vec::new(), true);
    worker_missing_vram.vram_bytes = 0;
    node.insert_test_peer(worker_missing_vram).await;

    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/model-targets HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    let target = payload["model_targets"]
        .as_array()
        .and_then(|targets| {
            targets
                .iter()
                .find(|entry| entry["model_ref"] == interest.model_ref)
        })
        .expect("target for explicit interest present");

    let advice = &target["derived"]["capacity_advice"];
    assert_eq!(advice["state"], json!("unknown_capacity"));
    assert_eq!(advice["reason"], json!("eligible_nodes_missing_capacity"));
    assert_eq!(advice["required_bytes"], json!(22_000_000_000_u64));
    assert_eq!(advice["eligible_node_count"], json!(0));
    assert_eq!(advice["missing_capacity_node_count"], json!(2));
    assert_eq!(advice["excluded_client_node_count"], json!(1));
    assert!(advice.get("best_single_node_capacity_bytes").is_none());

    handle.abort();
}

#[tokio::test]
async fn test_api_status_and_models_surface_wanted_targets() {
    let state = build_test_mesh_api().await;
    let node = {
        let inner = state.inner.lock().await;
        inner.node.clone()
    };
    let model_ref = qwen_coder_remote_catalog_ref();
    let (interest, _) = state
        .upsert_model_interest(model_ref.clone(), Some("ui".to_string()))
        .await;
    node.set_requested_models(vec![model_ref.clone()]).await;

    let (status_addr, status_handle) = spawn_management_test_server(state.clone()).await;
    let status_response = send_management_request(
        status_addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(status_response.starts_with("HTTP/1.1 200"));
    let status_payload = json_body(&status_response);
    assert_eq!(
        status_payload["wanted_model_refs"],
        json!([interest.model_ref.clone()])
    );
    status_handle.abort();

    let (models_addr, models_handle) = spawn_management_test_server(state).await;
    let models_response = send_management_request(
        models_addr,
        "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(models_response.starts_with("HTTP/1.1 200"));
    let models_payload = json_body(&models_response);
    let models = models_payload["mesh_models"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let model = models
        .into_iter()
        .find(|entry| entry["name"] == model_ref)
        .expect("catalog model present");
    assert_eq!(model["target_rank"], json!(1));
    assert_eq!(model["explicit_interest_count"], json!(1));
    assert_eq!(model["wanted"], json!(true));

    models_handle.abort();
}

#[test]
fn test_http_route_stats_only_count_http_callable_legacy_hosts() {
    let peers = vec![
        make_test_peer(
            0x41,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["legacy-host-model"],
            Vec::new(),
            false,
        ),
        make_test_peer(
            0x42,
            mesh::NodeRole::Worker,
            vec!["worker-only-model"],
            Vec::new(),
            false,
        ),
    ];

    let host_stats = http_route_stats("legacy-host-model", &peers, &[], None, 0.0);
    assert_eq!(host_stats.node_count, 1);
    assert_eq!(host_stats.active_nodes.len(), 1);
    assert!(host_stats.mesh_vram_gb > 0.0);

    let worker_stats = http_route_stats("worker-only-model", &peers, &[], None, 0.0);
    assert_eq!(worker_stats, HttpRouteStats::default());
}
