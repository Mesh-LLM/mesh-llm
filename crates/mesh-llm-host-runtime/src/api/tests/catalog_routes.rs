#[tokio::test]
async fn test_api_status_excludes_mesh_models_and_models_endpoint_serves_them() {
    let state = build_test_mesh_api().await;
    let (status_addr, status_handle) = spawn_management_test_server(state.clone()).await;
    let status_response = send_management_request(
        status_addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(status_response.starts_with("HTTP/1.1 200"));
    let status_body = json_body(&status_response);
    assert!(status_body.get("mesh_models").is_none());
    status_handle.abort();

    let (models_addr, models_handle) = spawn_management_test_server(state).await;
    let models_response = send_management_request(
        models_addr,
        "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(models_response.starts_with("HTTP/1.1 200"));
    let models_body = json_body(&models_response);
    assert!(models_body.get("mesh_models").is_some());

    models_handle.abort();
}

#[tokio::test]
#[serial]
async fn test_api_search_catalog_returns_canonical_model_refs() {
    let _catalog_guard = crate::models::remote_catalog::set_catalog_entries_for_test(vec![
        qwen_coder_remote_catalog_entry(),
    ]);
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
            addr,
            "GET /api/search?q=Qwen3-Coder-Next&catalog=true&artifact=gguf&limit=5&sort=trending HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["source"], json!("catalog"));
    assert_eq!(payload["filter"], json!("gguf"));
    assert_eq!(payload["sort"], json!("trending"));
    assert!(payload.get("machine").is_some());
    let results = payload["results"].as_array().cloned().unwrap_or_default();
    assert!(
        !results.is_empty(),
        "expected at least one catalog result for Qwen3-Coder-Next"
    );
    let catalog_ref = qwen_coder_remote_catalog_ref();
    let hit = results
        .into_iter()
        .find(|entry| entry["ref"] == json!(catalog_ref))
        .expect("canonical catalog model ref present");
    assert_eq!(hit["repo_id"], json!("Qwen/Qwen3-Coder-Next-GGUF"));
    assert_eq!(hit["type"], json!("gguf"));
    assert_eq!(
        hit["show"],
        json!(format!("mesh-llm models show {catalog_ref}"))
    );

    handle.abort();
}

#[tokio::test]
#[serial]
async fn test_api_search_caps_limit_and_uses_canonical_parameter_sort_name() {
    let _catalog_guard =
        crate::models::remote_catalog::set_catalog_entries_for_test(qwen_coder_catalog_entries(60));
    let uncapped_matches = crate::models::search_catalog_models("Qwen3-Coder-Next").unwrap();
    assert!(
        uncapped_matches.len() > 50,
        "test catalog must contain more matches than the API cap"
    );
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
            addr,
            "GET /api/search?q=Qwen3-Coder-Next&catalog=true&artifact=gguf&limit=999&sort=parameters-desc HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

    assert!(response.starts_with("HTTP/1.1 200"));
    let payload = json_body(&response);
    assert_eq!(payload["sort"], json!("parameters-desc"));
    let results = payload["results"].as_array().cloned().unwrap_or_default();
    assert_eq!(results.len(), 50);

    handle.abort();
}

fn qwen_coder_catalog_entries(count: usize) -> Vec<crate::models::remote_catalog::CatalogEntry> {
    (0..count)
        .map(|index| {
            let mut entry = qwen_coder_remote_catalog_entry();
            entry.source_repo = format!("Qwen/Qwen3-Coder-Next-GGUF-{index:03}");
            let mut variant = entry.variants.remove("Qwen3-Coder-Next-Q4_K_M").unwrap();
            variant.source.repo.clone_from(&entry.source_repo);
            variant.source.file = Some(format!("Qwen3-Coder-Next-Q4_K_M-{index:03}.gguf"));
            variant.curated.name = format!("Qwen3-Coder-Next-Q4_K_M-{index:03}");
            entry.variants.insert(variant.curated.name.clone(), variant);
            entry
        })
        .collect()
}

#[tokio::test]
async fn test_api_search_requires_q_query_parameter() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/search?catalog=true HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(
        payload["error"],
        json!("Missing required 'q' query parameter")
    );

    handle.abort();
}

#[tokio::test]
async fn test_api_search_rejects_invalid_sort_value() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/search?q=qwen&sort=random HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400"));
    let payload = json_body(&response);
    assert_eq!(
        payload["error"],
        json!(
            "Invalid 'sort' value 'random'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc"
        )
    );

    handle.abort();
}
