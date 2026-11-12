#[tokio::test]
async fn nested_single_model_directive_bypasses_installed_override() {
    use crate::network::openai::virtual_model::InferenceRpcBridge;
    let (port, captured, upstream) = spawn_capturing_upstream(
        r#"{"id":"core","model":"concrete-worker","choices":[]}"#,
    )
    .await;
    let manager = start_standalone_virtual_model_plugin_manager(
        "mesh",
        json!({"id":"override", "model":"plugin-worker", "choices":[]}),
    )
    .await;
    let (address, proxy) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("concrete-worker", port)]),
        manager.clone(),
    )
    .await;
    // Neither client-supplied JSON nor a guessed request identity grants the
    // host-internal bypass. Ordinary ingress must still invoke the plugin.
    let body = json!({
        "model":"mesh", "single_model_only":true,
        "messages":[{"role":"user","content":"hello"}]
    }).to_string();
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nx-request-id: {}\r\nContent-Length: {}\r\n\r\n{body}",
        uuid::Uuid::new_v4(), body.len()
    );
    let ordinary = send_request_and_read_response(address, vec![wire.into_bytes()]).await;
    assert!(ordinary.starts_with("HTTP/1.1 200"), "{ordinary}");
    assert!(ordinary.contains("plugin-worker"), "{ordinary}");
    let bridge = InferenceRpcBridge::new(address.port(), manager);
    let legacy = json!({"model_id":"mesh", "request":{"messages":[]}});
    assert!(bridge.infer(&legacy.to_string()).await.is_err());
    let result = bridge
        .infer(&json!({
            "model_id":"mesh",
            "single_model_only":true,
            "request":{"messages":[{"role":"user","content":"hello"}]},
        }).to_string())
        .await
        .unwrap();
    let response: mesh_llm_plugin::HostInferenceResponse =
        serde_json::from_str(&result.result_json).unwrap();
    assert_eq!(response.status_code, 200);
    assert_eq!(response.body["model"], "concrete-worker");
    let raw = String::from_utf8(captured.await.unwrap()).unwrap();
    assert!(raw.contains("concrete-worker"), "{raw}");
    proxy.abort();
    upstream.await.unwrap();
}

#[tokio::test]
async fn nested_single_model_directive_skips_override_on_bootstrap_proxy() {
    use crate::network::openai::virtual_model::InferenceRpcBridge;
    let manager = start_standalone_virtual_model_plugin_manager(
        "mesh",
        json!({"id":"override", "model":"plugin-worker", "choices":[]}),
    )
    .await;
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.set_plugin_manager(manager.clone()).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    // The bootstrap proxy's accept loop: every connection goes straight to
    // `handle_mesh_request`, without the buffered ingress that consumes the
    // nested capability.
    let bootstrap = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            let node = node.clone();
            tokio::spawn(async move {
                Box::pin(crate::network::openai::transport::handle_mesh_request(
                    node,
                    stream.into(),
                    true,
                    affinity::AffinityRouter::default(),
                ))
                .await;
            });
        }
    });
    let bridge = InferenceRpcBridge::new(port, manager);
    let result = bridge
        .infer(
            &json!({
                "model_id":"mesh",
                "single_model_only":true,
                "request":{"messages":[{"role":"user","content":"hello"}]},
            })
            .to_string(),
        )
        .await;
    let rendered = match &result {
        Ok(rpc) => rpc.result_json.clone(),
        Err(error) => format!("{error:?}"),
    };
    assert!(!rendered.contains("plugin-worker"), "{rendered}");
    bootstrap.abort();
}
