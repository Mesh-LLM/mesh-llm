use meshllm_ffi::{FfiError, create_node};
use std::net::TcpListener;
use std::time::Duration;

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind a free port")
        .local_addr()
        .expect("port")
        .port()
}

#[test]
#[ignore = "starts three embedded nodes and binds local sockets"]
fn each_embedded_role_starts_and_enforces_inference_access() {
    for role in ["client", "serve", "combined"] {
        let owner_key =
            std::env::temp_dir().join(format!("mesh-sdk-{role}-{}.json", uuid::Uuid::new_v4()));
        let handle = create_node(
            role.to_string(),
            vec![],
            vec![],
            false,
            Some(owner_key.to_string_lossy().into_owned()),
            free_port(),
            free_port(),
        )
        .expect("create node");
        handle
            .start()
            .unwrap_or_else(|error| panic!("{role} startup: {error}"));
        let status = handle.status().expect("running status");
        assert!(status.running);
        assert_eq!(status.mode, role);
        assert!(status.api_base_url.ends_with("/v1"));

        if role == "serve" {
            assert!(matches!(
                handle.inference_list_models(),
                Err(FfiError::ServingUnsupported(_))
            ));
        } else {
            handle
                .inference_list_models()
                .unwrap_or_else(|error| panic!("{role} model listing: {error}"));
        }

        handle.stop().expect("stop embedded node");
        let _ = std::fs::remove_file(owner_key);
    }
}

#[test]
#[ignore = "requires MESH_SDK_TEST_GGUF and a compatible installed native runtime"]
fn local_model_serves_client_and_combined_inference() {
    let Ok(model_path) = std::env::var("MESH_SDK_TEST_GGUF") else {
        return;
    };
    let key_dir = std::env::temp_dir().join(format!("mesh-sdk-roles-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&key_dir).expect("create role test directory");
    let make_node = |mode: &str, models: Vec<String>, join_tokens: Vec<String>| {
        let api_port = free_port();
        let mut console_port = free_port();
        while console_port == api_port {
            console_port = free_port();
        }
        create_node(
            mode.to_string(),
            join_tokens,
            models,
            false,
            Some(
                key_dir
                    .join(format!("{mode}.json"))
                    .to_string_lossy()
                    .into_owned(),
            ),
            api_port,
            console_port,
        )
        .expect("create embedded node")
    };

    let server = make_node("serve", vec![model_path.clone()], vec![]);
    server.start().expect("start serving node");
    assert!(matches!(
        server.inference_list_models(),
        Err(FfiError::ServingUnsupported(_))
    ));
    let server_status = server.status().expect("serving status");
    let server_payload: serde_json::Value =
        serde_json::from_str(&server_status.payload_json).expect("status JSON");
    let token = server_payload["token"]
        .as_str()
        .expect("serving node invite token")
        .to_string();
    let model_id = wait_for_model(&server_status.api_base_url);

    let client = make_node("client", vec![], vec![token.clone()]);
    client.start().expect("start client node");
    wait_for_client_model(&client, &model_id);
    assert_chat(&client, &model_id);

    let combined = make_node("combined", vec![model_path], vec![token]);
    combined.start().expect("start combined node");
    wait_for_client_model(&combined, &model_id);
    assert_chat(&combined, &model_id);

    combined.stop().expect("stop combined node");
    client.stop().expect("stop client node");
    server.stop().expect("stop serving node");
    std::fs::remove_dir_all(key_dir).expect("remove role test directory");
}

fn wait_for_model(base_url: &str) -> String {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("HTTP runtime");
    for _ in 0..120 {
        let models = runtime.block_on(async {
            reqwest::get(format!("{base_url}/models"))
                .await
                .ok()?
                .json::<serde_json::Value>()
                .await
                .ok()
        });
        if let Some(id) = models
            .as_ref()
            .and_then(|value| value["data"].as_array())
            .and_then(|models| models.first())
            .and_then(|model| model["id"].as_str())
        {
            return id.to_string();
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    panic!("serving node did not advertise a model within two minutes");
}

fn wait_for_client_model(node: &meshllm_ffi::MeshNodeHandle, model_id: &str) {
    for _ in 0..60 {
        let models = node
            .inference_list_models()
            .expect("list models through embedded node");
        if models.iter().any(|model| model.id == model_id) {
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    panic!("embedded node did not discover model {model_id} within one minute");
}

fn assert_chat(node: &meshllm_ffi::MeshNodeHandle, model_id: &str) {
    let body = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 8,
    });
    let response = node
        .openai_request("/v1/chat/completions".to_string(), body.to_string())
        .expect("send embedded inference request");
    assert_eq!(response.status_code, 200, "{}", response.body);
    let payload: serde_json::Value = serde_json::from_str(&response.body).expect("response JSON");
    assert!(
        payload["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|content| !content.is_empty()),
        "{}",
        response.body
    );
}
