#[test]
fn test_build_runtime_status_payload_uses_local_processes() {
    let result = build_runtime_status_payload(
        "Qwen",
        Some("llama".into()),
        None,
        true,
        true,
        Some(9337),
        vec![
            RuntimeProcessPayload {
                name: "Qwen".into(),
                instance_id: None,
                backend: "llama".into(),
                status: "ready".into(),
                port: 9337,
                pid: 100,
                slots: 4,
                context_length: None,
                profile: String::new(),
            },
            RuntimeProcessPayload {
                name: "Llama".into(),
                instance_id: None,
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 101,
                slots: 4,
                context_length: None,
                profile: String::new(),
            },
        ],
    );
    assert_eq!(result.models.len(), 2);
    assert_eq!(result.models[0].name, "Llama");
    assert_eq!(result.models[0].port, Some(9444));
    assert_eq!(result.models[1].name, "Qwen");
}
#[test]
fn test_build_runtime_status_payload_keeps_duplicate_model_instances() {
    let result = build_runtime_status_payload(
        "Qwen",
        Some("skippy".into()),
        None,
        true,
        true,
        Some(9337),
        vec![
            RuntimeProcessPayload {
                name: "Qwen".into(),
                instance_id: Some("runtime-1".into()),
                backend: "skippy".into(),
                status: "ready".into(),
                port: 41001,
                pid: 100,
                slots: 4,
                context_length: Some(8192),
                profile: String::new(),
            },
            RuntimeProcessPayload {
                name: "Qwen".into(),
                instance_id: Some("runtime-2".into()),
                backend: "skippy".into(),
                status: "ready".into(),
                port: 41002,
                pid: 100,
                slots: 4,
                context_length: Some(8192),
                profile: String::new(),
            },
        ],
    );

    assert_eq!(result.models.len(), 2);
    assert_eq!(result.models[0].name, "Qwen");
    assert_eq!(result.models[0].instance_id.as_deref(), Some("runtime-1"));
    assert_eq!(result.models[0].port, Some(41001));
    assert_eq!(result.models[1].name, "Qwen");
    assert_eq!(result.models[1].instance_id.as_deref(), Some("runtime-2"));
    assert_eq!(result.models[1].port, Some(41002));
}

#[test]
fn test_build_runtime_processes_payload_sorts_processes() {
    let payload = build_runtime_processes_payload(vec![
        RuntimeProcessPayload {
            name: "Zulu".into(),
            instance_id: None,
            backend: "llama".into(),
            status: "ready".into(),
            port: 9444,
            pid: 11,
            slots: 4,
            context_length: None,
            profile: String::new(),
        },
        RuntimeProcessPayload {
            name: "Alpha".into(),
            instance_id: None,
            backend: "llama".into(),
            status: "ready".into(),
            port: 9337,
            pid: 10,
            slots: 4,
            context_length: None,
            profile: String::new(),
        },
    ]);

    assert_eq!(payload.processes.len(), 2);
    assert_eq!(payload.processes[0].name, "Alpha");
    assert_eq!(payload.processes[1].name, "Zulu");
}

#[test]
fn test_runtime_processes_payload_includes_context_length() {
    let payload = build_runtime_processes_payload(vec![
        RuntimeProcessPayload {
            name: "model-a".into(),
            instance_id: None,
            backend: "llama".into(),
            status: "ready".into(),
            port: 9337,
            pid: 10,
            slots: 4,
            context_length: Some(65536),
            profile: String::new(),
        },
        RuntimeProcessPayload {
            name: "model-b".into(),
            instance_id: None,
            backend: "llama".into(),
            status: "ready".into(),
            port: 9444,
            pid: 11,
            slots: 2,
            context_length: None,
            profile: String::new(),
        },
    ]);

    assert_eq!(payload.processes.len(), 2);
    assert_eq!(payload.processes[0].name, "model-a");
    assert_eq!(payload.processes[0].context_length, Some(65536));
    assert_eq!(payload.processes[0].slots, 4);
    assert_eq!(payload.processes[1].context_length, None);

    // Verify serialization includes context_length when present
    let json = serde_json::to_string(&payload).expect("serialize payload");
    assert!(json.contains(r#""context_length":65536"#));
    // Verify context_length is omitted when None (skip_serializing_if)
    let model_b_section: serde_json::Value = serde_json::from_str(&json).expect("parse json");
    let processes = model_b_section["processes"]
        .as_array()
        .expect("processes array");
    assert!(
        processes[1].get("context_length").is_none() && processes[1]["context_length"].is_null()
    );
}

#[test]
fn test_classify_runtime_error_codes() {
    assert_eq!(classify_runtime_error("model 'x' is not loaded"), 404);
    assert_eq!(classify_runtime_error("model 'x' is already loaded"), 409);
    assert_eq!(
        classify_runtime_error("runtime load only supports models that fit locally"),
        422
    );
    assert_eq!(
        classify_runtime_error("runtime capacity for model 'x' exceeds node pool"),
        422
    );
    assert_eq!(classify_runtime_error("bad request"), 400);
}
