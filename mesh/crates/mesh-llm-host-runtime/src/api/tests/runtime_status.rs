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
