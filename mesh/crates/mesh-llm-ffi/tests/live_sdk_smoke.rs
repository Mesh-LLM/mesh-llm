use meshllm_ffi::create_node;

#[test]
fn embedded_client_runs_against_live_mesh_when_configured() {
    let Ok(invite_token) = std::env::var("MESH_SDK_INVITE_TOKEN") else {
        return;
    };
    let handle = create_node(
        "client".to_string(),
        vec![invite_token],
        vec![],
        false,
        None,
        19337,
        13131,
    )
    .expect("create embedded client");
    handle.start().expect("start embedded client");
    let status = handle.status().expect("status");
    assert!(status.running);
    assert_eq!(status.mode, "client");
    let models = handle.inference_list_models().expect("models");
    assert!(!models.is_empty());
    handle.stop().expect("stop embedded client");
}
