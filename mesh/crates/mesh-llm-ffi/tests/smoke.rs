use meshllm_ffi::{FfiError, create_node};

fn node(mode: &str) -> std::sync::Arc<meshllm_ffi::MeshNodeHandle> {
    create_node(
        mode.to_string(),
        Vec::new(),
        Vec::new(),
        false,
        None,
        19337,
        13131,
    )
    .expect("create embedded node")
}

#[test]
fn every_embedded_role_is_configurable() {
    for role in ["client", "serve", "combined"] {
        let handle = node(role);
        let status = handle.status().expect("status before start");
        assert!(!status.running);
        assert_eq!(status.mode, role);
    }
}

#[test]
fn invalid_role_fails_before_start() {
    let result = create_node(
        "thin-client".to_string(),
        vec![],
        vec![],
        false,
        None,
        9337,
        3131,
    );
    assert!(matches!(result, Err(FfiError::BuildFailed(_))));
}

#[test]
fn inference_requires_running_node() {
    assert!(matches!(
        node("client").inference_list_models(),
        Err(FfiError::HostUnavailable(_))
    ));
}

#[test]
fn ffi_errors_are_std_errors() {
    fn assert_error<E: std::error::Error>() {}
    assert_error::<FfiError>();
}
