use meshllm_ffi::{FfiError, create_node};
use std::net::TcpListener;

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
