//! Private identity RPCs never reach the node from an unverified connection.

use super::*;

#[tokio::test]
async fn unverified_identity_requests_are_rejected_without_forwarding_to_mesh() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(4);
    let (outbound, mut replies) = mpsc::channel(4);
    for (method, authenticated) in [
        ("ReadIdentityBundle", false),
        ("DelegatePluginSigningKey", false),
        ("ReadIdentityBundle", true),
        ("DelegatePluginSigningKey", true),
    ] {
        forward_plugin_request(
            "impersonated-installed-plugin".into(),
            123,
            proto::RpcRequest {
                method: method.into(),
                params_json: "{}".into(),
            },
            mesh_tx.clone(),
            Arc::new(Mutex::new(None)),
            outbound.clone(),
            PluginRequestPeer {
                authenticated,
                runtime: Arc::new(Mutex::new(None)),
                generation: 1,
            },
        );
        let reply = tokio::time::timeout(Duration::from_secs(1), replies.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reply.request_id, 123);
        let Some(proto::envelope::Payload::ErrorResponse(error)) = reply.payload else {
            panic!("unverified peer received an identity response");
        };
        assert!(
            error
                .message
                .contains("OS-authenticated launched plugin process")
        );
        assert!(matches!(
            mesh_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}
