//! Authenticated identity RPC dispatch over the existing generation-3 connection.

use super::{PluginMeshEvent, proto};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

fn plugin_mesh_stream_error(message: impl Into<String>) -> proto::ErrorResponse {
    proto::ErrorResponse {
        code: rmcp::model::ErrorCode::INTERNAL_ERROR.0,
        message: message.into(),
        data_json: String::new(),
    }
}

const PLUGIN_MESH_STREAM_RESPONSE_TIMEOUT: Duration =
    Duration::from_secs(super::REQUEST_TIMEOUT_SECS);

pub(super) async fn forward_identity_service(
    plugin_name: &str,
    request: super::proto::RpcRequest,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
) -> super::proto::envelope::Payload {
    let (response_tx, response_rx) = oneshot::channel();
    let result = tokio::time::timeout(PLUGIN_MESH_STREAM_RESPONSE_TIMEOUT, async {
        mesh_tx
            .send(PluginMeshEvent::IdentityService {
                plugin_id: plugin_name.to_owned(),
                request,
                response_tx,
            })
            .await
            .map_err(|_| plugin_mesh_stream_error("Identity service unavailable"))?;
        response_rx
            .await
            .map_err(|_| plugin_mesh_stream_error("Identity service response dropped"))?
    })
    .await
    .unwrap_or_else(|_| {
        Err(plugin_mesh_stream_error(
            "Identity service deadline exceeded",
        ))
    });
    match result {
        Ok(response) => super::proto::envelope::Payload::RpcResponse(response),
        Err(error) => super::proto::envelope::Payload::ErrorResponse(error),
    }
}

#[cfg(test)]
mod identity_transport_tests {
    use super::*;

    #[tokio::test]
    async fn identity_services_use_authenticated_connection_without_mcp_bridge() {
        let (mesh_tx, mut mesh_rx) = mpsc::channel(1);
        let broker = tokio::spawn(async move {
            let Some(PluginMeshEvent::IdentityService {
                plugin_id,
                request,
                response_tx,
            }) = mesh_rx.recv().await
            else {
                panic!("expected identity service event");
            };
            assert_eq!(plugin_id, "authenticated-observer");
            assert_eq!(request.method, "ReadIdentityBundle");
            response_tx
                .send(Ok(super::super::proto::RpcResponse {
                    result_json: "{}".into(),
                }))
                .unwrap();
        });
        let payload = forward_identity_service(
            "authenticated-observer",
            super::super::proto::RpcRequest {
                method: "ReadIdentityBundle".into(),
                params_json: "{}".into(),
            },
            mesh_tx,
        )
        .await;
        assert!(matches!(
            payload,
            super::super::proto::envelope::Payload::RpcResponse(_)
        ));
        broker.await.unwrap();
    }
}
