//! Authenticated, bounded forwarding for local provider requests.
use super::*;
use mesh_llm_plugin::operation_binding::{BoundOperationRequest, BoundOperationResponse};

pub(super) fn forward(
    caller: String,
    request_id: u64,
    request: proto::RpcRequest,
    peer: super::PluginRequestPeer,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    outbound: mpsc::Sender<proto::Envelope>,
    capacity: Arc<tokio::sync::Semaphore>,
) {
    let Ok(permit) = capacity.try_acquire_owned() else {
        // Reject synchronously rather than spawning an unbounded refusal task.
        let response = BoundOperationResponse::NotDispatched {
            reason: "caller capacity exhausted".into(),
        };
        let _ = outbound.try_send(proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: caller,
            request_id,
            payload: Some(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
                result_json: serde_json::to_string(&response).expect("serializable response"),
            })),
        });
        return;
    };
    tokio::spawn(async move {
        let _permit = permit;
        let response = dispatch(&caller, request, peer, mesh_tx).await;
        let payload = proto::envelope::Payload::RpcResponse(proto::RpcResponse {
            result_json: serde_json::to_string(&response).expect("serializable binding response"),
        });
        let _ = tokio::time::timeout(
            Duration::from_secs(1),
            outbound.send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: caller,
                request_id,
                payload: Some(payload),
            }),
        )
        .await;
    });
}

async fn dispatch(
    caller: &str,
    request: proto::RpcRequest,
    peer: super::PluginRequestPeer,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
) -> BoundOperationResponse {
    let denied = |reason: &str| BoundOperationResponse::NotDispatched {
        reason: reason.into(),
    };
    if !peer.authenticated
        || !peer
            .runtime
            .lock()
            .await
            .as_ref()
            .is_some_and(|runtime| runtime.generation == peer.generation)
    {
        return denied("authenticated current connection required");
    }
    if request.params_json.len() > 131_072 {
        return denied("binding request exceeds bound");
    }
    let Ok(request) = serde_json::from_str::<BoundOperationRequest>(&request.params_json) else {
        return denied("invalid binding request");
    };
    let (response_tx, response_rx) = oneshot::channel();
    // try_send bounds admission without occupying the connection reader.
    if mesh_tx
        .try_send(PluginMeshEvent::BoundOperation {
            plugin_id: caller.into(),
            generation: peer.generation,
            request,
            response_tx,
        })
        .is_err()
    {
        return denied("host queue unavailable");
    }
    match tokio::time::timeout(Duration::from_secs(32), response_rx).await {
        Ok(Ok(response)) => response,
        _ => BoundOperationResponse::Uncertain {
            reason: "host response unavailable".into(),
        },
    }
}
