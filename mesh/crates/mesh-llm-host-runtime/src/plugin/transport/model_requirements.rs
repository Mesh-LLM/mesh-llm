//! Bounded forwarding for plugin-originated model requirements.
use super::*;

pub(super) fn forward_request(
    plugin_name: String,
    request_id: u64,
    request: super::proto::ModelRequirementRequest,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    outbound_tx: mpsc::Sender<super::proto::Envelope>,
) {
    tokio::spawn(async move {
        let (response_tx, response_rx) = oneshot::channel();
        let deadline = tokio::time::Instant::now() + PLUGIN_MESH_STREAM_RESPONSE_TIMEOUT;
        let response = match tokio::time::timeout_at(
            deadline,
            mesh_tx.send(PluginMeshEvent::ModelRequirement {
                plugin_id: plugin_name.clone(),
                request,
                response_tx,
            }),
        )
        .await
        {
            Ok(Ok(())) => match tokio::time::timeout_at(deadline, response_rx).await {
                Ok(Ok(response)) => response,
                Ok(Err(_)) => Err(plugin_mesh_stream_error(
                    "Model requirement handler dropped the response",
                )),
                Err(_) => Err(plugin_mesh_stream_error(
                    "Model requirement request timed out",
                )),
            },
            Ok(Err(_)) => Err(plugin_mesh_stream_error(
                "Model requirement handler is unavailable",
            )),
            Err(_) => Err(plugin_mesh_stream_error(
                "Model requirement request timed out",
            )),
        };
        let payload = match response {
            Ok(response) => super::proto::envelope::Payload::ModelRequirementResponse(response),
            Err(error) => super::proto::envelope::Payload::ErrorResponse(error),
        };
        let _ = outbound_tx
            .send(super::proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: plugin_name,
                request_id,
                payload: Some(payload),
            })
            .await;
    });
}
