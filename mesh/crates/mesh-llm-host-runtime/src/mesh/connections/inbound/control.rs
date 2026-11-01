//! Bounded control-plane stream dispatch after connection admission.

use super::super::*;

impl Node {
    /// Keep each stream's capacity permit until its control request finishes.
    pub(super) async fn dispatch_control_streams(&self, conn: Connection, remote: EndpointId) {
        let permits = control_stream_semaphore();
        loop {
            let permit_started = std::time::Instant::now();
            let permit = match permits.clone().acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => {
                    record_mesh_operational_event_with_context(
                        MeshOperationalEvent::ControlHandlerFailed(
                            MeshHandlerFailureBoundary::CapacityPermit.failure_class(),
                        ),
                        mesh_peer_operational_context(remote, selected_path_observation(&conn))
                            .duration_ms(elapsed_ms_u64(permit_started.elapsed())),
                    );
                    break;
                }
            };
            let stream_accept_started = std::time::Instant::now();
            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(streams) => streams,
                Err(error) => {
                    record_mesh_operational_event_with_context(
                        MeshOperationalEvent::ControlHandlerFailed(
                            MeshHandlerFailureBoundary::StreamAccept.failure_class(),
                        ),
                        mesh_peer_operational_context(remote, selected_path_observation(&conn))
                            .duration_ms(elapsed_ms_u64(stream_accept_started.elapsed())),
                    );
                    tracing::debug!(
                        "Control-plane connection from {} closed: {error}",
                        remote.fmt_short()
                    );
                    break;
                }
            };
            let node = self.clone();
            let peer_context =
                mesh_peer_operational_context(remote, selected_path_observation(&conn));
            tokio::spawn(Box::pin(async move {
                let _permit = permit;
                let dispatch_started = std::time::Instant::now();
                if let Err(error) = node
                    .handle_control_stream(remote, &mut send, &mut recv)
                    .await
                {
                    record_mesh_operational_event_with_context(
                        MeshOperationalEvent::ControlHandlerFailed(
                            MeshHandlerFailureBoundary::ControlDispatch.failure_class(),
                        ),
                        peer_context.duration_ms(elapsed_ms_u64(dispatch_started.elapsed())),
                    );
                    tracing::debug!(
                        "Control-plane stream from {} failed: {error}",
                        remote.fmt_short()
                    );
                }
            }));
        }
    }
}
