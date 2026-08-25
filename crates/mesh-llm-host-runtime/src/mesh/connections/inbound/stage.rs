use super::super::*;

impl Node {
    pub(crate) async fn handle_stage_alpn(
        &self,
        alpn: &[u8],
        conn: Connection,
        remote: EndpointId,
    ) -> bool {
        if alpn != skippy_protocol::STAGE_ALPN_V2 {
            return false;
        }
        tracing::info!(
            "Inbound skippy stage connection from {}",
            remote.fmt_short()
        );
        record_mesh_operational_event_with_context(
            MeshOperationalEvent::QuicInboundAccepted(MeshQuicInboundOutcome::Accepted),
            mesh_peer_operational_context(remote, selected_path_observation(&conn))
                .numeric_summary(
                    "protocol_gen",
                    u64::from(skippy_protocol::STAGE_PROTOCOL_GENERATION),
                ),
        );
        self.dispatch_stage_streams(conn, remote).await;
        true
    }
}
