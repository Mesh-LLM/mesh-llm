use super::*;
use crate::logging::OperationalAuditContext;

mod stage;

impl Node {
    pub(crate) async fn handle_incoming(&self, incoming: iroh::endpoint::Incoming) -> Result<()> {
        let started = std::time::Instant::now();
        let mut accepting = incoming.accept().inspect_err(|_| {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::QuicHandlerFailed(
                    MeshHandlerFailureBoundary::AcceptSetup.failure_class(),
                ),
                OperationalAuditContext::new().duration_ms(elapsed_ms_u64(started.elapsed())),
            );
        })?;
        let alpn = accepting.alpn().await.inspect_err(|_| {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::QuicHandlerFailed(
                    MeshHandlerFailureBoundary::AlpnRead.failure_class(),
                ),
                OperationalAuditContext::new().duration_ms(elapsed_ms_u64(started.elapsed())),
            );
        })?;
        let conn = accepting.await.inspect_err(|_| {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::QuicHandlerFailed(
                    MeshHandlerFailureBoundary::Handshake.failure_class(),
                ),
                OperationalAuditContext::new().duration_ms(elapsed_ms_u64(started.elapsed())),
            );
        })?;
        let remote = conn.remote_id();
        if self.handle_stage_alpn(&alpn, conn.clone(), remote).await {
            return Ok(());
        }
        tracing::info!("Inbound connection from {}", remote.fmt_short());

        // Store connection for stream dispatch (tunneling, route requests, etc.)
        // Don't add to peer list yet — only gossip exchange promotes to peer.
        let (was_dead, admitted) = self.remember_incoming_connection(remote, &conn).await;
        let inbound_outcome = if was_dead {
            MeshQuicInboundOutcome::Readmitted
        } else {
            MeshQuicInboundOutcome::Accepted
        };
        record_mesh_operational_event_with_context(
            MeshOperationalEvent::QuicInboundAccepted(inbound_outcome),
            mesh_peer_operational_context(remote, selected_path_observation(&conn))
                .numeric_summary("protocol_gen", u64::from(NODE_PROTOCOL_GENERATION)),
        );
        self.capture_connection_event(ConnectionCaptureEvent {
            event: "peer_connection_accepted",
            remote,
            direction: "inbound",
            phase: "accept",
            protocol: Some(connection_protocol(&conn)),
            path_type: None,
            rtt_ms: None,
            admitted_peer: Some(admitted),
            reason: was_dead.then_some("previously_dead"),
        });
        self.capture_selected_connection_path(remote, &conn, "inbound_connection_accept_path");

        // If this peer was previously dead, immediately gossip to restore their
        // assigned/routable state in our peer list. Without this, models served by the
        // reconnecting peer stay invisible until the next heartbeat (up to 60s).
        if was_dead {
            self.spawn_reconnect_gossip(conn.clone(), remote);
        }

        self.dispatch_streams(conn, remote).await;
        Ok(())
    }

    pub(crate) async fn handle_control_incoming(
        &self,
        incoming: iroh::endpoint::Incoming,
    ) -> Result<()> {
        let started = std::time::Instant::now();
        let mut accepting = incoming.accept().inspect_err(|_| {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::ControlHandlerFailed(
                    MeshHandlerFailureBoundary::AcceptSetup.failure_class(),
                ),
                OperationalAuditContext::new().duration_ms(elapsed_ms_u64(started.elapsed())),
            );
        })?;
        let alpn = accepting.alpn().await.inspect_err(|_| {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::ControlHandlerFailed(
                    MeshHandlerFailureBoundary::AlpnRead.failure_class(),
                ),
                OperationalAuditContext::new().duration_ms(elapsed_ms_u64(started.elapsed())),
            );
        })?;
        if alpn.as_slice() != ALPN_CONTROL_V1 {
            record_mesh_operational_event(MeshOperationalEvent::ControlAlpnRejected);
            anyhow::bail!("unsupported control-plane ALPN");
        }
        let conn = accepting.await.inspect_err(|_| {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::ControlHandlerFailed(
                    MeshHandlerFailureBoundary::Handshake.failure_class(),
                ),
                OperationalAuditContext::new().duration_ms(elapsed_ms_u64(started.elapsed())),
            );
        })?;
        let remote = conn.remote_id();
        record_mesh_operational_event_with_context(
            MeshOperationalEvent::ControlConnectionAccepted,
            mesh_peer_operational_context(remote, selected_path_observation(&conn))
                .numeric_summary("protocol_gen", u64::from(NODE_PROTOCOL_GENERATION)),
        );
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
        Ok(())
    }

    pub(crate) async fn accept_mesh_stream(
        &self,
        conn: &Connection,
        protocol: ControlProtocol,
    ) -> Result<AcceptedMeshStream, ()> {
        let remote = conn.remote_id();
        let (send, mut recv) = conn.accept_bi().await.map_err(|error| {
            tracing::info!("Connection to {} closed: {error}", remote.fmt_short());
            self.capture_connection_event(ConnectionCaptureEvent {
                event: "peer_connection_closed",
                remote,
                direction: "unknown",
                phase: "accept_bi",
                protocol: Some(protocol),
                path_type: None,
                rtt_ms: None,
                admitted_peer: None,
                reason: Some("accept_bi_error"),
            });
        })?;
        let mut type_buf = [0u8; 1];
        if !matches!(
            tokio::time::timeout(
                MESH_STREAM_TYPE_READ_TIMEOUT,
                recv.read_exact(&mut type_buf),
            )
            .await,
            Ok(Ok(_))
        ) {
            let _ = recv.stop(0u32.into());
            return Err(());
        }
        Ok(AcceptedMeshStream {
            remote,
            send,
            recv,
            stream_type: type_buf[0],
        })
    }

    pub(crate) async fn admitted_mesh_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        stream_type: u8,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) -> Option<MeshBiStream> {
        let capture_streams = self.swarm_capture_enabled();
        if stream_allowed_before_admission(stream_type, self.trust_policy) {
            if capture_streams {
                self.capture_stream_observation(remote, stream_type, protocol, true);
            }
            return Some((send, recv));
        }
        let admitted = {
            let state = self.state.lock().await;
            state.peers.get(&remote).is_some_and(PeerInfo::is_admitted)
        };
        if capture_streams {
            self.capture_stream_observation(remote, stream_type, protocol, admitted);
        }
        if admitted {
            Some((send, recv))
        } else {
            self.capture_stream_rejected(remote, stream_type, protocol, "unadmitted_peer");
            tracing::warn!(
                "Quarantine: stream {:#04x} from unadmitted peer {} rejected — peer must complete gossip first",
                stream_type,
                remote.fmt_short()
            );
            drop((send, recv));
            None
        }
    }
}
