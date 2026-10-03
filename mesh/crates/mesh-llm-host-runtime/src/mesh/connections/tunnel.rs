use super::*;

impl Node {
    /// Dispatch bi-streams on a connection by type byte
    pub(crate) fn dispatch_streams(
        &self,
        conn: Connection,
        remote: EndpointId,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(self._dispatch_streams(conn, remote))
    }

    pub(crate) async fn dispatch_mesh_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        stream_type: u8,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) -> bool {
        if stream_type == STREAM_TUNNEL {
            return self.forward_tunnel_stream(send, recv).await;
        }
        if stream_type == STREAM_TUNNEL_HTTP {
            return self.forward_tunnel_http_stream(remote, send, recv).await;
        }

        self.spawn_non_tunnel_mesh_stream(remote, protocol, stream_type, send, recv);
        true
    }

    pub(crate) async fn forward_tunnel_stream(
        &self,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) -> bool {
        if self.tunnel_tx.send((send, recv)).await.is_err() {
            tracing::warn!("Tunnel receiver dropped");
            return false;
        }
        true
    }

    pub(crate) async fn forward_tunnel_http_stream(
        &self,
        remote: EndpointId,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) -> bool {
        if self
            .tunnel_http_tx
            .send((remote, send, recv))
            .await
            .is_err()
        {
            tracing::warn!("HTTP tunnel receiver dropped");
            return false;
        }
        true
    }

    pub(crate) async fn _dispatch_streams(&self, conn: Connection, remote: EndpointId) {
        let protocol = connection_protocol(&conn);
        let dispatcher_stable_id = conn.stable_id();
        loop {
            let accepted = match self.accept_mesh_stream(&conn, protocol).await {
                Ok(accepted) => accepted,
                Err(()) => {
                    self.recover_closed_connection(remote, dispatcher_stable_id)
                        .await;
                    break;
                }
            };
            let Some((send, recv)) = self
                .admitted_mesh_stream(
                    accepted.remote,
                    protocol,
                    accepted.stream_type,
                    accepted.send,
                    accepted.recv,
                )
                .await
            else {
                continue;
            };
            if !self
                .dispatch_mesh_stream(accepted.remote, protocol, accepted.stream_type, send, recv)
                .await
            {
                break;
            }
        }
    }

    pub(crate) async fn authenticated_peer_path(
        &self,
        remote: EndpointId,
    ) -> Option<SelectedPathObservation> {
        let conn = self.state.lock().await.connections.get(&remote).cloned()?;
        (conn.remote_id() == remote)
            .then(|| selected_path_observation(&conn))
            .flatten()
    }

    pub(crate) async fn remove_connection_if_stable_id(
        &self,
        peer_id: EndpointId,
        conn: &Connection,
    ) -> Option<Connection> {
        let stable_id = conn.stable_id();
        let mut state = self.state.lock().await;
        if state
            .connections
            .get(&peer_id)
            .is_some_and(|current| current.stable_id() == stable_id)
        {
            state.connections.remove(&peer_id)
        } else {
            None
        }
    }
}
