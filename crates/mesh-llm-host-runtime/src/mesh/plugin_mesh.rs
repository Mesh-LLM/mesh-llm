use super::*;
use crate::mesh::node::default_plugin_event_source;

impl Node {
    pub(crate) async fn forward_plugin_event(
        &self,
        event: crate::plugin::PluginMeshEvent,
    ) -> Result<()> {
        match event {
            crate::plugin::PluginMeshEvent::Channel {
                plugin_id,
                mut message,
            } => {
                if !self
                    .plugin_event_channel_declared(&plugin_id, &message.channel, "message")
                    .await
                {
                    return Ok(());
                }
                default_plugin_event_source(self.endpoint.id(), &mut message.source_peer_id);
                let frame = crate::plugin::proto::MeshChannelFrame {
                    plugin_id,
                    message_id: new_plugin_message_id(&message.source_peer_id),
                    message: Some(message),
                };
                if !self.remember_plugin_message(frame.message_id.clone()).await {
                    return Ok(());
                }
                self.broadcast_plugin_channel_frame(&frame, None).await
            }
            crate::plugin::PluginMeshEvent::BulkTransfer {
                plugin_id,
                mut message,
            } => {
                if !self
                    .plugin_event_channel_declared(&plugin_id, &message.channel, "bulk transfer")
                    .await
                {
                    return Ok(());
                }
                default_plugin_event_source(self.endpoint.id(), &mut message.source_peer_id);
                let frame = crate::plugin::proto::MeshBulkFrame {
                    plugin_id,
                    message_id: new_plugin_message_id(&message.source_peer_id),
                    message: Some(message),
                };
                if !self.remember_plugin_message(frame.message_id.clone()).await {
                    return Ok(());
                }
                self.broadcast_plugin_bulk_frame(&frame, None).await
            }
            crate::plugin::PluginMeshEvent::OpenStream {
                plugin_id,
                request,
                response_tx,
            } => {
                let response = self
                    .open_outbound_plugin_mesh_stream(plugin_id, request)
                    .await;
                let _ = response_tx.send(response);
                Ok(())
            }
        }
    }

    pub(crate) async fn plugin_event_channel_declared(
        &self,
        plugin_id: &str,
        channel: &str,
        noun: &str,
    ) -> bool {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        if let Some(plugin_manager) = plugin_manager
            && !plugin_manager
                .plugin_declares_mesh_channel(plugin_id, channel)
                .await
        {
            tracing::debug!(
                plugin = %plugin_id,
                channel = %channel,
                "Dropping outbound {noun} for undeclared mesh channel"
            );
            return false;
        }
        true
    }

    pub(crate) async fn remember_plugin_message(&self, message_id: String) -> bool {
        /// How long to remember a message ID. Any duplicate arriving within
        /// this window is suppressed. This must be longer than the worst-case
        /// propagation delay across alternate mesh paths — 120s is generous.
        const DEDUP_TTL: std::time::Duration = std::time::Duration::from_secs(120);
        /// Hard cap to bound memory even if message volume is extreme.
        const DEDUP_HARD_CAP: usize = 100_000;

        let now = std::time::Instant::now();
        let mut state = self.state.lock().await;

        // Evict entries older than the TTL
        while let Some((ts, _)) = state.seen_plugin_message_order.front() {
            if now.duration_since(*ts) >= DEDUP_TTL {
                if let Some((_, id)) = state.seen_plugin_message_order.pop_front() {
                    state.seen_plugin_messages.remove(&id);
                }
            } else {
                break;
            }
        }

        // Already seen?
        if state.seen_plugin_messages.contains_key(&message_id) {
            return false;
        }

        // Hard cap: if under extreme load we still accumulate too many,
        // evict the oldest regardless of TTL.
        while state.seen_plugin_message_order.len() >= DEDUP_HARD_CAP {
            if let Some((_, id)) = state.seen_plugin_message_order.pop_front() {
                state.seen_plugin_messages.remove(&id);
            }
        }

        state.seen_plugin_messages.insert(message_id.clone(), now);
        state.seen_plugin_message_order.push_back((now, message_id));
        true
    }

    pub(crate) async fn broadcast_plugin_channel_frame(
        &self,
        frame: &crate::plugin::proto::MeshChannelFrame,
        skip_peer: Option<EndpointId>,
    ) -> Result<()> {
        let data = frame.encode_to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(peer_id, _)| Some(**peer_id) != skip_peer)
                .map(|(peer_id, conn)| (*peer_id, conn.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let result = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PLUGIN_CHANNEL]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = result {
                    tracing::debug!(
                        "Failed to broadcast plugin frame to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
        Ok(())
    }

    pub(crate) async fn broadcast_plugin_bulk_frame(
        &self,
        frame: &crate::plugin::proto::MeshBulkFrame,
        skip_peer: Option<EndpointId>,
    ) -> Result<()> {
        let data = frame.encode_to_vec();
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .filter(|(peer_id, _)| Some(**peer_id) != skip_peer)
                .map(|(peer_id, conn)| (*peer_id, conn.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let result = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_PLUGIN_BULK_TRANSFER]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = result {
                    tracing::debug!(
                        "Failed to broadcast plugin bulk frame to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
        Ok(())
    }

    pub(crate) async fn handle_plugin_channel_stream(
        &self,
        _remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 10_000_000 {
            anyhow::bail!("Plugin channel frame too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        send.finish()?;

        let frame = crate::plugin::proto::MeshChannelFrame::decode(buf.as_slice())?;
        if frame.plugin_id.is_empty() || frame.message_id.is_empty() {
            return Ok(());
        }
        if !self.remember_plugin_message(frame.message_id.clone()).await {
            return Ok(());
        }

        let Some(message) = frame.message.clone() else {
            return Ok(());
        };
        let local_peer_id = endpoint_id_hex(self.endpoint.id());
        let deliver_local =
            message.target_peer_id.is_empty() || message.target_peer_id == local_peer_id;

        if deliver_local {
            let plugin_manager = self.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                plugin_manager
                    .dispatch_channel_message(crate::plugin::PluginMeshEvent::Channel {
                        plugin_id: frame.plugin_id.clone(),
                        message: message.clone(),
                    })
                    .await?;
            }
        }

        // Targeted messages: forward only to the specific target peer if we
        // have a direct connection.  Do NOT flood-broadcast targeted messages
        // to all connections — that causes O(N²) amplification across the mesh.
        // Untargeted broadcasts: deliver locally only.  The originator already
        // sent to all their direct connections.
        if !message.target_peer_id.is_empty() && message.target_peer_id != local_peer_id {
            // Look up connection to the target peer by hex ID
            let target_conn = {
                let state = self.state.lock().await;
                state
                    .connections
                    .iter()
                    .find(|(id, _)| endpoint_id_hex(**id) == message.target_peer_id)
                    .map(|(id, conn)| (*id, conn.clone()))
            };
            if let Some((_target_id, conn)) = target_conn {
                let data = frame.encode_to_vec();
                tokio::spawn(async move {
                    let result = async {
                        let (mut send, _recv) = conn.open_bi().await?;
                        send.write_all(&[STREAM_PLUGIN_CHANNEL]).await?;
                        send.write_all(&(data.len() as u32).to_le_bytes()).await?;
                        send.write_all(&data).await?;
                        send.finish()?;
                        Ok::<_, anyhow::Error>(())
                    }
                    .await;
                    if let Err(e) = result {
                        tracing::debug!("Failed to forward targeted plugin frame: {e}");
                    }
                });
            }
        }

        Ok(())
    }

    pub(crate) async fn handle_plugin_bulk_stream(
        &self,
        _remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 64_000_000 {
            anyhow::bail!("Plugin bulk frame too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        send.finish()?;

        let frame = crate::plugin::proto::MeshBulkFrame::decode(buf.as_slice())?;
        if frame.plugin_id.is_empty() || frame.message_id.is_empty() {
            return Ok(());
        }
        if !self.remember_plugin_message(frame.message_id.clone()).await {
            return Ok(());
        }

        let Some(message) = frame.message.clone() else {
            return Ok(());
        };
        let local_peer_id = endpoint_id_hex(self.endpoint.id());
        let deliver_local =
            message.target_peer_id.is_empty() || message.target_peer_id == local_peer_id;

        if deliver_local {
            let plugin_manager = self.plugin_manager.lock().await.clone();
            if let Some(plugin_manager) = plugin_manager {
                plugin_manager
                    .dispatch_bulk_transfer_message(crate::plugin::PluginMeshEvent::BulkTransfer {
                        plugin_id: frame.plugin_id.clone(),
                        message: message.clone(),
                    })
                    .await?;
            }
        }

        // Same policy as channel frames: targeted → forward to target only,
        // broadcast → deliver locally only (originator already sent to their
        // direct connections).
        if !message.target_peer_id.is_empty() && message.target_peer_id != local_peer_id {
            let target_conn = {
                let state = self.state.lock().await;
                state
                    .connections
                    .iter()
                    .find(|(id, _)| endpoint_id_hex(**id) == message.target_peer_id)
                    .map(|(id, conn)| (*id, conn.clone()))
            };
            if let Some((_target_id, conn)) = target_conn {
                let data = frame.encode_to_vec();
                tokio::spawn(async move {
                    let result = async {
                        let (mut send, _recv) = conn.open_bi().await?;
                        send.write_all(&[STREAM_PLUGIN_BULK_TRANSFER]).await?;
                        send.write_all(&(data.len() as u32).to_le_bytes()).await?;
                        send.write_all(&data).await?;
                        send.finish()?;
                        Ok::<_, anyhow::Error>(())
                    }
                    .await;
                    if let Err(e) = result {
                        tracing::debug!("Failed to forward targeted plugin bulk frame: {e}");
                    }
                });
            }
        }

        Ok(())
    }
}
