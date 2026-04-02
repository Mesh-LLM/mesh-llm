//! Plugin mesh transport — floods plugin channel/bulk-transfer frames across all peers.
//!
//! These `impl Node` methods are called when the plugin subsystem sends a
//! `PluginMeshEvent` outbound, or when an incoming QUIC bi-stream is tagged with
//! `STREAM_PLUGIN_CHANNEL` / `STREAM_PLUGIN_BULK_TRANSFER`.

use super::*;

impl Node {
    pub(super) async fn forward_plugin_event(
        &self,
        event: crate::plugin::PluginMeshEvent,
    ) -> Result<()> {
        match event {
            crate::plugin::PluginMeshEvent::Channel {
                plugin_id,
                mut message,
            } => {
                if message.source_peer_id.is_empty() {
                    message.source_peer_id = endpoint_id_hex(self.endpoint.id());
                }
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
                if message.source_peer_id.is_empty() {
                    message.source_peer_id = endpoint_id_hex(self.endpoint.id());
                }
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
        }
    }

    async fn remember_plugin_message(&self, message_id: String) -> bool {
        const MAX_SEEN_PLUGIN_MESSAGES: usize = 4096;

        let mut state = self.state.lock().await;
        if !state.seen_plugin_messages.insert(message_id.clone()) {
            return false;
        }
        state.seen_plugin_message_order.push_back(message_id);
        while state.seen_plugin_message_order.len() > MAX_SEEN_PLUGIN_MESSAGES {
            if let Some(oldest) = state.seen_plugin_message_order.pop_front() {
                state.seen_plugin_messages.remove(&oldest);
            }
        }
        true
    }

    async fn broadcast_plugin_channel_frame(
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

    async fn broadcast_plugin_bulk_frame(
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

    pub(super) async fn handle_plugin_channel_stream(
        &self,
        remote: EndpointId,
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

        if message.target_peer_id != local_peer_id {
            self.broadcast_plugin_channel_frame(&frame, Some(remote))
                .await?;
        }

        Ok(())
    }

    pub(super) async fn handle_plugin_bulk_stream(
        &self,
        remote: EndpointId,
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

        if message.target_peer_id != local_peer_id {
            self.broadcast_plugin_bulk_frame(&frame, Some(remote))
                .await?;
        }

        Ok(())
    }
}
