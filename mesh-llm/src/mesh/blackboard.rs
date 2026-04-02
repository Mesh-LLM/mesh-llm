//! Blackboard mesh transport — replicates blackboard items to all peers.
//!
//! These `impl Node` methods handle `STREAM_BLACKBOARD` bi-streams: broadcast,
//! sync (digest-based reconciliation), and fetch-by-ids.

use super::*;

impl Node {
    pub async fn broadcast_blackboard(&self, item: &crate::blackboard::BlackboardItem) {
        if !self.blackboard.is_enabled() {
            return;
        }
        let msg = crate::blackboard::BlackboardMessage::Post(item.clone());
        let data = match serde_json::to_vec(&msg) {
            Ok(d) => d,
            Err(_) => return,
        };
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let bytes = data.clone();
            tokio::spawn(async move {
                let res = async {
                    let (mut send, _recv) = conn.open_bi().await?;
                    send.write_all(&[STREAM_BLACKBOARD]).await?;
                    send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                    send.write_all(&bytes).await?;
                    send.finish()?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                if let Err(e) = res {
                    tracing::debug!(
                        "Failed to broadcast blackboard to {}: {e}",
                        peer_id.fmt_short()
                    );
                }
            });
        }
    }

    /// Sync blackboard with a peer: exchange digests, fetch missing items.
    pub async fn sync_blackboard(&self, conn: Connection, remote: EndpointId) {
        if !self.blackboard.is_enabled() {
            return;
        }
        let res = async {
            let (mut send, mut recv) = conn.open_bi().await?;
            send.write_all(&[STREAM_BLACKBOARD]).await?;

            // Send SyncRequest
            let req = crate::blackboard::BlackboardMessage::SyncRequest;
            let req_data = serde_json::to_vec(&req)?;
            send.write_all(&(req_data.len() as u32).to_le_bytes())
                .await?;
            send.write_all(&req_data).await?;

            // Read their digest
            let mut len_buf = [0u8; 4];
            recv.read_exact(&mut len_buf).await?;
            let len = u32::from_le_bytes(len_buf) as usize;
            if len > 1_000_000 {
                anyhow::bail!("Blackboard sync response too large");
            }
            let mut buf = vec![0u8; len];
            recv.read_exact(&mut buf).await?;
            let their_msg: crate::blackboard::BlackboardMessage = serde_json::from_slice(&buf)?;

            if let crate::blackboard::BlackboardMessage::SyncDigest(their_ids) = their_msg {
                // Figure out what we're missing
                let our_ids = self.blackboard.ids().await;
                let missing: Vec<u64> = their_ids
                    .iter()
                    .filter(|id| !our_ids.contains(id))
                    .cloned()
                    .collect();

                if !missing.is_empty() {
                    // Request missing items
                    let fetch = crate::blackboard::BlackboardMessage::FetchRequest(missing);
                    let fetch_data = serde_json::to_vec(&fetch)?;
                    send.write_all(&(fetch_data.len() as u32).to_le_bytes())
                        .await?;
                    send.write_all(&fetch_data).await?;

                    // Read their response
                    let mut len_buf2 = [0u8; 4];
                    recv.read_exact(&mut len_buf2).await?;
                    let len2 = u32::from_le_bytes(len_buf2) as usize;
                    if len2 > 10_000_000 {
                        anyhow::bail!("Knowledge fetch response too large");
                    }
                    let mut buf2 = vec![0u8; len2];
                    recv.read_exact(&mut buf2).await?;
                    let items_msg: crate::blackboard::BlackboardMessage =
                        serde_json::from_slice(&buf2)?;

                    if let crate::blackboard::BlackboardMessage::FetchResponse(items) = items_msg {
                        let count = items.len();
                        for item in items {
                            self.blackboard.insert(item).await;
                        }
                        if count > 0 {
                            tracing::info!(
                                "Blackboard sync: got {} items from {}",
                                count,
                                remote.fmt_short()
                            );
                        }
                    }
                }
            }

            send.finish()?;
            Ok::<_, anyhow::Error>(())
        }
        .await;
        if let Err(e) = res {
            tracing::debug!("Blackboard sync with {} failed: {e}", remote.fmt_short());
        }
    }

    /// Handle an inbound blackboard stream from a peer.
    pub(super) async fn handle_blackboard_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        // Read the message
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 10_000_000 {
            anyhow::bail!("Knowledge message too large");
        }
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf).await?;
        let msg: crate::blackboard::BlackboardMessage = serde_json::from_slice(&buf)?;

        match msg {
            crate::blackboard::BlackboardMessage::Post(item) => {
                // Insert and re-broadcast if new
                let peer_name = item.from.clone();
                if self.blackboard.insert(item.clone()).await {
                    eprintln!(
                        "📝 Blackboard from {}: {}",
                        peer_name,
                        if item.text.len() > 80 {
                            format!("{}...", &item.text[..80])
                        } else {
                            item.text.clone()
                        }
                    );
                    // Forward to other peers (flood-fill)
                    let data =
                        serde_json::to_vec(&crate::blackboard::BlackboardMessage::Post(item))?;
                    let conns: Vec<(EndpointId, Connection)> = {
                        let state = self.state.lock().await;
                        state
                            .connections
                            .iter()
                            .filter(|(id, _)| **id != remote)
                            .map(|(id, c)| (*id, c.clone()))
                            .collect()
                    };
                    for (peer_id, conn) in conns {
                        let bytes = data.clone();
                        tokio::spawn(async move {
                            let res = async {
                                let (mut send, _recv) = conn.open_bi().await?;
                                send.write_all(&[STREAM_BLACKBOARD]).await?;
                                send.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
                                send.write_all(&bytes).await?;
                                send.finish()?;
                                Ok::<_, anyhow::Error>(())
                            }
                            .await;
                            if let Err(e) = res {
                                tracing::debug!(
                                    "Failed to forward blackboard to {}: {e}",
                                    peer_id.fmt_short()
                                );
                            }
                        });
                    }
                }
            }
            crate::blackboard::BlackboardMessage::SyncRequest => {
                // Send our digest
                let ids = self.blackboard.ids().await;
                let digest = crate::blackboard::BlackboardMessage::SyncDigest(ids);
                let data = serde_json::to_vec(&digest)?;
                send.write_all(&(data.len() as u32).to_le_bytes()).await?;
                send.write_all(&data).await?;

                // Check if they send a fetch request
                let mut len_buf2 = [0u8; 4];
                match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    recv.read_exact(&mut len_buf2),
                )
                .await
                {
                    Ok(Ok(())) => {
                        let len2 = u32::from_le_bytes(len_buf2) as usize;
                        if len2 > 1_000_000 {
                            anyhow::bail!("Fetch request too large");
                        }
                        let mut buf2 = vec![0u8; len2];
                        recv.read_exact(&mut buf2).await?;
                        let fetch_msg: crate::blackboard::BlackboardMessage =
                            serde_json::from_slice(&buf2)?;
                        if let crate::blackboard::BlackboardMessage::FetchRequest(wanted_ids) =
                            fetch_msg
                        {
                            let items = self.blackboard.get_by_ids(&wanted_ids).await;
                            let resp = crate::blackboard::BlackboardMessage::FetchResponse(items);
                            let resp_data = serde_json::to_vec(&resp)?;
                            send.write_all(&(resp_data.len() as u32).to_le_bytes())
                                .await?;
                            send.write_all(&resp_data).await?;
                        }
                    }
                    _ => {} // No fetch request, that's fine
                }
                send.finish()?;
            }
            _ => {} // Unexpected message type
        }

        Ok(())
    }
}
