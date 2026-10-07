mod message_history;
pub(super) use message_history::PluginMessageHistory;

use super::*;
use crate::mesh::node::stamp_plugin_event_source;
use std::sync::atomic::{AtomicU64, Ordering};

const PLUGIN_FRAME_PREFIX_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const PLUGIN_FRAME_BODY_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const PLUGIN_CHANNEL_FRAME_MAX_BYTES: usize = 10_000_000;
const PLUGIN_BULK_FRAME_MAX_BYTES: usize = 64_000_000;

/// Shortest interval between plugin-frame warnings for one sending peer.
///
/// A remote can open a fresh stream per frame and distinct `message_id` values
/// bypass the dedup window, so warning once per frame is log I/O that peer
/// controls. The runtime writes each event to stderr and flushes it, so an
/// unthrottled warning is both unbounded output and a backpressure stall.
const PLUGIN_FRAME_WARN_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(60);

/// Upper bound on peers the plugin-frame warning throttle tracks. Reaching it
/// evicts the least recently warned peer, which costs that peer one extra
/// warning and never grows the map past this size.
const PLUGIN_FRAME_WARN_MAX_PEERS: usize = 1024;

/// `frame_kind` value for warnings raised on the channel-frame path.
const PLUGIN_FRAME_KIND_CHANNEL: &str = "channel";

/// `frame_kind` value for warnings raised on the bulk-transfer path.
const PLUGIN_FRAME_KIND_BULK: &str = "bulk";

/// Throttle state for plugin-frame warnings from one sending peer.
///
/// A remote can open a fresh stream for every frame, so one warning per frame
/// is log I/O that peer can drive without bound; the cooldown bounds it and
/// `suppressed` keeps the volume visible in the warning that is eventually
/// emitted.
pub(crate) struct PluginFrameWarnState {
    /// When this peer's last warning was emitted.
    pub(crate) last_warn_at: std::time::Instant,
    /// Warnings folded into this peer's next emitted warning.
    pub(crate) suppressed: u32,
}

/// Local-only plugin-frame telemetry: the mismatch counter and the per-peer
/// warning throttle that keeps its output bounded.
///
/// The throttle holds its own mutex rather than living in the mesh-wide
/// `MeshState`, so a plugin-frame decision never contends with — or holds — the
/// lock every other mesh path serializes behind, and the counter is a lock-free
/// atomic read for the status path.
#[derive(Default)]
pub(crate) struct PluginFrameTelemetry {
    /// Received plugin-mesh frames whose non-empty `source_peer_id` was not the
    /// sending peer. Advanced before throttling, so it is the true volume even
    /// while a peer's warning is folded.
    source_mismatch_total: AtomicU64,
    /// Per-peer warning throttle; deliberately its own mutex, not `MeshState`.
    warn: std::sync::Mutex<HashMap<EndpointId, PluginFrameWarnState>>,
}

impl PluginFrameTelemetry {
    /// Record one received frame whose claimed source is not the sending peer.
    ///
    /// The counter advances on every call; `now` decides the warning for
    /// `remote`: `Some` with the number of warnings folded in when the caller
    /// should emit one, or `None` while the peer is inside its cooldown.
    pub(crate) fn record_source_mismatch(
        &self,
        remote: EndpointId,
        now: std::time::Instant,
    ) -> Option<u32> {
        self.source_mismatch_total.fetch_add(1, Ordering::Relaxed);
        let mut warn = self
            .warn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        plugin_frame_warn_slot(&mut warn, remote, now)
    }

    /// Total plugin-frame source mismatches observed on this node.
    pub(crate) fn source_mismatch_total(&self) -> u64 {
        self.source_mismatch_total.load(Ordering::Relaxed)
    }
}

/// Throttle decision behind [`PluginFrameTelemetry::record_source_mismatch`],
/// over `now` so the cooldown is deterministic under test.
///
/// Returns the number of warnings folded into the one the caller should emit
/// now, or `None` while `remote` is still inside its cooldown.
pub(crate) fn plugin_frame_warn_slot(
    warn: &mut HashMap<EndpointId, PluginFrameWarnState>,
    remote: EndpointId,
    now: std::time::Instant,
) -> Option<u32> {
    if let Some(entry) = warn.get_mut(&remote) {
        if now.duration_since(entry.last_warn_at) < PLUGIN_FRAME_WARN_COOLDOWN {
            entry.suppressed = entry.suppressed.saturating_add(1);
            return None;
        }
        entry.last_warn_at = now;
        return Some(std::mem::take(&mut entry.suppressed));
    }

    // The least recently warned peer is the one to evict. Resolved through a
    // helper so the scrutinee holds no borrow of `warn`.
    if warn.len() >= PLUGIN_FRAME_WARN_MAX_PEERS
        && let Some(oldest) = least_recently_warned(warn)
    {
        warn.remove(&oldest);
    }
    warn.insert(
        remote,
        PluginFrameWarnState {
            last_warn_at: now,
            suppressed: 0,
        },
    );
    Some(0)
}

pub(crate) async fn read_plugin_frame_bytes<R>(reader: &mut R, max_len: usize) -> Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut len_buf = [0u8; 4];
    tokio::time::timeout(
        PLUGIN_FRAME_PREFIX_READ_TIMEOUT,
        tokio::io::AsyncReadExt::read_exact(reader, &mut len_buf),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "timeout reading plugin frame prefix after {PLUGIN_FRAME_PREFIX_READ_TIMEOUT:?}"
        )
    })?
    .context("read plugin frame prefix")?;
    let len = u32::from_le_bytes(len_buf) as usize;
    anyhow::ensure!(len <= max_len, "Plugin frame too large");
    let mut buf = vec![0u8; len];
    tokio::time::timeout(
        PLUGIN_FRAME_BODY_READ_TIMEOUT,
        tokio::io::AsyncReadExt::read_exact(reader, &mut buf),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "timeout reading plugin frame body after {PLUGIN_FRAME_BODY_READ_TIMEOUT:?}"
        )
    })?
    .context("read plugin frame body")?;
    Ok(buf)
}

impl Node {
    pub(crate) async fn forward_plugin_event(
        &self,
        event: crate::plugin::PluginMeshEvent,
    ) -> Result<()> {
        match event {
            crate::plugin::PluginMeshEvent::IdentityService {
                plugin_id,
                request,
                response_tx,
            } => {
                let node = self.clone();
                tokio::spawn(async move {
                    let response = crate::plugin::identity_services::handle_request(
                        &node, &plugin_id, request,
                    )
                    .await;
                    let _ = response_tx.send(response);
                });
                Ok(())
            }
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
                stamp_plugin_event_source(self.endpoint.id(), &mut message.source_peer_id);
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
                stamp_plugin_event_source(self.endpoint.id(), &mut message.source_peer_id);
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
            crate::plugin::PluginMeshEvent::PeerBlock {
                plugin_id,
                request,
                response_tx,
            } => {
                let response =
                    crate::network::peer_blocks::apply_plugin_request(self, plugin_id, request)
                        .await;
                let _ = response_tx.send(response);
                Ok(())
            }
            crate::plugin::PluginMeshEvent::PluginKey {
                plugin_id,
                request,
                response_tx,
            } => {
                let _ = response_tx.send(self.apply_plugin_key_request(&plugin_id, request));
                Ok(())
            }
        }
    }

    /// Announce (or withdraw) `plugin_id`'s public key, bound to this node.
    pub(crate) fn apply_plugin_key_request(
        &self,
        plugin_id: &str,
        request: crate::plugin::proto::PluginKeyRequest,
    ) -> Result<crate::plugin::proto::PluginKeyResponse, crate::plugin::proto::ErrorResponse> {
        use crate::mesh::plugin_keys;
        let invalid = |message: String| crate::plugin::proto::ErrorResponse {
            code: rmcp::model::ErrorCode::INVALID_PARAMS.0,
            message,
            data_json: String::new(),
        };
        let node_id = hex::encode(self.endpoint.id().as_bytes());
        if !plugin_keys::valid_plugin_name(plugin_id) {
            return Err(invalid(format!(
                "plugin `{plugin_id}` cannot announce a key: its name is not 1 to {} letters, digits, '-', '_' or '.'",
                plugin_keys::MAX_PLUGIN_NAME_BYTES
            )));
        }
        if request.public_key.is_empty() {
            self.plugin_keys.remove_own(plugin_id);
            return Ok(crate::plugin::proto::PluginKeyResponse {
                node_id,
                binding_signature: Vec::new(),
            });
        }
        let Ok(public_key) = <[u8; 32]>::try_from(request.public_key.as_slice()) else {
            return Err(invalid("public_key must be 32 bytes (Ed25519), or empty to withdraw".into()));
        };
        if !plugin_keys::valid_public_key(&public_key) {
            return Err(invalid("public_key is not a valid Ed25519 public key".into()));
        }
        let bound = plugin_keys::bind(&self.endpoint_secret_key, plugin_id, public_key);
        self.plugin_keys.set_own(bound.clone()).map_err(invalid)?;
        Ok(crate::plugin::proto::PluginKeyResponse {
            node_id,
            binding_signature: bound.binding_signature.to_vec(),
        })
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

    /// Record a plugin-frame source mismatch from `remote` and decide the warning.
    ///
    /// Both plugin-frame receive paths record the same class of event — a frame
    /// whose claimed `source_peer_id` is not the peer that sent it — and neither
    /// may warn per frame: a remote can open a stream per frame, and the runtime
    /// writes every warning to stderr and flushes it synchronously. Every call
    /// advances the local mismatch counter; the warning itself is throttled to at
    /// most one per [`PLUGIN_FRAME_WARN_COOLDOWN`] per sending peer, tracking at
    /// most [`PLUGIN_FRAME_WARN_MAX_PEERS`] peers. Neither the counter nor the
    /// throttle takes the mesh-wide [`MeshState`] lock.
    ///
    /// Returns the number of warnings folded into the one the caller should emit
    /// now, or `None` while the peer is still inside its cooldown (the caller
    /// stays silent and the occurrence is counted instead).
    pub(crate) fn plugin_frame_source_mismatch(&self, remote: EndpointId) -> Option<u32> {
        self.plugin_frame_telemetry
            .record_source_mismatch(remote, std::time::Instant::now())
    }

    /// Total plugin-frame source mismatches this node has observed.
    ///
    /// The per-peer warning is throttled, so this counter — not the log line —
    /// is the record of how often a peer claims a source that is not itself.
    /// Surfaced on the local status payload.
    pub(crate) fn plugin_frame_source_mismatch_total(&self) -> u64 {
        self.plugin_frame_telemetry.source_mismatch_total()
    }

    pub(crate) async fn remember_plugin_message(&self, message_id: String) -> bool {
        let now = std::time::Instant::now();
        self.plugin_message_history
            .lock()
            .await
            .remember(message_id, now)
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

    pub(crate) async fn forward_targeted_plugin_frame(
        &self,
        target_peer_id: &str,
        stream_type: u8,
        data: Vec<u8>,
    ) {
        let target = target_peer_id.to_string();
        let node = self.clone();
        tokio::spawn(async move {
            let result = async {
                let conn = node
                    .connection_for_peer_hex(&target)
                    .await
                    .with_context(|| format!("target peer {target} is not routable"))?;
                let (mut send, _recv) = conn.open_bi().await?;
                send.write_all(&[stream_type]).await?;
                send.write_all(&(data.len() as u32).to_le_bytes()).await?;
                send.write_all(&data).await?;
                send.finish()?;
                Ok::<_, anyhow::Error>(())
            }
            .await;
            if let Err(error) = result {
                tracing::debug!("Failed to forward targeted plugin frame: {error}");
            }
        });
    }

    /// Handle an inbound plugin-mesh frame carrying channel frames.
    ///
    /// `remote` is the peer the frame arrived from, which is not necessarily its
    /// origin: a relayed frame is re-encoded by each hop. The claimed
    /// `source_peer_id` is unauthenticated until plugin frame origin signing
    /// lands, so a mismatch with `remote` is recorded (throttled per peer) and
    /// the frame keeps its current handling.
    pub(crate) async fn handle_plugin_channel_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let buf = read_plugin_frame_bytes(&mut recv, PLUGIN_CHANNEL_FRAME_MAX_BYTES).await?;
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
        // `source_peer_id` arrives from the wire and no frame carries an origin
        // signature yet, so a value that is not the sending peer is an
        // unauthenticated claim. Record it; enforcement lands with plugin frame
        // origin signing. Warning is throttled per peer and reports how many
        // claims it folded in, so this cannot be turned into unbounded log I/O.
        if !message.source_peer_id.is_empty()
            && message.source_peer_id != endpoint_id_hex(remote)
            && let Some(suppressed) = self.plugin_frame_source_mismatch(remote)
        {
            tracing::warn!(
                claimed_source = %message.source_peer_id,
                sending_peer = %remote.fmt_short(),
                channel = %message.channel,
                frame_kind = %PLUGIN_FRAME_KIND_CHANNEL,
                suppressed,
                "Plugin frame claims a source_peer_id that is not the sending peer"
            );
        }
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

        // Targeted messages: forward only toward the specific target peer.
        // Do NOT flood-broadcast targeted messages to all connections — that
        // causes O(N²) amplification across the mesh.
        // Untargeted broadcasts: deliver locally only.  The originator already
        // sent to all their direct connections.
        if !message.target_peer_id.is_empty() && message.target_peer_id != local_peer_id {
            self.forward_targeted_plugin_frame(
                &message.target_peer_id,
                STREAM_PLUGIN_CHANNEL,
                frame.encode_to_vec(),
            )
            .await;
        }

        Ok(())
    }

    /// Handle an inbound plugin-mesh frame carrying bulk-transfer frames.
    ///
    /// `remote` is the peer the frame arrived from, which is not necessarily its
    /// origin: a relayed frame is re-encoded by each hop. The claimed
    /// `source_peer_id` is unauthenticated until plugin frame origin signing
    /// lands, so a mismatch with `remote` is recorded (throttled per peer) and
    /// the frame keeps its current handling.
    pub(crate) async fn handle_plugin_bulk_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let buf = read_plugin_frame_bytes(&mut recv, PLUGIN_BULK_FRAME_MAX_BYTES).await?;
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
        // `source_peer_id` arrives from the wire and no frame carries an origin
        // signature yet, so a value that is not the sending peer is an
        // unauthenticated claim. Record it; enforcement lands with plugin frame
        // origin signing. Warning is throttled per peer and reports how many
        // claims it folded in, so this cannot be turned into unbounded log I/O.
        if !message.source_peer_id.is_empty()
            && message.source_peer_id != endpoint_id_hex(remote)
            && let Some(suppressed) = self.plugin_frame_source_mismatch(remote)
        {
            tracing::warn!(
                claimed_source = %message.source_peer_id,
                sending_peer = %remote.fmt_short(),
                channel = %message.channel,
                frame_kind = %PLUGIN_FRAME_KIND_BULK,
                suppressed,
                "Plugin frame claims a source_peer_id that is not the sending peer"
            );
        }
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

        // Same policy as channel frames: targeted -> forward to target only,
        // broadcast → deliver locally only (originator already sent to their
        // direct connections).
        if !message.target_peer_id.is_empty() && message.target_peer_id != local_peer_id {
            self.forward_targeted_plugin_frame(
                &message.target_peer_id,
                STREAM_PLUGIN_BULK_TRANSFER,
                frame.encode_to_vec(),
            )
            .await;
        }

        Ok(())
    }
}

/// The peer this throttle would evict first: the one warned longest ago.
fn least_recently_warned(warn: &HashMap<EndpointId, PluginFrameWarnState>) -> Option<EndpointId> {
    warn.iter()
        .min_by_key(|(_, entry)| entry.last_warn_at)
        .map(|(peer, _)| *peer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(seed: u32) -> EndpointId {
        let mut bytes = [0u8; 32];
        bytes[..4].copy_from_slice(&seed.to_be_bytes());
        EndpointId::from(SecretKey::from_bytes(&bytes).public())
    }

    #[test]
    fn plugin_frame_warnings_are_throttled_per_peer() {
        let mut warn = HashMap::new();
        let remote = peer(1);
        let now = std::time::Instant::now();

        // The first claim warns; every claim inside the cooldown is folded into
        // the next warning instead of producing its own.
        assert_eq!(plugin_frame_warn_slot(&mut warn, remote, now), Some(0));
        assert_eq!(plugin_frame_warn_slot(&mut warn, remote, now), None);
        assert_eq!(plugin_frame_warn_slot(&mut warn, remote, now), None);
        assert_eq!(warn.get(&remote).expect("tracked peer").suppressed, 2);

        // Once the cooldown lapses the folded count is reported, not repeated.
        let later = now + PLUGIN_FRAME_WARN_COOLDOWN;
        assert_eq!(plugin_frame_warn_slot(&mut warn, remote, later), Some(2));
        assert_eq!(warn.get(&remote).expect("tracked peer").suppressed, 0);

        // Each peer has its own budget.
        assert_eq!(plugin_frame_warn_slot(&mut warn, peer(2), now), Some(0));
    }

    #[test]
    fn plugin_frame_warning_state_stays_bounded() {
        let mut warn = HashMap::new();
        let start = std::time::Instant::now();
        for seed in 0..PLUGIN_FRAME_WARN_MAX_PEERS as u32 {
            // Later peers age, so eviction has an unambiguous oldest entry.
            let at = start + std::time::Duration::from_millis(u64::from(seed));
            assert_eq!(plugin_frame_warn_slot(&mut warn, peer(seed), at), Some(0));
        }
        assert_eq!(warn.len(), PLUGIN_FRAME_WARN_MAX_PEERS);

        let newest = peer(PLUGIN_FRAME_WARN_MAX_PEERS as u32);
        let at = start + PLUGIN_FRAME_WARN_COOLDOWN;
        assert_eq!(plugin_frame_warn_slot(&mut warn, newest, at), Some(0));
        assert_eq!(warn.len(), PLUGIN_FRAME_WARN_MAX_PEERS);
        assert!(warn.contains_key(&newest));
        assert!(!warn.contains_key(&peer(0)));
    }

    #[test]
    fn plugin_frame_telemetry_counts_every_mismatch() {
        let telemetry = PluginFrameTelemetry::default();
        let remote = peer(7);
        let now = std::time::Instant::now();

        // The first mismatch warns and is counted.
        assert_eq!(telemetry.record_source_mismatch(remote, now), Some(0));
        assert_eq!(telemetry.source_mismatch_total(), 1);

        // Mismatches folded into the cooldown still advance the counter, so the
        // volume is visible even though the warning is not emitted.
        assert_eq!(telemetry.record_source_mismatch(remote, now), None);
        assert_eq!(telemetry.record_source_mismatch(remote, now), None);
        assert_eq!(telemetry.source_mismatch_total(), 3);

        // The counter is node-wide, not per peer.
        assert_eq!(telemetry.record_source_mismatch(peer(8), now), Some(0));
        assert_eq!(telemetry.source_mismatch_total(), 4);
    }
}
