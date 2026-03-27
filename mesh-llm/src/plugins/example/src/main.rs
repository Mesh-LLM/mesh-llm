use anyhow::Result;
use mesh_llm_plugin::{
    Plugin, PluginContext, PluginResult, PluginRuntime, ToolCallRequest, async_trait,
    bulk_transfer_message, channel_message, empty_object_schema, json_bytes, json_schema_tool,
    json_string, list_tools, parse_optional_json, plugin_server_info, proto,
    structured_tool_result, tool_error, tool_with_schema,
};
use rmcp::model::{CallToolResult, ListToolsResult, ServerInfo};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::Mutex;

const PLUGIN_ID: &str = "example";
const EXAMPLE_CHANNEL: &str = "example.v1";
const DEFAULT_BULK_CHUNK_SIZE: usize = 16 * 1024;
const DEFAULT_SNAPSHOT_LIMIT: usize = 20;
const MAX_RECENT_ITEMS: usize = 128;

#[derive(Debug, Deserialize, Default)]
struct SnapshotParams {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SendMessageArguments {
    text: String,
    #[serde(default)]
    target_peer_id: Option<String>,
    #[serde(default)]
    request_ack: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SendBulkArguments {
    text: String,
    #[serde(default)]
    target_peer_id: Option<String>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    request_ack: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct ClearParams {}

#[derive(Clone, Debug, Serialize)]
struct PeerSummary {
    peer_id: String,
    role: String,
    version: String,
    capabilities: Vec<String>,
    models: Vec<String>,
    serving_models: Vec<String>,
    rtt_ms: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedMeshEvent {
    kind: String,
    peer_id: Option<String>,
    local_peer_id: String,
    mesh_id: String,
    detail_json: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedChannelMessage {
    direction: String,
    channel: String,
    source_peer_id: String,
    target_peer_id: String,
    message_kind: String,
    correlation_id: String,
    content_type: String,
    metadata_json: Option<serde_json::Value>,
    text_preview: String,
    body_len: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedBulkEvent {
    direction: String,
    kind: String,
    transfer_id: String,
    source_peer_id: String,
    target_peer_id: String,
    correlation_id: String,
    content_type: String,
    metadata_json: Option<serde_json::Value>,
    total_bytes: u64,
    offset: u64,
    body_len: usize,
    final_chunk: bool,
}

#[derive(Clone, Debug, Serialize)]
struct CompletedTransfer {
    transfer_id: String,
    source_peer_id: String,
    target_peer_id: String,
    content_type: String,
    total_bytes: u64,
    received_bytes: u64,
    preview: String,
}

#[derive(Default)]
struct TransferAccumulator {
    source_peer_id: String,
    target_peer_id: String,
    content_type: String,
    total_bytes: u64,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct ExampleState {
    local_peer_id: String,
    mesh_id: String,
    known_peers: BTreeMap<String, PeerSummary>,
    mesh_events: Vec<RecordedMeshEvent>,
    channel_messages: Vec<RecordedChannelMessage>,
    bulk_events: Vec<RecordedBulkEvent>,
    completed_transfers: BTreeMap<String, CompletedTransfer>,
    transfer_state: HashMap<String, TransferAccumulator>,
    sent_channel_messages: usize,
    sent_bulk_transfers: usize,
    next_id: u64,
}

impl ExampleState {
    fn next_token(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("{prefix}-{}-{}", now_millis(), self.next_id)
    }

    fn snapshot(&self, limit: usize) -> serde_json::Value {
        let limit = limit.max(1);
        let completed_transfers = self
            .completed_transfers
            .values()
            .cloned()
            .collect::<Vec<_>>();
        json!({
            "plugin": PLUGIN_ID,
            "channel": EXAMPLE_CHANNEL,
            "local_peer_id": self.local_peer_id,
            "mesh_id": self.mesh_id,
            "known_peers": self.known_peers.values().cloned().collect::<Vec<_>>(),
            "stats": {
                "known_peer_count": self.known_peers.len(),
                "mesh_event_count": self.mesh_events.len(),
                "channel_message_count": self.channel_messages.len(),
                "bulk_event_count": self.bulk_events.len(),
                "completed_transfer_count": self.completed_transfers.len(),
                "sent_channel_messages": self.sent_channel_messages,
                "sent_bulk_transfers": self.sent_bulk_transfers,
            },
            "recent_mesh_events": recent_items(&self.mesh_events, limit),
            "recent_channel_messages": recent_items(&self.channel_messages, limit),
            "recent_bulk_events": recent_items(&self.bulk_events, limit),
            "completed_transfers": recent_items(&completed_transfers, limit),
        })
    }

    fn clear_history(&mut self) {
        self.mesh_events.clear();
        self.channel_messages.clear();
        self.bulk_events.clear();
        self.completed_transfers.clear();
        self.transfer_state.clear();
        self.sent_channel_messages = 0;
        self.sent_bulk_transfers = 0;
    }

    fn record_mesh_event(&mut self, event: &proto::MeshEvent) {
        if !event.local_peer_id.is_empty() {
            self.local_peer_id = event.local_peer_id.clone();
        }
        if !event.mesh_id.is_empty() {
            self.mesh_id = event.mesh_id.clone();
        }
        if let Some(peer) = &event.peer {
            let peer_id = peer.peer_id.clone();
            match proto::mesh_event::Kind::try_from(event.kind).ok() {
                Some(proto::mesh_event::Kind::PeerDown) => {
                    self.known_peers.remove(&peer_id);
                }
                _ => {
                    self.known_peers.insert(peer_id, peer_summary(peer));
                }
            }
        }

        push_bounded(
            &mut self.mesh_events,
            RecordedMeshEvent {
                kind: mesh_event_kind_name(event.kind).into(),
                peer_id: event.peer.as_ref().map(|peer| peer.peer_id.clone()),
                local_peer_id: event.local_peer_id.clone(),
                mesh_id: event.mesh_id.clone(),
                detail_json: parse_optional_json(&event.detail_json),
            },
        );
    }

    fn record_channel_message(&mut self, direction: &str, message: &proto::ChannelMessage) {
        if direction == "outbound" {
            self.sent_channel_messages += 1;
        }
        push_bounded(
            &mut self.channel_messages,
            RecordedChannelMessage {
                direction: direction.to_string(),
                channel: message.channel.clone(),
                source_peer_id: message.source_peer_id.clone(),
                target_peer_id: message.target_peer_id.clone(),
                message_kind: message.message_kind.clone(),
                correlation_id: message.correlation_id.clone(),
                content_type: message.content_type.clone(),
                metadata_json: parse_optional_json(&message.metadata_json),
                text_preview: preview_bytes(&message.body),
                body_len: message.body.len(),
            },
        );
    }

    fn record_bulk_message(&mut self, direction: &str, message: &proto::BulkTransferMessage) {
        if direction == "outbound"
            && matches!(
                proto::bulk_transfer_message::Kind::try_from(message.kind).ok(),
                Some(proto::bulk_transfer_message::Kind::Offer)
            )
        {
            self.sent_bulk_transfers += 1;
        }
        push_bounded(
            &mut self.bulk_events,
            RecordedBulkEvent {
                direction: direction.to_string(),
                kind: bulk_kind_name(message.kind).into(),
                transfer_id: message.transfer_id.clone(),
                source_peer_id: message.source_peer_id.clone(),
                target_peer_id: message.target_peer_id.clone(),
                correlation_id: message.correlation_id.clone(),
                content_type: message.content_type.clone(),
                metadata_json: parse_optional_json(&message.metadata_json),
                total_bytes: message.total_bytes,
                offset: message.offset,
                body_len: message.body.len(),
                final_chunk: message.final_chunk,
            },
        );
    }

    fn note_bulk_receive(&mut self, message: &proto::BulkTransferMessage) {
        match proto::bulk_transfer_message::Kind::try_from(message.kind).ok() {
            Some(proto::bulk_transfer_message::Kind::Offer) => {
                self.transfer_state.insert(
                    message.transfer_id.clone(),
                    TransferAccumulator {
                        source_peer_id: message.source_peer_id.clone(),
                        target_peer_id: message.target_peer_id.clone(),
                        content_type: message.content_type.clone(),
                        total_bytes: message.total_bytes,
                        bytes: Vec::new(),
                    },
                );
            }
            Some(proto::bulk_transfer_message::Kind::Chunk) => {
                let entry = self
                    .transfer_state
                    .entry(message.transfer_id.clone())
                    .or_default();
                if entry.source_peer_id.is_empty() {
                    entry.source_peer_id = message.source_peer_id.clone();
                }
                if entry.target_peer_id.is_empty() {
                    entry.target_peer_id = message.target_peer_id.clone();
                }
                if entry.content_type.is_empty() {
                    entry.content_type = message.content_type.clone();
                }
                if entry.total_bytes == 0 {
                    entry.total_bytes = message.total_bytes;
                }
                entry.bytes.extend_from_slice(&message.body);
            }
            Some(proto::bulk_transfer_message::Kind::Complete) => {
                if let Some(entry) = self.transfer_state.remove(&message.transfer_id) {
                    self.completed_transfers.insert(
                        message.transfer_id.clone(),
                        CompletedTransfer {
                            transfer_id: message.transfer_id.clone(),
                            source_peer_id: entry.source_peer_id,
                            target_peer_id: entry.target_peer_id,
                            content_type: entry.content_type,
                            total_bytes: entry.total_bytes,
                            received_bytes: entry.bytes.len() as u64,
                            preview: preview_bytes(&entry.bytes),
                        },
                    );
                }
            }
            _ => {}
        }
    }
}

struct ExamplePlugin {
    state: Arc<Mutex<ExampleState>>,
}

#[async_trait]
impl Plugin for ExamplePlugin {
    fn plugin_id(&self) -> &str {
        PLUGIN_ID
    }

    fn plugin_version(&self) -> String {
        env!("CARGO_PKG_VERSION").into()
    }

    fn server_info(&self) -> ServerInfo {
        server_info()
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            format!("channel:{EXAMPLE_CHANNEL}"),
            "bulk:example".into(),
            "mesh-events".into(),
        ]
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        let state = self.state.lock().await;
        Ok(format!(
            "peers={} messages={} bulk={} mesh_events={}",
            state.known_peers.len(),
            state.channel_messages.len(),
            state.bulk_events.len(),
            state.mesh_events.len()
        ))
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(Some(list_tools_result()))
    }

    async fn call_tool(
        &mut self,
        request: ToolCallRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        Ok(Some(handle_tool_call(&self.state, context, request).await?))
    }

    async fn on_channel_message(
        &mut self,
        message: proto::ChannelMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        let mut state = self.state.lock().await;
        state.record_channel_message("inbound", &message);
        if should_ack_channel(&message) {
            let ack = proto::ChannelMessage {
                body: json_bytes(&json!({
                    "acknowledged_kind": message.message_kind,
                    "received_bytes": message.body.len(),
                    "received_by": state.local_peer_id,
                }))?,
                ..channel_message(
                    message.channel.clone(),
                    message.source_peer_id.clone(),
                    "application/json",
                    Vec::new(),
                    "example.ack",
                )
            };
            let ack = proto::ChannelMessage {
                correlation_id: if message.correlation_id.is_empty() {
                    state.next_token("ack")
                } else {
                    message.correlation_id.clone()
                },
                metadata_json: json_string(&json!({
                    "reply_to": message.correlation_id,
                }))?,
                ..ack
            };
            state.record_channel_message("outbound", &ack);
            context.send_channel(ack).await?;
        }
        Ok(())
    }

    async fn on_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        let mut state = self.state.lock().await;
        state.record_bulk_message("inbound", &message);
        state.note_bulk_receive(&message);
        if should_ack_bulk_offer(&message) {
            let ack = proto::BulkTransferMessage {
                transfer_id: message.transfer_id.clone(),
                correlation_id: message.correlation_id.clone(),
                metadata_json: json_string(&json!({
                    "accepted_by": state.local_peer_id,
                }))?,
                ..bulk_transfer_message(
                    proto::bulk_transfer_message::Kind::Accept as i32,
                    message.channel.clone(),
                    message.source_peer_id.clone(),
                    message.content_type.clone(),
                    message.total_bytes,
                    0,
                    Vec::new(),
                    false,
                )
            };
            state.record_bulk_message("outbound", &ack);
            context.send_bulk(ack).await?;
        }
        Ok(())
    }

    async fn on_mesh_event(
        &mut self,
        event: proto::MeshEvent,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        self.state.lock().await.record_mesh_event(&event);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    PluginRuntime::run(ExamplePlugin {
        state: Arc::new(Mutex::new(ExampleState::default())),
    })
    .await
}

async fn handle_tool_call(
    state: &Arc<Mutex<ExampleState>>,
    context: &mut PluginContext<'_>,
    request: ToolCallRequest,
) -> PluginResult<CallToolResult> {
    match request.name.as_str() {
        "snapshot" => {
            let params: SnapshotParams = request.arguments_or_default()?;
            let snapshot = state
                .lock()
                .await
                .snapshot(params.limit.unwrap_or(DEFAULT_SNAPSHOT_LIMIT));
            structured_tool_result(snapshot)
        }
        "clear" => {
            let _params: ClearParams = request.arguments_or_default()?;
            state.lock().await.clear_history();
            structured_tool_result(json!({
                "ok": true,
                "cleared": ["mesh_events", "channel_messages", "bulk_events", "completed_transfers"],
            }))
        }
        "send_message" => {
            let params: SendMessageArguments = request.arguments()?;
            let mut state = state.lock().await;
            let target_peer_id = normalize_target_peer_id(params.target_peer_id);
            let correlation_id = state.next_token("msg");
            let message = proto::ChannelMessage {
                correlation_id: correlation_id.clone(),
                metadata_json: json_string(&json!({
                    "request_ack": params.request_ack.unwrap_or(true),
                }))?,
                ..channel_message(
                    EXAMPLE_CHANNEL,
                    target_peer_id.clone(),
                    "text/plain",
                    params.text.into_bytes(),
                    "example.message",
                )
            };
            state.record_channel_message("outbound", &message);
            context.send_channel(message).await?;
            structured_tool_result(json!({
                "ok": true,
                "channel": EXAMPLE_CHANNEL,
                "target_peer_id": render_target(&target_peer_id),
                "correlation_id": correlation_id,
            }))
        }
        "send_bulk" => {
            let params: SendBulkArguments = request.arguments()?;
            let mut state = state.lock().await;
            let target_peer_id = normalize_target_peer_id(params.target_peer_id);
            let correlation_id = state.next_token("bulk-corr");
            let transfer_id = state.next_token("bulk");
            let bytes = params.text.into_bytes();
            let chunk_size = params.chunk_size.unwrap_or(DEFAULT_BULK_CHUNK_SIZE).max(1);
            let metadata_json = json_string(&json!({
                "request_ack": params.request_ack.unwrap_or(true),
            }))?;

            let offer = proto::BulkTransferMessage {
                transfer_id: transfer_id.clone(),
                correlation_id: correlation_id.clone(),
                metadata_json: metadata_json.clone(),
                ..bulk_transfer_message(
                    proto::bulk_transfer_message::Kind::Offer as i32,
                    EXAMPLE_CHANNEL,
                    target_peer_id.clone(),
                    "text/plain",
                    bytes.len() as u64,
                    0,
                    Vec::new(),
                    false,
                )
            };
            state.record_bulk_message("outbound", &offer);
            context.send_bulk(offer).await?;

            let mut offset = 0usize;
            for chunk in bytes.chunks(chunk_size) {
                let message = proto::BulkTransferMessage {
                    transfer_id: transfer_id.clone(),
                    correlation_id: correlation_id.clone(),
                    metadata_json: metadata_json.clone(),
                    ..bulk_transfer_message(
                        proto::bulk_transfer_message::Kind::Chunk as i32,
                        EXAMPLE_CHANNEL,
                        target_peer_id.clone(),
                        "text/plain",
                        bytes.len() as u64,
                        offset as u64,
                        chunk.to_vec(),
                        false,
                    )
                };
                offset += chunk.len();
                state.record_bulk_message("outbound", &message);
                context.send_bulk(message).await?;
            }

            let complete = proto::BulkTransferMessage {
                transfer_id: transfer_id.clone(),
                correlation_id: correlation_id.clone(),
                metadata_json,
                ..bulk_transfer_message(
                    proto::bulk_transfer_message::Kind::Complete as i32,
                    EXAMPLE_CHANNEL,
                    target_peer_id.clone(),
                    "text/plain",
                    bytes.len() as u64,
                    bytes.len() as u64,
                    Vec::new(),
                    true,
                )
            };
            state.record_bulk_message("outbound", &complete);
            context.send_bulk(complete).await?;

            structured_tool_result(json!({
                "ok": true,
                "transfer_id": transfer_id,
                "target_peer_id": render_target(&target_peer_id),
                "total_bytes": bytes.len(),
                "chunk_size": chunk_size,
            }))
        }
        other => Ok(tool_error(format!("unknown tool '{other}'"))),
    }
}

fn server_info() -> ServerInfo {
    plugin_server_info(
        PLUGIN_ID,
        env!("CARGO_PKG_VERSION"),
        "Plugin Surface Example",
        "Standalone example plugin that exercises tools, channel messages, bulk transfers, and mesh events.",
        None::<String>,
    )
}

fn list_tools_result() -> ListToolsResult {
    list_tools(vec![
        tool_with_schema(
            "snapshot",
            "Inspect the example plugin state: known peers, mesh events, recent channel messages, recent bulk transfers, and counters.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1 }
                }
            })
            .as_object()
            .cloned()
            .unwrap(),
        ),
        json_schema_tool::<SendMessageArguments>(
            "send_message",
            "Send a plugin channel message to one peer or broadcast to all peers. Leave target_peer_id empty or set it to 'all' to broadcast.",
        ),
        json_schema_tool::<SendBulkArguments>(
            "send_bulk",
            "Send a bulk transfer to one peer or broadcast to all peers. This emits OFFER, CHUNK, and COMPLETE frames so the full bulk transport path is exercised.",
        ),
        tool_with_schema(
            "clear",
            "Clear recorded example-plugin history while keeping the current peer snapshot.",
            empty_object_schema(),
        ),
    ])
}

fn should_ack_channel(message: &proto::ChannelMessage) -> bool {
    !message.source_peer_id.is_empty()
        && message.message_kind != "example.ack"
        && parse_optional_json(&message.metadata_json)
            .and_then(|value| value.get("request_ack").and_then(|v| v.as_bool()))
            .unwrap_or(false)
}

fn should_ack_bulk_offer(message: &proto::BulkTransferMessage) -> bool {
    matches!(
        proto::bulk_transfer_message::Kind::try_from(message.kind).ok(),
        Some(proto::bulk_transfer_message::Kind::Offer)
    ) && !message.source_peer_id.is_empty()
        && parse_optional_json(&message.metadata_json)
            .and_then(|value| value.get("request_ack").and_then(|v| v.as_bool()))
            .unwrap_or(false)
}

fn preview_bytes(bytes: &[u8]) -> String {
    let mut preview = String::from_utf8_lossy(bytes).to_string();
    if preview.len() > 160 {
        preview.truncate(160);
        preview.push_str("...");
    }
    preview
}

fn normalize_target_peer_id(target_peer_id: Option<String>) -> String {
    let Some(target_peer_id) = target_peer_id else {
        return String::new();
    };
    let trimmed = target_peer_id.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("all")
        || trimmed.eq_ignore_ascii_case("broadcast")
        || trimmed == "*"
    {
        String::new()
    } else {
        trimmed.to_string()
    }
}

fn render_target(target_peer_id: &str) -> String {
    if target_peer_id.is_empty() {
        "all".into()
    } else {
        target_peer_id.to_string()
    }
}

fn push_bounded<T>(items: &mut Vec<T>, item: T) {
    items.push(item);
    if items.len() > MAX_RECENT_ITEMS {
        let overflow = items.len() - MAX_RECENT_ITEMS;
        items.drain(0..overflow);
    }
}

fn recent_items<T: Clone>(items: &[T], limit: usize) -> Vec<T> {
    let len = items.len();
    let start = len.saturating_sub(limit);
    items[start..].to_vec()
}

fn peer_summary(peer: &proto::MeshPeer) -> PeerSummary {
    PeerSummary {
        peer_id: peer.peer_id.clone(),
        role: peer.role.clone(),
        version: peer.version.clone(),
        capabilities: peer.capabilities.clone(),
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        rtt_ms: peer.rtt_ms,
    }
}

fn mesh_event_kind_name(kind: i32) -> &'static str {
    match proto::mesh_event::Kind::try_from(kind).ok() {
        Some(proto::mesh_event::Kind::PeerUp) => "peer_up",
        Some(proto::mesh_event::Kind::PeerDown) => "peer_down",
        Some(proto::mesh_event::Kind::PeerUpdated) => "peer_updated",
        Some(proto::mesh_event::Kind::LocalAccepting) => "local_accepting",
        Some(proto::mesh_event::Kind::LocalStandby) => "local_standby",
        Some(proto::mesh_event::Kind::MeshIdUpdated) => "mesh_id_updated",
        _ => "unknown",
    }
}

fn bulk_kind_name(kind: i32) -> &'static str {
    match proto::bulk_transfer_message::Kind::try_from(kind).ok() {
        Some(proto::bulk_transfer_message::Kind::Offer) => "offer",
        Some(proto::bulk_transfer_message::Kind::Accept) => "accept",
        Some(proto::bulk_transfer_message::Kind::Reject) => "reject",
        Some(proto::bulk_transfer_message::Kind::Chunk) => "chunk",
        Some(proto::bulk_transfer_message::Kind::Complete) => "complete",
        Some(proto::bulk_transfer_message::Kind::Cancel) => "cancel",
        Some(proto::bulk_transfer_message::Kind::Error) => "error",
        _ => "unknown",
    }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
