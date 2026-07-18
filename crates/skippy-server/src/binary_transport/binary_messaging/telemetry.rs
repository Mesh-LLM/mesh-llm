use crate::binary_transport::stage_execution::binary_message_attrs;
use crate::binary_transport::stage_execution::estimated_reply_wire_bytes;
use crate::binary_transport::stage_execution::ms_to_us;
use crate::runtime_state::RuntimeSessionStats;
use crate::telemetry::Telemetry;
use serde_json::Value;
use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireMessageKind;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::state_flags;
use std::collections::BTreeMap;

pub(super) struct UpstreamReplyWriteSpan {
    pub(super) reply_kind: WireReplyKind,
    pub(super) predicted_token_count: usize,
    pub(super) start_unix_nanos: u64,
    pub(super) end_unix_nanos: u64,
    pub(super) write_ms: f64,
}

#[derive(Clone, Copy)]
pub(super) struct SessionControlTiming {
    pub(super) flush_us: i64,
    pub(super) prefill_drain_us: i64,
    pub(super) local_us: i64,
    pub(super) downstream_write_us: i64,
    pub(super) downstream_wait_us: i64,
    pub(super) total_us: i64,
    pub(super) prefill_drained_replies: i64,
}

pub(super) fn emit_upstream_reply_write_span(
    telemetry: &Telemetry,
    config: &StageConfig,
    session_id: u64,
    message: &StageWireMessage,
    span: UpstreamReplyWriteSpan,
) {
    let mut attrs = binary_message_attrs(config, session_id, message);
    attrs.insert(
        "llama_stage.reply_kind".to_string(),
        json!(format!("{:?}", span.reply_kind)),
    );
    attrs.insert(
        "llama_stage.reply_predicted_token_count".to_string(),
        json!(span.predicted_token_count),
    );
    attrs.insert(
        "llama_stage.upstream_reply_ms".to_string(),
        json!(span.write_ms),
    );
    attrs.insert(
        "llama_stage.reply_wire_bytes".to_string(),
        json!(estimated_reply_wire_bytes(
            span.reply_kind,
            span.predicted_token_count
        )),
    );
    attrs.insert(
        "llama_stage.upstream_reply_start_unix_nanos".to_string(),
        json!(span.start_unix_nanos),
    );
    attrs.insert(
        "llama_stage.upstream_reply_end_unix_nanos".to_string(),
        json!(span.end_unix_nanos),
    );
    telemetry.emit_debug_span(
        "stage.binary_upstream_reply_write",
        attrs,
        span.start_unix_nanos,
        span.end_unix_nanos,
    );
}

pub(super) fn insert_runtime_session_stats(
    attrs: &mut BTreeMap<String, Value>,
    prefix: &str,
    stats: &RuntimeSessionStats,
) {
    attrs.insert(
        format!("{prefix}.active_sessions"),
        json!(stats.active_sessions),
    );
    attrs.insert(
        format!("{prefix}.idle_sessions"),
        json!(stats.idle_sessions),
    );
    attrs.insert(
        format!("{prefix}.idle_resident_prefixes"),
        json!(stats.idle_resident_prefixes),
    );
    attrs.insert(
        format!("{prefix}.tracked_token_counts"),
        json!(stats.tracked_token_counts),
    );
    attrs.insert(format!("{prefix}.checkpoints"), json!(stats.checkpoints));
}

pub(super) fn record_session_control_timing(
    stats: &mut StageReplyStats,
    kind: WireMessageKind,
    timing: SessionControlTiming,
) {
    match kind {
        WireMessageKind::CheckpointSession => {
            stats.checkpoint_flush_us += timing.flush_us;
            stats.checkpoint_prefill_drain_us += timing.prefill_drain_us;
            stats.checkpoint_local_us += timing.local_us;
            stats.checkpoint_downstream_write_us += timing.downstream_write_us;
            stats.checkpoint_downstream_wait_us += timing.downstream_wait_us;
            stats.checkpoint_total_us += timing.total_us;
            stats.checkpoint_prefill_drained_replies += timing.prefill_drained_replies;
        }
        WireMessageKind::RestoreSession => {
            stats.restore_flush_us += timing.flush_us;
            stats.restore_prefill_drain_us += timing.prefill_drain_us;
            stats.restore_local_us += timing.local_us;
            stats.restore_downstream_write_us += timing.downstream_write_us;
            stats.restore_downstream_wait_us += timing.downstream_wait_us;
            stats.restore_total_us += timing.total_us;
            stats.restore_prefill_drained_replies += timing.prefill_drained_replies;
        }
        _ => {}
    }
}

pub(super) fn record_prefill_edge_transport(
    stats: &mut StageReplyStats,
    config: &StageConfig,
    message: &StageWireMessage,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
    activation_bytes: usize,
) {
    if !message.kind.is_prefill() || config.downstream.is_none() {
        return;
    }
    stats.observe_prefill_edge_transport(
        config.stage_index,
        ms_to_us(forward_write_ms),
        ms_to_us(downstream_wait_ms),
        activation_bytes,
    );
}

pub(super) fn record_verify_window_timing(
    stats: &mut StageReplyStats,
    message: &StageWireMessage,
    compute_ms: f64,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
) {
    if message.kind != WireMessageKind::VerifyWindow {
        return;
    }
    let compute_us = ms_to_us(compute_ms);
    let forward_write_us = ms_to_us(forward_write_ms);
    let downstream_wait_us = ms_to_us(downstream_wait_ms);
    let token_count = i64::from(message.token_count.max(0));
    stats.verify_window_compute_us += compute_us;
    stats.verify_window_forward_write_us += forward_write_us;
    stats.verify_window_downstream_wait_us += downstream_wait_us;
    stats.verify_window_total_us += compute_us + forward_write_us + downstream_wait_us;
    stats.verify_window_stage_count += 1;
    stats.verify_window_request_count += 1;
    stats.verify_window_token_count += token_count;
    stats.verify_window_max_tokens = stats.verify_window_max_tokens.max(token_count);
    if (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0 {
        stats.verify_window_checkpointed_requests += 1;
    } else {
        stats.verify_window_skip_checkpoint_requests += 1;
    }
}
