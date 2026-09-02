use crate::binary_transport::stage_execution::binary_message_attrs;
use crate::binary_transport::stage_execution::estimated_reply_wire_bytes;
use crate::binary_transport::stage_execution::insert_optional_unix_nanos;
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
use std::collections::BTreeMap;

pub(super) struct UpstreamReplyWriteSpan {
    pub(super) reply_kind: WireReplyKind,
    pub(super) predicted_token_count: usize,
    pub(super) start_unix_nanos: u64,
    pub(super) end_unix_nanos: u64,
    pub(super) write_ms: f64,
}

pub(super) struct BinaryMessageTiming<'a> {
    pub(super) message_start_unix_nanos: u64,
    pub(super) message_end_unix_nanos: u64,
    pub(super) compute_start_unix_nanos: u64,
    pub(super) compute_end_unix_nanos: u64,
    pub(super) forward_write_start_unix_nanos: Option<u64>,
    pub(super) forward_write_end_unix_nanos: Option<u64>,
    pub(super) downstream_wait_start_unix_nanos: Option<u64>,
    pub(super) downstream_wait_end_unix_nanos: Option<u64>,
    pub(super) upstream_reply_start_unix_nanos: Option<u64>,
    pub(super) upstream_reply_end_unix_nanos: Option<u64>,
    pub(super) compute_ms: f64,
    pub(super) recv_read_ms: f64,
    pub(super) input_activation_decode_ms: f64,
    pub(super) runtime_lock_wait_ms: f64,
    pub(super) runtime_lock_hold_ms: f64,
    pub(super) runtime_lock_acquires: usize,
    pub(super) runtime_sessions_before: Option<&'a RuntimeSessionStats>,
    pub(super) runtime_sessions_after: Option<&'a RuntimeSessionStats>,
    pub(super) forward_write_ms: f64,
    pub(super) forward_activation_encode_ms: f64,
    pub(super) downstream_wait_ms: f64,
    pub(super) upstream_reply_ms: f64,
    pub(super) forward_mode: &'a str,
    pub(super) message_elapsed_ms: f64,
    pub(super) input_activation_bytes: usize,
    pub(super) output_activation_bytes: usize,
    pub(super) max_deferred_prefill_replies: usize,
    pub(super) pending_prefill_replies_before: usize,
    pub(super) pending_prefill_replies_after: usize,
    pub(super) credit_wait_count: usize,
    pub(super) deferred_prefill_replies_drained: usize,
}

pub(super) fn emit_binary_message_timing(
    telemetry: &Telemetry,
    config: &StageConfig,
    session_id: u64,
    message: &StageWireMessage,
    timing: BinaryMessageTiming<'_>,
) {
    if !telemetry.is_debug_enabled() {
        return;
    }
    let mut attrs = binary_message_attrs(config, session_id, message);
    attrs.insert(
        "llama_stage.message_start_unix_nanos".to_string(),
        json!(timing.message_start_unix_nanos),
    );
    attrs.insert(
        "llama_stage.message_end_unix_nanos".to_string(),
        json!(timing.message_end_unix_nanos),
    );
    attrs.insert(
        "llama_stage.compute_start_unix_nanos".to_string(),
        json!(timing.compute_start_unix_nanos),
    );
    attrs.insert(
        "llama_stage.compute_end_unix_nanos".to_string(),
        json!(timing.compute_end_unix_nanos),
    );
    attrs.insert(
        "llama_stage.compute_ms".to_string(),
        json!(timing.compute_ms),
    );
    attrs.insert(
        "llama_stage.recv_read_ms".to_string(),
        json!(timing.recv_read_ms),
    );
    attrs.insert(
        "skippy.upstream_message_wait_ms".to_string(),
        json!(timing.recv_read_ms),
    );
    attrs.insert(
        "llama_stage.input_activation_decode_ms".to_string(),
        json!(timing.input_activation_decode_ms),
    );
    attrs.insert(
        "llama_stage.runtime_lock_wait_ms".to_string(),
        json!(timing.runtime_lock_wait_ms),
    );
    attrs.insert(
        "llama_stage.runtime_lock_hold_ms".to_string(),
        json!(timing.runtime_lock_hold_ms),
    );
    attrs.insert(
        "llama_stage.runtime_lock_acquires".to_string(),
        json!(timing.runtime_lock_acquires),
    );
    if let Some(stats) = timing.runtime_sessions_before {
        insert_runtime_session_stats(&mut attrs, "llama_stage.runtime_sessions_before", stats);
    }
    if let Some(stats) = timing.runtime_sessions_after {
        insert_runtime_session_stats(&mut attrs, "llama_stage.runtime_sessions_after", stats);
    }
    attrs.insert(
        "llama_stage.forward_write_ms".to_string(),
        json!(timing.forward_write_ms),
    );
    attrs.insert(
        "llama_stage.activation_encode_ms".to_string(),
        json!(timing.forward_activation_encode_ms),
    );
    attrs.insert(
        "llama_stage.downstream_wait_ms".to_string(),
        json!(timing.downstream_wait_ms),
    );
    attrs.insert("skippy.compute_ms".to_string(), json!(timing.compute_ms));
    attrs.insert(
        "skippy.forward_write_ms".to_string(),
        json!(timing.forward_write_ms),
    );
    attrs.insert(
        "skippy.downstream_wait_ms".to_string(),
        json!(timing.downstream_wait_ms),
    );
    attrs.insert(
        "skippy.upstream_reply_ms".to_string(),
        json!(timing.upstream_reply_ms),
    );
    attrs.insert(
        "llama_stage.forward_mode".to_string(),
        json!(timing.forward_mode),
    );
    insert_optional_unix_nanos(
        &mut attrs,
        "llama_stage.forward_write_start_unix_nanos",
        timing.forward_write_start_unix_nanos,
    );
    insert_optional_unix_nanos(
        &mut attrs,
        "llama_stage.forward_write_end_unix_nanos",
        timing.forward_write_end_unix_nanos,
    );
    insert_optional_unix_nanos(
        &mut attrs,
        "llama_stage.downstream_wait_start_unix_nanos",
        timing.downstream_wait_start_unix_nanos,
    );
    insert_optional_unix_nanos(
        &mut attrs,
        "llama_stage.downstream_wait_end_unix_nanos",
        timing.downstream_wait_end_unix_nanos,
    );
    insert_optional_unix_nanos(
        &mut attrs,
        "llama_stage.upstream_reply_start_unix_nanos",
        timing.upstream_reply_start_unix_nanos,
    );
    insert_optional_unix_nanos(
        &mut attrs,
        "llama_stage.upstream_reply_end_unix_nanos",
        timing.upstream_reply_end_unix_nanos,
    );
    attrs.insert(
        "skippy.message_elapsed_ms".to_string(),
        json!(timing.message_elapsed_ms),
    );
    attrs.insert(
        "skippy.input_activation_bytes".to_string(),
        json!(timing.input_activation_bytes),
    );
    attrs.insert(
        "skippy.output_activation_bytes".to_string(),
        json!(timing.output_activation_bytes),
    );
    attrs.insert(
        "skippy.prefill_credit_limit".to_string(),
        json!(timing.max_deferred_prefill_replies),
    );
    attrs.insert(
        "skippy.prefill_pending_replies_before".to_string(),
        json!(timing.pending_prefill_replies_before),
    );
    attrs.insert(
        "skippy.prefill_pending_replies_after".to_string(),
        json!(timing.pending_prefill_replies_after),
    );
    attrs.insert(
        "skippy.prefill_credit_wait_count".to_string(),
        json!(timing.credit_wait_count),
    );
    attrs.insert(
        "skippy.prefill_deferred_replies_drained".to_string(),
        json!(timing.deferred_prefill_replies_drained),
    );
    telemetry.emit_debug_span(
        "stage.binary_message_timing",
        attrs,
        timing.message_start_unix_nanos,
        timing.message_end_unix_nanos,
    );
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

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_binary_message_received(
    telemetry: &Telemetry,
    config: &StageConfig,
    session_id: u64,
    message: &StageWireMessage,
    start_unix_nanos: u64,
    end_unix_nanos: u64,
    read_ms: f64,
) {
    if !telemetry.is_debug_enabled() {
        return;
    }
    let mut attrs = binary_message_attrs(config, session_id, message);
    attrs.insert(
        "llama_stage.recv_start_unix_nanos".to_string(),
        json!(start_unix_nanos),
    );
    attrs.insert(
        "llama_stage.recv_end_unix_nanos".to_string(),
        json!(end_unix_nanos),
    );
    attrs.insert("llama_stage.recv_read_ms".to_string(), json!(read_ms));
    attrs.insert(
        "skippy.upstream_message_wait_ms".to_string(),
        json!(read_ms),
    );
    attrs.insert(
        "llama_stage.source_stage_index".to_string(),
        json!(message.state.source_stage_index),
    );
    attrs.insert(
        "llama_stage.configured_upstream_stage_index".to_string(),
        json!(config.upstream.as_ref().map(|peer| peer.stage_index)),
    );
    attrs.insert(
        "llama_stage.message_wire_bytes".to_string(),
        json!(message.estimated_wire_bytes()),
    );
    attrs.insert(
        "skippy.activation_bytes".to_string(),
        json!(message.activation.len()),
    );
    telemetry.emit_debug_span("stage.binary_recv", attrs, start_unix_nanos, end_unix_nanos);
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

pub(super) fn record_prefill_stage_compute(
    stats: &mut StageReplyStats,
    config: &StageConfig,
    message: &StageWireMessage,
    compute_ms: f64,
) {
    if !message.kind.is_prefill() {
        return;
    }
    stats.observe_prefill_compute(
        config.stage_index,
        ms_to_us(compute_ms),
        message.token_count.max(0) as usize,
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
}
