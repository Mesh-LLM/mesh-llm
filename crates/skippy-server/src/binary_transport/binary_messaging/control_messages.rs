use super::async_forwarder::AsyncForwarder;
use super::reply::{
    configure_prediction_return_stream, drain_deferred_prefill_replies,
    normalize_downstream_prefix_restore_reply,
};
use super::session_tracker::ConnectionSessionTracker;
use super::summary::BinaryRequestSummary;
use crate::binary_transport::WireCondition;
use crate::binary_transport::binary_kv::{
    maybe_prefix_cache_control, maybe_record_binary_full_prefill, take_shared_prefill_tokens,
};
use crate::binary_transport::direct_return::PredictionReturnSinks;
use crate::binary_transport::restore_prefill_decode::handle_binary_restore_prefill_decode_control;
use crate::binary_transport::stage_execution::{
    binary_message_attrs, elapsed_ms, runtime_sampling_config, stage_mask, token_sideband_or_fill,
};
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::kv_integration::KvStageIntegration;
use crate::telemetry::{Telemetry, now_unix_nanos};
use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_protocol::binary::{
    StageReplyStats, StageWireMessage, WireMessageKind, WireReplyKind, recv_reply,
    send_reply_ack_with_stats,
};
use skippy_protocol::{StageConfig, StageTopology};
use std::collections::BTreeMap;
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_stop(
    config: &StageConfig,
    iteration_scheduler: &IterationScheduler,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    upstream: &mut TcpStream,
    mut downstream: Option<&mut TcpStream>,
    downstream_wire_condition: WireCondition,
    message: &StageWireMessage,
    session_key: &str,
    session_id: u64,
    pending_prefill_replies: usize,
    pending_reply_stats: &mut StageReplyStats,
    request_summary: &mut BinaryRequestSummary,
    async_forwarder: Option<&mut AsyncForwarder>,
    session_tracker: &mut ConnectionSessionTracker,
    prediction_return_streams: &mut BTreeMap<(u64, u64), TcpStream>,
    prediction_return_sinks: &PredictionReturnSinks,
) -> Result<()> {
    if pending_prefill_replies != 0 {
        bail!("cannot stop with {pending_prefill_replies} deferred prefill replies");
    }
    let mut stop_stats = std::mem::take(pending_reply_stats);
    request_summary.emit(telemetry, config, session_id);
    *request_summary = BinaryRequestSummary::default();
    if let Some(downstream) = downstream.as_mut() {
        if let Some(forwarder) = async_forwarder {
            forwarder
                .flush()
                .context("flush async forwards before stop")?;
        }
        write_stage_message_conditioned(&mut **downstream, message, downstream_wire_condition)
            .context("forward binary stop")?;
        let reply = recv_reply(&mut **downstream).context("stop downstream ACK")?;
        if reply.kind != WireReplyKind::Ack {
            bail!("stop expected downstream ACK");
        }
        stop_stats.merge(reply.stats);
    }
    let reset_start_unix_nanos = now_unix_nanos() as u64;
    let reset_timer = Instant::now();
    let accumulated =
        kv.and_then(|cache| take_shared_prefill_tokens(&cache.split_prefill_tokens, session_key));
    let scheduler_config = config.clone();
    let scheduler_kv = kv.cloned();
    let scheduler_telemetry = telemetry.clone();
    let scheduler_session_key = session_key.to_string();
    let scheduler_message = message.clone();
    let outcome = iteration_scheduler
        .execute_runtime_timed("binary-session-stop", move |runtime| {
            let record = accumulated.map(|tokens| {
                maybe_record_binary_full_prefill(
                    &scheduler_config,
                    runtime,
                    scheduler_kv.as_ref(),
                    &scheduler_telemetry,
                    &scheduler_session_key,
                    &scheduler_message,
                    &tokens,
                )
            });
            let drop_stats = runtime
                .drop_session_timed(&scheduler_session_key)
                .map_err(|error| openai_frontend::OpenAiError::backend(format!("{error:#}")))?;
            Ok((record, drop_stats))
        })
        .map_err(|error| anyhow::anyhow!(format!("{error:#}")))
        .context("reset binary stage session")?;
    let runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
    let (record, drop_stats) = outcome.value;
    if let Some(record) = record
        && record.recorded_pages > 0
    {
        stop_stats.kv_recorded_pages += record.recorded_pages as i64;
        stop_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
    }
    let reset_end_unix_nanos = now_unix_nanos() as u64;
    let mut reset_attrs = binary_message_attrs(config, session_id, message);
    reset_attrs.insert(
        "llama_stage.runtime_lock_wait_ms".to_string(),
        json!(runtime_lock_wait_ms),
    );
    reset_attrs.insert(
        "llama_stage.session_reset_ms".to_string(),
        json!(drop_stats.reset_ms),
    );
    reset_attrs.insert(
        "llama_stage.session_reset".to_string(),
        json!(drop_stats.reset_session),
    );
    reset_attrs.insert(
        "llama_stage.lane_discarded".to_string(),
        json!(drop_stats.lane_discarded),
    );
    if let Some(reason) = drop_stats.lane_discard_reason.as_deref() {
        reset_attrs.insert("llama_stage.lane_discard_reason".to_string(), json!(reason));
    }
    reset_attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(elapsed_ms(reset_timer)),
    );
    super::telemetry::insert_runtime_session_stats(
        &mut reset_attrs,
        "llama_stage.runtime_sessions_after",
        &drop_stats.stats_after,
    );
    telemetry.emit_debug_span(
        "stage.binary_session_stop",
        reset_attrs,
        reset_start_unix_nanos,
        reset_end_unix_nanos,
    );
    session_tracker.stopped(session_key);
    prediction_return_streams.remove(&(message.request_id, message.session_id));
    prediction_return_sinks.remove(message.request_id, message.session_id);
    send_reply_ack_with_stats(upstream, stop_stats).context("send stop ACK")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_verify_retirement(
    iteration_scheduler: &IterationScheduler,
    mut downstream: Option<&mut TcpStream>,
    downstream_wire_condition: WireCondition,
    message: &StageWireMessage,
    session_key: &str,
    async_forwarder: Option<&mut AsyncForwarder>,
) -> Result<()> {
    if let Some(forwarder) = async_forwarder {
        forwarder
            .flush()
            .context("flush async forwards before verify retirement")?;
    }
    let token_start = u64::try_from(message.pos_start)
        .context("verify retirement position must be non-negative")?;
    let token_count = u64::try_from(message.token_count)
        .context("verify retirement count must be non-negative")?;
    let scheduler_session_key = session_key.to_string();
    iteration_scheduler
        .execute_runtime("binary-verify-retire", move |runtime| {
            runtime
                .retire_verify_checkpoint(&scheduler_session_key, token_start, token_count)
                .map_err(|error| openai_frontend::OpenAiError::backend(format!("{error:#}")))
        })
        .map_err(|error| anyhow::anyhow!(format!("{error:#}")))
        .context("retire binary stage verify checkpoint")?;
    if let Some(downstream) = downstream.as_mut() {
        write_stage_message_conditioned(&mut **downstream, message, downstream_wire_condition)
            .context("forward verify retirement")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_session_control(
    iteration_scheduler: &IterationScheduler,
    upstream: &mut TcpStream,
    mut downstream: Option<&mut TcpStream>,
    downstream_wire_condition: WireCondition,
    message: &StageWireMessage,
    session_key: &str,
    pending_prefill_replies: &mut usize,
    pending_reply_stats: &mut StageReplyStats,
    async_forwarder: Option<&mut AsyncForwarder>,
) -> Result<()> {
    let mut control_stats = std::mem::take(pending_reply_stats);
    if let Some(forwarder) = async_forwarder {
        forwarder
            .flush()
            .context("flush async forwards before session control")?;
    }
    drain_deferred_prefill_replies(
        downstream.as_deref_mut(),
        pending_prefill_replies,
        &mut control_stats,
    )
    .context("drain deferred replies before session control")?;
    match message.kind {
        WireMessageKind::TrimSession => {
            let scheduler_session_key = session_key.to_string();
            let token_count = message.token_count.max(0) as u64;
            iteration_scheduler
                .execute_runtime("binary-session-trim", move |runtime| {
                    runtime
                        .trim_session(&scheduler_session_key, token_count)
                        .map_err(|error| {
                            openai_frontend::OpenAiError::backend(format!("{error:#}"))
                        })
                })
                .map_err(|error| anyhow::anyhow!(format!("{error:#}")))
                .context("trim binary stage session")?;
        }
        _ => unreachable!("session control checked above"),
    }
    if let Some(downstream) = downstream.as_mut() {
        write_stage_message_conditioned(&mut **downstream, message, downstream_wire_condition)
            .context("forward session control")?;
        let reply = recv_reply(&mut **downstream).context("session control downstream ACK")?;
        if reply.kind != WireReplyKind::Ack {
            bail!("session control expected downstream ACK");
        }
        control_stats.merge(reply.stats);
    }
    send_reply_ack_with_stats(upstream, control_stats).context("session control ack")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_generation_control(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    iteration_scheduler: &IterationScheduler,
    upstream: &mut TcpStream,
    mut downstream: Option<&mut TcpStream>,
    downstream_wire_condition: WireCondition,
    downstream_connect_timeout_secs: u64,
    message: &StageWireMessage,
    session_key: &str,
    pending_prefill_replies: &mut usize,
    pending_reply_stats: &mut StageReplyStats,
    async_forwarder: Option<&mut AsyncForwarder>,
    prediction_return_sinks: &PredictionReturnSinks,
    prediction_return_streams: &mut BTreeMap<(u64, u64), TcpStream>,
) -> Result<()> {
    let mut generation_stats = std::mem::take(pending_reply_stats);
    if let Some(forwarder) = async_forwarder {
        forwarder
            .flush()
            .context("flush async forwards before generation config")?;
    }
    drain_deferred_prefill_replies(
        downstream.as_deref_mut(),
        pending_prefill_replies,
        &mut generation_stats,
    )
    .context("drain deferred replies before generation config")?;
    if let Some(downstream) = downstream.as_mut() {
        write_stage_message_conditioned(&mut **downstream, message, downstream_wire_condition)
            .context("forward generation config")?;
        let reply = recv_reply(&mut **downstream).context("generation config downstream ACK")?;
        if reply.kind != WireReplyKind::Ack {
            bail!("generation config expected downstream ACK");
        }
        generation_stats.merge(reply.stats);
    } else {
        if let Some(metadata) = message.chat_sampling_metadata.as_deref() {
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            let scheduler_session_key = session_key.to_string();
            let metadata = metadata.to_string();
            let prompt_token_count = message.state.prompt_token_count.max(0) as u64;
            iteration_scheduler
                .execute_runtime("binary-generation-config", move |runtime| {
                    runtime
                        .configure_chat_sampling(
                            &scheduler_session_key,
                            &metadata,
                            prompt_token_count,
                            sampling.as_ref(),
                        )
                        .map_err(|error| {
                            openai_frontend::OpenAiError::backend(format!("{error:#}"))
                        })
                })
                .map_err(|error| anyhow::anyhow!(format!("{error:#}")))
                .context("configure binary stage generation")?;
        }
        configure_prediction_return_stream(
            config,
            topology,
            message.request_id,
            message.session_id,
            downstream_connect_timeout_secs,
            prediction_return_sinks,
            prediction_return_streams,
        );
    }
    send_reply_ack_with_stats(upstream, generation_stats).context("generation config ack")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_prefix_cache_control(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    iteration_scheduler: &IterationScheduler,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    upstream: &mut TcpStream,
    mut downstream: Option<&mut TcpStream>,
    downstream_wire_condition: WireCondition,
    downstream_connect_timeout_secs: u64,
    activation_width: i32,
    native_mtp_enabled: bool,
    message: StageWireMessage,
    session_key: &str,
    session_id: u64,
    pending_prefill_replies: &mut usize,
    pending_reply_stats: &mut StageReplyStats,
    async_forwarder: Option<&mut AsyncForwarder>,
    prediction_return_sinks: &PredictionReturnSinks,
    prediction_return_streams: &mut BTreeMap<(u64, u64), TcpStream>,
) -> Result<()> {
    let control_started = Instant::now();
    let mut control_stats = std::mem::take(pending_reply_stats);
    if let Some(forwarder) = async_forwarder {
        forwarder
            .flush()
            .context("flush async forwards before prefix cache control")?;
    }
    drain_deferred_prefill_replies(
        downstream.as_deref_mut(),
        pending_prefill_replies,
        &mut control_stats,
    )
    .context("drain deferred replies before prefix cache control")?;
    if message.kind == WireMessageKind::TryRestorePrefillDecode {
        return handle_binary_restore_prefill_decode_control(
            config,
            topology,
            iteration_scheduler,
            kv,
            telemetry,
            session_key,
            session_id,
            message,
            downstream,
            downstream_wire_condition,
            activation_width,
            control_started,
            control_stats,
            prediction_return_sinks,
            prediction_return_streams,
            downstream_connect_timeout_secs,
            native_mtp_enabled,
        )
        .context("handle restore-prefill-decode control");
    }
    let token_ids =
        token_sideband_or_fill(&message).context("read prefix cache control token sideband")?;
    let scheduler_config = config.clone();
    let scheduler_kv = kv.cloned();
    let scheduler_telemetry = telemetry.clone();
    let scheduler_session_key = session_key.to_string();
    let scheduler_message = message.clone();
    let local = iteration_scheduler
        .execute_runtime("binary-prefix-control", move |runtime| {
            Ok(maybe_prefix_cache_control(
                &scheduler_config,
                runtime,
                scheduler_kv.as_ref(),
                &scheduler_telemetry,
                &scheduler_session_key,
                &scheduler_message,
                &token_ids,
            ))
        })
        .map_err(|error| anyhow::anyhow!(format!("{error:#}")))?;
    control_stats.merge(local.stats);
    if local.hit
        && let Some(downstream) = downstream.as_mut()
    {
        write_stage_message_conditioned(&mut **downstream, &message, downstream_wire_condition)
            .context("forward prefix cache control")?;
        let mut reply = recv_reply(&mut **downstream).context("prefix cache downstream ACK")?;
        if reply.kind != WireReplyKind::Ack {
            bail!("prefix cache control expected downstream ACK");
        }
        let downstream_missed =
            normalize_downstream_prefix_restore_reply(message.kind, &mut reply.stats);
        control_stats.merge(reply.stats);
        if downstream_missed {
            let scheduler_session_key = session_key.to_string();
            let _ = iteration_scheduler.execute_runtime("binary-prefix-rollback", move |runtime| {
                runtime
                    .drop_session_timed(&scheduler_session_key)
                    .map(|_| ())
                    .map_err(|error| openai_frontend::OpenAiError::backend(format!("{error:#}")))
            });
        }
    }
    let mut attrs = binary_message_attrs(config, session_id, &message);
    attrs.insert("skippy.kv.control_hit".to_string(), json!(local.hit));
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(elapsed_ms(control_started)),
    );
    telemetry.emit_debug("stage.binary_prefix_cache_control", attrs);
    send_reply_ack_with_stats(upstream, control_stats).context("prefix cache control ack")
}
