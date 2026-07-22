use super::async_forwarder::AsyncForwarder;
use super::reply::drain_deferred_prefill_replies;
use super::reply::send_stage_reply;
use super::reply::{configure_prediction_return_stream, reply_window_for_message};
use super::summary::BinaryMessageObservation;
use super::summary::BinaryRequestSummary;
use super::telemetry::UpstreamReplyWriteSpan;
use super::telemetry::{
    emit_upstream_reply_write_span, insert_runtime_session_stats, record_prefill_edge_transport,
    record_verify_window_timing,
};
use crate::binary_transport::BinaryStageExecutionOptions;
use crate::binary_transport::DecodeFrameBatcher;
use crate::binary_transport::WireCondition;
use crate::binary_transport::binary_kv::accumulate_prefill_tokens;
use crate::binary_transport::binary_kv::add_binary_record_stats;
use crate::binary_transport::binary_kv::emit_binary_proactive_eviction;
use crate::binary_transport::binary_kv::maybe_lookup_binary_prefill;
use crate::binary_transport::binary_kv::maybe_prefix_cache_control;
use crate::binary_transport::binary_kv::maybe_record_binary_full_prefill;
use crate::binary_transport::binary_kv::maybe_record_binary_prefill;
use crate::binary_transport::direct_return;
use crate::binary_transport::direct_return::PredictionReturnSinks;
use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::kv_eviction::binary_proactive_eviction_plan;
use crate::binary_transport::kv_eviction::evict_binary_resident_prefix_for_decode;
use crate::binary_transport::restore_prefill_decode::handle_binary_restore_prefill_decode_control;
use crate::binary_transport::run_binary_stage_message;
use crate::binary_transport::send_client_ready_hello_if_enabled;
use crate::binary_transport::stage_execution::binary_message_attrs;
use crate::binary_transport::stage_execution::binary_message_base;
use crate::binary_transport::stage_execution::binary_message_session_id;
use crate::binary_transport::stage_execution::decode_record_tokens_sideband;
use crate::binary_transport::stage_execution::elapsed_ms;
use crate::binary_transport::stage_execution::empty_activation_frame;
use crate::binary_transport::stage_execution::input_activation_frame;
use crate::binary_transport::stage_execution::insert_optional_unix_nanos;
use crate::binary_transport::stage_execution::is_decode_frame_batch_candidate;
use crate::binary_transport::stage_execution::nanos_delta_ms;
use crate::binary_transport::stage_execution::runtime_sampling_config;
use crate::binary_transport::stage_execution::split_native_mtp_reply;
use crate::binary_transport::stage_execution::stage_mask;
use crate::binary_transport::stage_execution::token_sideband_or_fill;
use crate::binary_transport::stage_output_activation_capacity;
use crate::binary_transport::write_stage_message_conditioned;
use crate::kv_integration::{KvStageIntegration, model_requires_recurrent_state};
use crate::runtime_state::RuntimeState;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_protocol::StageTopology;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireActivationDType;
use skippy_protocol::binary::WireMessageKind;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::read_stage_message;
use skippy_protocol::binary::recv_reply;
use skippy_protocol::binary::send_reply_ack;
use skippy_protocol::binary::send_reply_ack_with_stats;
use std::collections::BTreeMap;
use std::io;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_binary_connection(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    runtime: &Arc<Mutex<RuntimeState>>,
    decode_frame_batcher: &DecodeFrameBatcher,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    upstream: &mut TcpStream,
    mut downstream: Option<TcpStream>,
    activation_width: i32,
    wire_dtype: WireActivationDType,
    max_inflight: usize,
    reply_credit_limit: Option<usize>,
    async_prefill_forward: bool,
    downstream_wire_condition: WireCondition,
    downstream_connect_timeout_secs: u64,
    native_mtp_enabled: bool,
    prediction_return_sinks: &PredictionReturnSinks,
    first_message: StageWireMessage,
) -> Result<()> {
    if let Some(downstream) = downstream.as_mut() {
        send_client_ready_hello_if_enabled(&mut *downstream)
            .context("send downstream client ready hello")?;
        skippy_protocol::binary::recv_ready(&mut *downstream)
            .context("downstream binary stage did not become ready")?;
    }

    let connection_session_id = BINARY_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let positional_speculation_supported = !model_requires_recurrent_state(config);
    let max_deferred_prefill_replies =
        reply_credit_limit.unwrap_or_else(|| max_inflight.saturating_sub(1));
    let mut pending_prefill_replies = 0usize;
    let mut pending_reply_stats = StageReplyStats::default();
    let mut request_summary = BinaryRequestSummary::default();
    let mut accumulated_prefill_tokens: BTreeMap<String, Vec<i32>> = BTreeMap::new();
    let mut prediction_return_streams: BTreeMap<(u64, u64), TcpStream> = BTreeMap::new();
    let mut next_message = Some(first_message);
    let mut async_forwarder = if async_prefill_forward || max_inflight > 1 {
        downstream
            .as_ref()
            .map(|downstream| {
                AsyncForwarder::new(downstream, telemetry.clone(), max_inflight.max(1))
            })
            .transpose()
            .context("create async activation forwarder")?
    } else {
        None
    };

    loop {
        let recv_start_unix_nanos = now_unix_nanos() as u64;
        let recv_started = Instant::now();
        let mut message = if let Some(message) = next_message.take() {
            message
        } else {
            match read_stage_message(&mut *upstream, activation_width) {
                Ok(message) => message,
                Err(error)
                    if error.kind() == io::ErrorKind::UnexpectedEof
                        && pending_prefill_replies == 0
                        && request_summary.message_count == 0 =>
                {
                    return Ok(());
                }
                Err(error) => return Err(error).context("read binary stage message"),
            }
        };
        let recv_end_unix_nanos = now_unix_nanos() as u64;
        let recv_read_ms = elapsed_ms(recv_started);
        let message_start_unix_nanos = now_unix_nanos() as u64;
        let message_started = Instant::now();
        let session_id = binary_message_session_id(connection_session_id, &message);
        let session_key = session_id.to_string();
        if message.kind == WireMessageKind::VerifyWindow && !positional_speculation_supported {
            bail!(
                "stage-state v10 positional speculation requires an attention-only stage; {} contains recurrent state",
                config.stage_id
            );
        }
        if telemetry.is_debug_enabled() {
            let mut recv_attrs = binary_message_attrs(config, session_id, &message);
            recv_attrs.insert(
                "llama_stage.recv_start_unix_nanos".to_string(),
                json!(recv_start_unix_nanos),
            );
            recv_attrs.insert(
                "llama_stage.recv_end_unix_nanos".to_string(),
                json!(recv_end_unix_nanos),
            );
            recv_attrs.insert("llama_stage.recv_read_ms".to_string(), json!(recv_read_ms));
            recv_attrs.insert(
                "skippy.upstream_message_wait_ms".to_string(),
                json!(recv_read_ms),
            );
            recv_attrs.insert(
                "llama_stage.source_stage_index".to_string(),
                json!(message.state.source_stage_index),
            );
            recv_attrs.insert(
                "llama_stage.configured_upstream_stage_index".to_string(),
                json!(config.upstream.as_ref().map(|peer| peer.stage_index)),
            );
            recv_attrs.insert(
                "llama_stage.message_wire_bytes".to_string(),
                json!(message.estimated_wire_bytes()),
            );
            recv_attrs.insert(
                "skippy.activation_bytes".to_string(),
                json!(message.activation.len()),
            );
            telemetry.emit_debug_span(
                "stage.binary_recv",
                recv_attrs,
                recv_start_unix_nanos,
                recv_end_unix_nanos,
            );
        }

        if message.kind == WireMessageKind::Stop {
            if pending_prefill_replies != 0 {
                bail!("cannot stop with {pending_prefill_replies} deferred prefill replies");
            }
            let mut stop_stats = std::mem::take(&mut pending_reply_stats);
            request_summary.emit(telemetry, config, session_id);
            request_summary = BinaryRequestSummary::default();
            if let Some(downstream) = downstream.as_mut() {
                if let Some(forwarder) = async_forwarder.as_mut() {
                    forwarder
                        .flush()
                        .context("flush async forwards before stop")?;
                }
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward binary stop")?;
                let reply = recv_reply(&mut *downstream).context("stop downstream ACK")?;
                if reply.kind != WireReplyKind::Ack {
                    bail!("stop expected downstream ACK");
                }
                stop_stats.merge(reply.stats);
            }
            let reset_start_unix_nanos = now_unix_nanos() as u64;
            let reset_timer = Instant::now();
            let lock_timer = Instant::now();
            let mut runtime = runtime.lock().expect("runtime lock poisoned");
            let runtime_lock_wait_ms = elapsed_ms(lock_timer);
            let accumulated = std::mem::take(&mut accumulated_prefill_tokens);
            for (prefill_session_key, tokens) in accumulated {
                let record = maybe_record_binary_full_prefill(
                    config,
                    &mut runtime,
                    kv,
                    telemetry,
                    &prefill_session_key,
                    &message,
                    &tokens,
                );
                if record.recorded_pages > 0 {
                    stop_stats.kv_recorded_pages += record.recorded_pages as i64;
                    stop_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
                }
            }
            let drop_stats = runtime
                .drop_session_timed(&session_key)
                .context("reset binary stage session")?;
            drop(runtime);
            let reset_end_unix_nanos = now_unix_nanos() as u64;
            let mut reset_attrs = binary_message_attrs(config, session_id, &message);
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
            insert_runtime_session_stats(
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
            prediction_return_streams.remove(&(message.request_id, message.session_id));
            prediction_return_sinks.remove(message.request_id, message.session_id);
            send_reply_ack_with_stats(&mut *upstream, stop_stats).context("send stop ACK")?;
            continue;
        }

        if message.kind.is_session_control() {
            let mut control_stats = std::mem::take(&mut pending_reply_stats);
            if let Some(forwarder) = async_forwarder.as_mut() {
                forwarder
                    .flush()
                    .context("flush async forwards before session control")?;
            }
            drain_deferred_prefill_replies(
                downstream.as_mut(),
                &mut pending_prefill_replies,
                &mut control_stats,
            )
            .context("drain deferred replies before session control")?;
            {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                match message.kind {
                    WireMessageKind::TrimSession => runtime
                        .trim_session(&session_key, message.token_count.max(0) as u64)
                        .context("trim binary stage session")?,
                    _ => unreachable!("session control checked above"),
                }
            }
            if let Some(downstream) = downstream.as_mut() {
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward session control")?;
                let reply =
                    recv_reply(&mut *downstream).context("session control downstream ACK")?;
                if reply.kind != WireReplyKind::Ack {
                    bail!("session control expected downstream ACK");
                }
                control_stats.merge(reply.stats);
            }
            send_reply_ack_with_stats(&mut *upstream, control_stats)
                .context("session control ack")?;
            continue;
        }

        if message.kind.is_generation_control() {
            let mut generation_stats = std::mem::take(&mut pending_reply_stats);
            if let Some(forwarder) = async_forwarder.as_mut() {
                forwarder
                    .flush()
                    .context("flush async forwards before generation config")?;
            }
            drain_deferred_prefill_replies(
                downstream.as_mut(),
                &mut pending_prefill_replies,
                &mut generation_stats,
            )
            .context("drain deferred replies before generation config")?;
            if let Some(downstream) = downstream.as_mut() {
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward generation config")?;
                let reply =
                    recv_reply(&mut *downstream).context("generation config downstream ACK")?;
                if reply.kind != WireReplyKind::Ack {
                    bail!("generation config expected downstream ACK");
                }
                generation_stats.merge(reply.stats);
            } else {
                if let Some(metadata) = message.chat_sampling_metadata.as_deref() {
                    let sampling = runtime_sampling_config(message.sampling.as_ref());
                    let mut runtime = runtime.lock().expect("runtime lock poisoned");
                    runtime
                        .configure_chat_sampling(
                            &session_key,
                            metadata,
                            message.state.prompt_token_count.max(0) as u64,
                            sampling.as_ref(),
                        )
                        .context("configure binary stage generation")?;
                }
                configure_prediction_return_stream(
                    config,
                    topology,
                    message.request_id,
                    message.session_id,
                    wire_dtype,
                    downstream_connect_timeout_secs,
                    prediction_return_sinks,
                    &mut prediction_return_streams,
                );
            }
            send_reply_ack_with_stats(&mut *upstream, generation_stats)
                .context("generation config ack")?;
            continue;
        }

        if message.kind.is_prefix_cache_control() {
            let control_started = Instant::now();
            let mut control_stats = std::mem::take(&mut pending_reply_stats);
            if let Some(forwarder) = async_forwarder.as_mut() {
                forwarder
                    .flush()
                    .context("flush async forwards before prefix cache control")?;
            }
            drain_deferred_prefill_replies(
                downstream.as_mut(),
                &mut pending_prefill_replies,
                &mut control_stats,
            )
            .context("drain deferred replies before prefix cache control")?;
            if message.kind == WireMessageKind::TryRestorePrefillDecode {
                handle_binary_restore_prefill_decode_control(
                    config,
                    topology,
                    runtime,
                    kv,
                    telemetry,
                    &session_key,
                    session_id,
                    message,
                    downstream.as_mut(),
                    wire_dtype,
                    downstream_wire_condition,
                    activation_width,
                    control_started,
                    control_stats,
                    prediction_return_sinks,
                    &mut prediction_return_streams,
                    downstream_connect_timeout_secs,
                    native_mtp_enabled,
                )
                .context("handle restore-prefill-decode control")?;
                continue;
            }
            let token_ids = token_sideband_or_fill(&message)
                .context("read prefix cache control token sideband")?;
            let local = maybe_prefix_cache_control(
                config,
                runtime,
                kv,
                telemetry,
                &session_key,
                &message,
                &token_ids,
            );
            control_stats.merge(local.stats);
            if local.hit
                && let Some(downstream) = downstream.as_mut()
            {
                write_stage_message_conditioned(
                    &mut *downstream,
                    &message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward prefix cache control")?;
                let reply = recv_reply(&mut *downstream).context("prefix cache downstream ACK")?;
                if reply.kind != WireReplyKind::Ack {
                    bail!("prefix cache control expected downstream ACK");
                }
                let downstream_missed = message.kind == WireMessageKind::TryRestorePrefill
                    && (reply.stats.kv_lookup_misses > 0
                        || reply.stats.kv_lookup_errors > 0
                        || reply.stats.kv_lookup_hits == 0);
                control_stats.merge(reply.stats);
                if downstream_missed {
                    let mut runtime = runtime.lock().expect("runtime lock poisoned");
                    let _ = runtime.drop_session_timed(&session_key);
                }
            }
            let mut attrs = binary_message_attrs(config, session_id, &message);
            attrs.insert("skippy.kv.control_hit".to_string(), json!(local.hit));
            attrs.insert(
                "llama_stage.elapsed_ms".to_string(),
                json!(elapsed_ms(control_started)),
            );
            telemetry.emit_debug("stage.binary_prefix_cache_control", attrs);
            send_reply_ack_with_stats(&mut *upstream, control_stats)
                .context("prefix cache control ack")?;
            continue;
        }

        if message.kind == WireMessageKind::StateImport {
            bail!("binary state import is no longer supported by the skippy runtime ABI");
        }

        if message.kind == WireMessageKind::StateExport {
            bail!("binary state export is no longer supported by the skippy runtime ABI");
        }

        if !message.state.matches_kind(message.kind) {
            bail!("binary stage state does not match message kind");
        }

        let requires_predicted = message.kind.requires_predicted_reply();
        let early_prefill_ack = message.kind.is_prefill() && !requires_predicted;
        let mut upstream_reply_start_unix_nanos = None;
        let mut upstream_reply_end_unix_nanos = None;
        let mut early_reply_ms = 0.0;
        if early_prefill_ack {
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos = Some(reply_start_unix_nanos);
            let reply_started = Instant::now();
            send_reply_ack(&mut *upstream).context("early prefill ack")?;
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            early_reply_ms = elapsed_ms(reply_started);
        }

        let token_ids = token_sideband_or_fill(&message)?;
        let mut session_auto_align_count = 0usize;
        let mut session_auto_align_ms = 0.0;
        let mut session_auto_align_trimmed_tokens = 0u64;
        if let Some(target_token_count) = message.authoritative_session_position() {
            let align_started = Instant::now();
            let align = {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                runtime
                    .align_session_to_token_count_if_ahead(&session_key, target_token_count)
                    .context("auto-align binary stage session")?
            };
            if let Some(align) = align {
                let align_ms = elapsed_ms(align_started);
                session_auto_align_count = 1;
                session_auto_align_ms = align_ms;
                session_auto_align_trimmed_tokens = align
                    .before_token_count
                    .saturating_sub(align.after_token_count);
                let mut attrs = binary_message_attrs(config, session_id, &message);
                attrs.insert(
                    "llama_stage.session_auto_align_before_tokens".to_string(),
                    json!(align.before_token_count),
                );
                attrs.insert(
                    "llama_stage.session_auto_align_after_tokens".to_string(),
                    json!(align.after_token_count),
                );
                attrs.insert("llama_stage.elapsed_ms".to_string(), json!(align_ms));
                telemetry.emit_debug("stage.binary_session_auto_align", attrs);
            }
        }
        if message.kind.is_prefill() {
            accumulate_prefill_tokens(
                &mut accumulated_prefill_tokens,
                &session_key,
                message.pos_start.max(0) as usize,
                &token_ids,
            );
        }
        let mut message_reply_stats = StageReplyStats::default();
        let lookup_result = maybe_lookup_binary_prefill(
            config,
            runtime,
            kv,
            telemetry,
            &session_key,
            &message,
            &token_ids,
            activation_width,
        );
        message_reply_stats.merge(lookup_result.stats);
        let restored_prefill =
            lookup_result.restored_tokens >= token_ids.len() && !token_ids.is_empty();
        let executable_token_ids = if message.kind.is_prefill()
            && lookup_result.restored_tokens > 0
            && lookup_result.restored_tokens < token_ids.len()
            && config.layer_start == 0
        {
            &token_ids[lookup_result.restored_tokens..]
        } else {
            &token_ids
        };
        let compute_start_unix_nanos: u64;
        let compute_end_unix_nanos: u64;
        let mut input_activation_decode_ms = 0.0;
        let mut runtime_lock_wait_ms = 0.0;
        let mut runtime_lock_hold_ms = 0.0;
        let mut runtime_lock_acquires = 0usize;
        let mut runtime_sessions_before = None;
        let mut runtime_sessions_after = None;
        let mut decode_batch_size = 1usize;
        let mut decode_batch_wait_ms = 0.0;
        let input_activation_bytes = message.activation.len();
        let mut proactive_eviction = None;
        let (predicted_token, mut predicted_tokens, output, compute_ms) = if restored_prefill {
            let now = now_unix_nanos() as u64;
            compute_start_unix_nanos = now;
            compute_end_unix_nanos = now;
            (
                message.state.current_token,
                Vec::new(),
                lookup_result
                    .activation
                    .clone()
                    .unwrap_or_else(|| empty_activation_frame(config, &message)),
                0.0,
            )
        } else {
            let input_decode_started = Instant::now();
            let input = input_activation_frame(config, topology, &mut message, activation_width)?;
            input_activation_decode_ms = if input_activation_bytes == 0 {
                0.0
            } else {
                elapsed_ms(input_decode_started)
            };
            compute_start_unix_nanos = now_unix_nanos() as u64;
            let compute_started = Instant::now();
            let use_decode_frame_batch =
                is_decode_frame_batch_candidate(config, &message, executable_token_ids);
            let result = if use_decode_frame_batch {
                let token_id = executable_token_ids
                    .first()
                    .copied()
                    .unwrap_or(message.state.current_token);
                let sampling = runtime_sampling_config(message.sampling.as_ref());
                let target_token_count =
                    message.authoritative_session_position().ok_or_else(|| {
                        anyhow::anyhow!("batched decode frame has no authoritative position")
                    })?;
                let outcome = decode_frame_batcher
                    .decode(
                        &session_key,
                        target_token_count,
                        token_id,
                        sampling.as_ref(),
                        input,
                    )
                    .context("execute batched binary decode frame")?;
                runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
                runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
                runtime_lock_acquires = 1;
                decode_batch_size = outcome.batch_size;
                decode_batch_wait_ms = outcome.batch_wait_ms;
                (outcome.predicted, Vec::new(), outcome.output)
            } else {
                let lock_started = Instant::now();
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                runtime_lock_wait_ms = elapsed_ms(lock_started);
                runtime_lock_acquires = 1;
                let lock_hold_started = Instant::now();
                runtime_sessions_before = Some(runtime.session_stats());
                let eviction_plan = binary_proactive_eviction_plan(
                    message.kind,
                    restored_prefill,
                    executable_token_ids.len(),
                    (message.state.prompt_token_count.max(0) as usize)
                        .saturating_sub(message.pos_start.max(0) as usize),
                );
                if eviction_plan.required {
                    proactive_eviction = Some(evict_binary_resident_prefix_for_decode(
                        &mut runtime,
                        kv,
                        &session_key,
                        eviction_plan,
                    )?);
                }
                let result = run_binary_stage_message(
                    &mut runtime,
                    &session_key,
                    &message,
                    executable_token_ids,
                    input.as_ref(),
                    BinaryStageExecutionOptions::new(
                        message.kind == WireMessageKind::PrefillFinalEmbd && downstream.is_none(),
                        stage_output_activation_capacity(
                            config,
                            message.token_count,
                            activation_width,
                        )?,
                        native_mtp_enabled,
                    ),
                )
                .context("execute binary stage message")?;
                runtime_sessions_after = Some(runtime.session_stats());
                runtime_lock_hold_ms = elapsed_ms(lock_hold_started);
                result
            };
            let compute_ms = elapsed_ms(compute_started);
            compute_end_unix_nanos = now_unix_nanos() as u64;
            (result.0, result.1, result.2, compute_ms)
        };
        if telemetry.is_debug_enabled() {
            let mut decode_attrs = binary_message_attrs(config, session_id, &message);
            decode_attrs.insert(
                "skippy.output_activation_bytes".to_string(),
                json!(output.payload.len()),
            );
            decode_attrs.insert("skippy.compute_ms".to_string(), json!(compute_ms));
            decode_attrs.insert(
                "llama_stage.input_activation_decode_ms".to_string(),
                json!(input_activation_decode_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_size".to_string(),
                json!(decode_batch_size),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_wait_ms".to_string(),
                json!(decode_batch_wait_ms),
            );
            if let Some(stats) = runtime_sessions_before.as_ref() {
                insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = runtime_sessions_after.as_ref() {
                insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            if let Some(eviction) = proactive_eviction.as_ref() {
                eviction.insert_attrs(&mut decode_attrs);
            }
            decode_attrs.insert(
                "skippy.kv.restored_prefill".to_string(),
                json!(restored_prefill),
            );
            decode_attrs.insert(
                "llama_stage.compute_start_unix_nanos".to_string(),
                json!(compute_start_unix_nanos),
            );
            decode_attrs.insert(
                "llama_stage.compute_end_unix_nanos".to_string(),
                json!(compute_end_unix_nanos),
            );
            telemetry.emit_debug_span(
                "stage.binary_llama_decode",
                decode_attrs,
                compute_start_unix_nanos,
                compute_end_unix_nanos,
            );
        }
        if let Some(eviction) = proactive_eviction {
            emit_binary_proactive_eviction(telemetry, &eviction);
        }

        if message.kind.is_prefill() && !restored_prefill {
            let record = if let Some(tokens) = accumulated_prefill_tokens.get(&session_key).cloned()
            {
                let mut runtime = runtime.lock().expect("runtime lock poisoned");
                let mut record = maybe_record_binary_full_prefill(
                    config,
                    &mut runtime,
                    kv,
                    telemetry,
                    &session_key,
                    &message,
                    &tokens,
                );
                drop(runtime);
                if let Some(kv) = kv
                    && config.downstream.is_some()
                {
                    let base = binary_message_base(config, &session_key, &message);
                    if let Some(activation) = kv.record_resident_activation(
                        config,
                        &base,
                        0,
                        &tokens,
                        activation_width,
                        &output,
                    ) {
                        record.recorded_activations = record.recorded_activations.saturating_add(1);
                        record.recorded_activation_bytes = record
                            .recorded_activation_bytes
                            .saturating_add(activation.payload_bytes as u64);
                        record.evicted_activation_entries = record
                            .evicted_activation_entries
                            .saturating_add(activation.evicted_entries);
                        record.evicted_activation_bytes = record
                            .evicted_activation_bytes
                            .saturating_add(activation.evicted_bytes);
                    }
                }
                record
            } else {
                maybe_record_binary_prefill(
                    config,
                    runtime,
                    kv,
                    telemetry,
                    &session_key,
                    &message,
                    &token_ids,
                    lookup_result.restored_tokens as u64,
                    activation_width,
                    Some(&output),
                )
            };
            if record.recorded_pages > 0 {
                message_reply_stats.kv_recorded_pages += record.recorded_pages as i64;
                message_reply_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
            }
            if record.recorded_activations > 0 {
                message_reply_stats.kv_recorded_bytes = message_reply_stats
                    .kv_recorded_bytes
                    .saturating_add(record.recorded_activation_bytes as i64);
            }
        }

        if let Some(full_prompt_tokens) = decode_record_tokens_sideband(&message) {
            let mut runtime = runtime.lock().expect("runtime lock poisoned");
            let record = maybe_record_binary_full_prefill(
                config,
                &mut runtime,
                kv,
                telemetry,
                &session_key,
                &message,
                full_prompt_tokens,
            );
            drop(runtime);
            add_binary_record_stats(&mut message_reply_stats, config, &record);
        }

        let mut forward_write_ms = 0.0;
        let mut forward_activation_encode_ms = 0.0;
        let mut forward_activation_bytes = 0usize;
        let mut downstream_wait_ms = 0.0;
        let mut upstream_reply_ms = early_reply_ms;
        let mut forward_write_start_unix_nanos = None;
        let mut forward_write_end_unix_nanos = None;
        let mut downstream_wait_start_unix_nanos = None;
        let mut downstream_wait_end_unix_nanos = None;
        let mut forward_mode = "none";
        let pending_prefill_replies_before = pending_prefill_replies;
        let mut credit_wait_count = 0usize;
        let mut deferred_prefill_replies_drained = 0usize;

        if let Some(downstream) = downstream.as_mut() {
            if output.payload.is_empty() {
                bail!("stage has downstream but produced an empty activation payload");
            }
            let forwarded = forwarded_stage_message_timed(
                config,
                &message,
                &output,
                wire_dtype,
                activation_width,
            )?;
            forward_activation_encode_ms += forwarded.activation_encode_ms;
            forward_activation_bytes = forwarded.message.activation.len();
            let mut downstream_write_attrs = BTreeMap::new();
            if telemetry.is_debug_enabled() {
                downstream_write_attrs = binary_message_attrs(config, session_id, &message);
                downstream_write_attrs.insert(
                    "llama_stage.forward_activation_bytes".to_string(),
                    json!(forward_activation_bytes),
                );
                downstream_write_attrs.insert(
                    "llama_stage.activation_encode_ms".to_string(),
                    json!(forwarded.activation_encode_ms),
                );
                downstream_write_attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(output.payload.len()),
                );
            }
            let forward_start_unix_nanos = now_unix_nanos() as u64;
            forward_write_start_unix_nanos = Some(forward_start_unix_nanos);
            let forward_started = Instant::now();
            let async_verify_forward =
                message.kind == WireMessageKind::VerifyWindow && max_inflight > 1;
            if (async_prefill_forward && early_prefill_ack && max_deferred_prefill_replies > 0)
                || async_verify_forward
            {
                forward_mode = "async_enqueue";
                if telemetry.is_debug_enabled() {
                    downstream_write_attrs.insert(
                        "llama_stage.forward_mode".to_string(),
                        json!("async_writer"),
                    );
                }
                let forwarder = async_forwarder
                    .as_mut()
                    .context("missing async activation forwarder")?;
                forwarder
                    .send(
                        forwarded.message,
                        wire_dtype,
                        downstream_wire_condition,
                        downstream_write_attrs,
                    )
                    .context("queue async activation frame downstream")?;
            } else {
                forward_mode = "sync_write";
                if telemetry.is_debug_enabled() {
                    downstream_write_attrs
                        .insert("llama_stage.forward_mode".to_string(), json!("sync_write"));
                }
                if let Some(forwarder) = async_forwarder.as_mut() {
                    forwarder.flush().context("flush async activation frames")?;
                }
                let downstream_write_start_unix_nanos = now_unix_nanos() as u64;
                let downstream_write_started = Instant::now();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    wire_dtype,
                    downstream_wire_condition,
                )
                .context("forward activation frame downstream")?;
                let downstream_write_end_unix_nanos = now_unix_nanos() as u64;
                if telemetry.is_debug_enabled() {
                    downstream_write_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(elapsed_ms(downstream_write_started)),
                    );
                    telemetry.emit_debug_span(
                        "stage.binary_downstream_write",
                        downstream_write_attrs,
                        downstream_write_start_unix_nanos,
                        downstream_write_end_unix_nanos,
                    );
                }
            }
            forward_write_end_unix_nanos = Some(now_unix_nanos() as u64);
            forward_write_ms += elapsed_ms(forward_started);

            if requires_predicted {
                while pending_prefill_replies > 0 {
                    let wait_start_unix_nanos = now_unix_nanos() as u64;
                    downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                    let wait_started = Instant::now();
                    let reply = recv_reply(&mut *downstream)
                        .context("drain deferred downstream prefill reply")?;
                    downstream_wait_ms += elapsed_ms(wait_started);
                    if reply.kind != WireReplyKind::Ack {
                        bail!("expected deferred downstream ACK");
                    }
                    pending_reply_stats.merge(reply.stats);
                    pending_prefill_replies -= 1;
                    deferred_prefill_replies_drained += 1;
                }
            } else if max_deferred_prefill_replies == 0 {
                let wait_start_unix_nanos = now_unix_nanos() as u64;
                downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                let wait_started = Instant::now();
                let reply = recv_reply(&mut *downstream).context("downstream ACK")?;
                downstream_wait_end_unix_nanos = Some(now_unix_nanos() as u64);
                downstream_wait_ms += elapsed_ms(wait_started);
                if reply.kind != WireReplyKind::Ack {
                    bail!("expected downstream ACK");
                }
                message_reply_stats.merge(reply.stats);
                if !early_prefill_ack {
                    let reply_start_unix_nanos = now_unix_nanos() as u64;
                    upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
                    let reply_started = Instant::now();
                    send_reply_ack_with_stats(&mut *upstream, message_reply_stats)
                        .context("relay ACK")?;
                    upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
                    let reply_write_ms = elapsed_ms(reply_started);
                    upstream_reply_ms += reply_write_ms;
                    emit_upstream_reply_write_span(
                        telemetry,
                        config,
                        session_id,
                        &message,
                        UpstreamReplyWriteSpan {
                            reply_kind: WireReplyKind::Ack,
                            predicted_token_count: 0,
                            start_unix_nanos: reply_start_unix_nanos,
                            end_unix_nanos: upstream_reply_end_unix_nanos
                                .unwrap_or(reply_start_unix_nanos),
                            write_ms: reply_write_ms,
                        },
                    );
                } else {
                    pending_reply_stats.merge(message_reply_stats);
                }
            } else {
                while pending_prefill_replies >= max_deferred_prefill_replies {
                    credit_wait_count += 1;
                    let wait_start_unix_nanos = now_unix_nanos() as u64;
                    downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                    let wait_started = Instant::now();
                    let reply =
                        recv_reply(&mut *downstream).context("bounded-credit downstream ACK")?;
                    downstream_wait_end_unix_nanos = Some(now_unix_nanos() as u64);
                    downstream_wait_ms += elapsed_ms(wait_started);
                    if reply.kind != WireReplyKind::Ack {
                        bail!("expected downstream ACK while enforcing credit");
                    }
                    pending_reply_stats.merge(reply.stats);
                    pending_prefill_replies -= 1;
                    deferred_prefill_replies_drained += 1;
                }
                pending_prefill_replies += 1;
                if !early_prefill_ack {
                    let reply_start_unix_nanos = now_unix_nanos() as u64;
                    upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
                    let reply_started = Instant::now();
                    send_reply_ack_with_stats(&mut *upstream, message_reply_stats)
                        .context("deferred relay ACK")?;
                    upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
                    let reply_write_ms = elapsed_ms(reply_started);
                    upstream_reply_ms += reply_write_ms;
                    emit_upstream_reply_write_span(
                        telemetry,
                        config,
                        session_id,
                        &message,
                        UpstreamReplyWriteSpan {
                            reply_kind: WireReplyKind::Ack,
                            predicted_token_count: 0,
                            start_unix_nanos: reply_start_unix_nanos,
                            end_unix_nanos: upstream_reply_end_unix_nanos
                                .unwrap_or(reply_start_unix_nanos),
                            write_ms: reply_write_ms,
                        },
                    );
                } else {
                    pending_reply_stats.merge(message_reply_stats);
                }
            }
        } else if requires_predicted {
            record_prefill_edge_transport(
                &mut message_reply_stats,
                config,
                &message,
                forward_write_ms,
                downstream_wait_ms,
                forward_activation_bytes,
            );
            message_reply_stats.merge(pending_reply_stats);
            pending_reply_stats = StageReplyStats::default();
            record_verify_window_timing(
                &mut message_reply_stats,
                &message,
                compute_ms,
                forward_write_ms,
                downstream_wait_ms,
            );
            let reply_kind = if message.kind == WireMessageKind::VerifyWindow {
                WireReplyKind::PredictedTokens
            } else {
                WireReplyKind::PredictedToken
            };
            let native_mtp_draft = split_native_mtp_reply(&message, &mut predicted_tokens)?;
            let predicted_token_count = if message.kind == WireMessageKind::VerifyWindow {
                predicted_tokens.len()
            } else {
                predicted_tokens.len().max(1)
            };
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
            let reply_started = Instant::now();
            let reply_window = reply_window_for_message(&message);
            let reply = StageReply {
                kind: reply_kind,
                predicted: predicted_token,
                predicted_tokens,
                native_mtp_draft,
                window: reply_window,
                stats: message_reply_stats,
            };
            if let Some(return_stream) =
                prediction_return_streams.get_mut(&(message.request_id, message.session_id))
            {
                direct_return::send_direct_prediction_return(return_stream, reply)
                    .context("send direct predicted reply")?;
            } else {
                send_stage_reply(&mut *upstream, reply)
                    .context("send fallback upstream predicted reply")?;
            }
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            let reply_write_ms = elapsed_ms(reply_started);
            upstream_reply_ms += reply_write_ms;
            emit_upstream_reply_write_span(
                telemetry,
                config,
                session_id,
                &message,
                UpstreamReplyWriteSpan {
                    reply_kind,
                    predicted_token_count,
                    start_unix_nanos: reply_start_unix_nanos,
                    end_unix_nanos: upstream_reply_end_unix_nanos.unwrap_or(reply_start_unix_nanos),
                    write_ms: reply_write_ms,
                },
            );
        } else if !early_prefill_ack {
            record_prefill_edge_transport(
                &mut message_reply_stats,
                config,
                &message,
                forward_write_ms,
                downstream_wait_ms,
                forward_activation_bytes,
            );
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
            let reply_started = Instant::now();
            send_reply_ack_with_stats(&mut *upstream, message_reply_stats).context("send ACK")?;
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            let reply_write_ms = elapsed_ms(reply_started);
            upstream_reply_ms += reply_write_ms;
            emit_upstream_reply_write_span(
                telemetry,
                config,
                session_id,
                &message,
                UpstreamReplyWriteSpan {
                    reply_kind: WireReplyKind::Ack,
                    predicted_token_count: 0,
                    start_unix_nanos: reply_start_unix_nanos,
                    end_unix_nanos: upstream_reply_end_unix_nanos.unwrap_or(reply_start_unix_nanos),
                    write_ms: reply_write_ms,
                },
            );
        } else {
            record_prefill_edge_transport(
                &mut message_reply_stats,
                config,
                &message,
                forward_write_ms,
                downstream_wait_ms,
                forward_activation_bytes,
            );
            pending_reply_stats.merge(message_reply_stats);
        }

        let message_end_unix_nanos = now_unix_nanos() as u64;
        let message_elapsed_ms = elapsed_ms(message_started);
        let verify_window_pre_compute_ms = if message.kind == WireMessageKind::VerifyWindow {
            nanos_delta_ms(message_start_unix_nanos, compute_start_unix_nanos)
        } else {
            0.0
        };
        let verify_window_post_compute_ms = if message.kind == WireMessageKind::VerifyWindow {
            nanos_delta_ms(compute_end_unix_nanos, message_end_unix_nanos)
        } else {
            0.0
        };
        let verify_window_pre_reply_ms = if message.kind == WireMessageKind::VerifyWindow {
            upstream_reply_start_unix_nanos
                .map(|reply_start| nanos_delta_ms(compute_end_unix_nanos, reply_start))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let verify_window_after_reply_ms = if message.kind == WireMessageKind::VerifyWindow {
            upstream_reply_end_unix_nanos
                .map(|reply_end| nanos_delta_ms(reply_end, message_end_unix_nanos))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        request_summary.observe(BinaryMessageObservation {
            config,
            message: &message,
            reply_stats: message_reply_stats,
            compute_ms,
            forward_write_ms,
            downstream_wait_ms,
            upstream_reply_ms,
            message_elapsed_ms,
            input_activation_bytes,
            output_activation_bytes: output.payload.len(),
            input_activation_decode_ms,
            forward_activation_encode_ms,
            runtime_lock_hold_ms,
            prefill_credit_limit: max_deferred_prefill_replies,
            pending_prefill_replies_before,
            pending_prefill_replies_after: pending_prefill_replies,
            credit_wait_count,
            deferred_prefill_replies_drained,
            session_auto_align_count,
            session_auto_align_ms,
            session_auto_align_trimmed_tokens,
            verify_window_pre_compute_ms,
            verify_window_post_compute_ms,
            verify_window_pre_reply_ms,
            verify_window_after_reply_ms,
            upstream_message_wait_ms: recv_read_ms,
        });

        if telemetry.is_debug_enabled() {
            let mut timing_attrs = binary_message_attrs(config, session_id, &message);
            timing_attrs.insert(
                "llama_stage.message_start_unix_nanos".to_string(),
                json!(message_start_unix_nanos),
            );
            timing_attrs.insert(
                "llama_stage.message_end_unix_nanos".to_string(),
                json!(message_end_unix_nanos),
            );
            timing_attrs.insert(
                "llama_stage.compute_start_unix_nanos".to_string(),
                json!(compute_start_unix_nanos),
            );
            timing_attrs.insert(
                "llama_stage.compute_end_unix_nanos".to_string(),
                json!(compute_end_unix_nanos),
            );
            timing_attrs.insert("llama_stage.compute_ms".to_string(), json!(compute_ms));
            timing_attrs.insert("llama_stage.recv_read_ms".to_string(), json!(recv_read_ms));
            timing_attrs.insert(
                "skippy.upstream_message_wait_ms".to_string(),
                json!(recv_read_ms),
            );
            timing_attrs.insert(
                "llama_stage.input_activation_decode_ms".to_string(),
                json!(input_activation_decode_ms),
            );
            timing_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            timing_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            timing_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            if let Some(stats) = runtime_sessions_before.as_ref() {
                insert_runtime_session_stats(
                    &mut timing_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = runtime_sessions_after.as_ref() {
                insert_runtime_session_stats(
                    &mut timing_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            timing_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(forward_write_ms),
            );
            timing_attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(forward_activation_encode_ms),
            );
            timing_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(downstream_wait_ms),
            );
            timing_attrs.insert("skippy.compute_ms".to_string(), json!(compute_ms));
            timing_attrs.insert(
                "skippy.forward_write_ms".to_string(),
                json!(forward_write_ms),
            );
            timing_attrs.insert(
                "skippy.downstream_wait_ms".to_string(),
                json!(downstream_wait_ms),
            );
            timing_attrs.insert(
                "skippy.upstream_reply_ms".to_string(),
                json!(upstream_reply_ms),
            );
            timing_attrs.insert("llama_stage.forward_mode".to_string(), json!(forward_mode));
            insert_optional_unix_nanos(
                &mut timing_attrs,
                "llama_stage.forward_write_start_unix_nanos",
                forward_write_start_unix_nanos,
            );
            insert_optional_unix_nanos(
                &mut timing_attrs,
                "llama_stage.forward_write_end_unix_nanos",
                forward_write_end_unix_nanos,
            );
            insert_optional_unix_nanos(
                &mut timing_attrs,
                "llama_stage.downstream_wait_start_unix_nanos",
                downstream_wait_start_unix_nanos,
            );
            insert_optional_unix_nanos(
                &mut timing_attrs,
                "llama_stage.downstream_wait_end_unix_nanos",
                downstream_wait_end_unix_nanos,
            );
            insert_optional_unix_nanos(
                &mut timing_attrs,
                "llama_stage.upstream_reply_start_unix_nanos",
                upstream_reply_start_unix_nanos,
            );
            insert_optional_unix_nanos(
                &mut timing_attrs,
                "llama_stage.upstream_reply_end_unix_nanos",
                upstream_reply_end_unix_nanos,
            );
            timing_attrs.insert(
                "skippy.message_elapsed_ms".to_string(),
                json!(message_elapsed_ms),
            );
            timing_attrs.insert(
                "skippy.input_activation_bytes".to_string(),
                json!(input_activation_bytes),
            );
            timing_attrs.insert(
                "skippy.output_activation_bytes".to_string(),
                json!(output.payload.len()),
            );
            timing_attrs.insert(
                "skippy.prefill_credit_limit".to_string(),
                json!(max_deferred_prefill_replies),
            );
            timing_attrs.insert(
                "skippy.prefill_pending_replies_before".to_string(),
                json!(pending_prefill_replies_before),
            );
            timing_attrs.insert(
                "skippy.prefill_pending_replies_after".to_string(),
                json!(pending_prefill_replies),
            );
            timing_attrs.insert(
                "skippy.prefill_credit_wait_count".to_string(),
                json!(credit_wait_count),
            );
            timing_attrs.insert(
                "skippy.prefill_deferred_replies_drained".to_string(),
                json!(deferred_prefill_replies_drained),
            );
            telemetry.emit_debug_span(
                "stage.binary_message_timing",
                timing_attrs,
                message_start_unix_nanos,
                message_end_unix_nanos,
            );
        }
    }
}
