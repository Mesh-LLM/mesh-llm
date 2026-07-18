use std::{
    collections::BTreeMap,
    net::TcpStream,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_protocol::{
    StageConfig, StageTopology,
    binary::{
        StageReply, StageReplyStats, StageStateHeader, StageWireMessage, WireActivationDType,
        WireMessageKind, WireReplyKind,
    },
};

use crate::{
    kv_integration::KvStageIntegration, runtime_state::RuntimeState, telemetry::Telemetry,
};

use super::{
    BinaryStageExecutionOptions, WireCondition, direct_return, forwarding,
    forwarding::forwarded_stage_message_timed, wire::write_stage_message_conditioned,
};
use super::{
    binary_kv::{
        add_binary_record_stats, emit_binary_proactive_eviction, maybe_prefix_cache_control,
        maybe_record_binary_full_prefill,
    },
    binary_messaging::reply::configure_prediction_return_stream,
    direct_return::PredictionReturnSinks,
    kv_eviction::{
        BinaryProactiveEviction, BinaryProactiveEvictionPlan,
        evict_binary_resident_prefix_for_decode,
    },
    stage_execution::{
        binary_message_attrs, elapsed_ms, input_activation_frame, run_binary_stage_message,
        runtime_sampling_config, stage_output_activation_capacity,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RestorePrefillDecodeRoute {
    DirectMiss,
    ForwardHit,
    DirectHit,
}

fn restore_prefill_decode_route(
    local_hit: bool,
    has_downstream: bool,
) -> RestorePrefillDecodeRoute {
    match (local_hit, has_downstream) {
        (false, _) => RestorePrefillDecodeRoute::DirectMiss,
        (true, true) => RestorePrefillDecodeRoute::ForwardHit,
        (true, false) => RestorePrefillDecodeRoute::DirectHit,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_binary_restore_prefill_decode_control(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    runtime: &Arc<Mutex<RuntimeState>>,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_id: &str,
    wire_session_id: u64,
    mut message: StageWireMessage,
    downstream: Option<&mut TcpStream>,
    wire_dtype: WireActivationDType,
    downstream_wire_condition: WireCondition,
    activation_width: i32,
    control_started: Instant,
    mut control_stats: StageReplyStats,
    prediction_return_sinks: &PredictionReturnSinks,
    prediction_return_streams: &mut BTreeMap<(u64, u64), TcpStream>,
    downstream_connect_timeout_secs: u64,
    native_mtp_enabled: bool,
) -> Result<()> {
    let has_downstream = downstream.is_some();
    if !has_downstream {
        configure_prediction_return_stream(
            config,
            topology,
            message.request_id,
            message.session_id,
            wire_dtype,
            downstream_connect_timeout_secs,
            prediction_return_sinks,
            prediction_return_streams,
        );
    }

    let (prefix_tokens, current_token) = restore_decode_sideband(&message)?;
    let local = maybe_prefix_cache_control(
        config,
        runtime,
        kv,
        telemetry,
        session_id,
        &message,
        prefix_tokens,
    );
    control_stats.merge(local.stats);
    let route = restore_prefill_decode_route(local.hit, has_downstream);
    if route == RestorePrefillDecodeRoute::DirectMiss {
        emit_restore_decode_control(
            config,
            telemetry,
            wire_session_id,
            &message,
            control_started,
            false,
            None,
        );
        send_restore_decode_direct_reply(
            config,
            topology,
            &message,
            wire_dtype,
            downstream_connect_timeout_secs,
            prediction_return_streams,
            StageReply {
                kind: WireReplyKind::Ack,
                predicted: 0,
                predicted_tokens: Vec::new(),
                native_mtp_draft: None,
                window: Default::default(),
                stats: control_stats,
            },
        )
        .context("send restore-decode miss direct ACK")?;
        return Ok(());
    }

    let input = input_activation_frame(config, topology, &mut message, activation_width)?;
    let decode_message = restore_prefill_decode_as_decode_message(&message, current_token);
    let compute_started = Instant::now();
    let (predicted_token, output, runtime_lock_wait_ms, runtime_lock_hold_ms, proactive_eviction) = {
        let lock_started = Instant::now();
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        let runtime_lock_wait_ms = elapsed_ms(lock_started);
        let lock_hold_started = Instant::now();
        if let Some(metadata) = message.chat_sampling_metadata.as_deref() {
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            runtime
                .configure_chat_sampling(
                    session_id,
                    metadata,
                    message.state.prompt_token_count.max(0) as u64,
                    sampling.as_ref(),
                )
                .context("configure restore-decode chat sampling")?;
        }
        let proactive_eviction = evict_binary_resident_prefix_for_decode(
            &mut runtime,
            kv,
            session_id,
            BinaryProactiveEvictionPlan {
                required: true,
                ensure_session_before_eviction: false,
            },
        )?;
        let (predicted, _, output) = run_binary_stage_message(
            &mut runtime,
            session_id,
            &decode_message,
            &[current_token],
            input.as_ref(),
            BinaryStageExecutionOptions::new(
                route == RestorePrefillDecodeRoute::DirectHit,
                stage_output_activation_capacity(
                    config,
                    decode_message.token_count,
                    activation_width,
                )?,
                native_mtp_enabled,
            ),
        )
        .context("execute restore-decode stage message")?;
        (
            predicted,
            output,
            runtime_lock_wait_ms,
            elapsed_ms(lock_hold_started),
            proactive_eviction,
        )
    };
    let compute_ms = elapsed_ms(compute_started);
    emit_binary_proactive_eviction(telemetry, &proactive_eviction);

    if route == RestorePrefillDecodeRoute::ForwardHit {
        let downstream = downstream.expect("forward route requires downstream stage");
        let forwarded =
            forwarded_stage_message_timed(config, &message, &output, wire_dtype, activation_width)
                .context("forward restore-decode activation")?;
        write_stage_message_conditioned(
            &mut *downstream,
            &forwarded.message,
            wire_dtype,
            downstream_wire_condition,
        )
        .context("forward restore-decode downstream")?;
        emit_restore_decode_control(
            config,
            telemetry,
            wire_session_id,
            &message,
            control_started,
            true,
            Some(RestoreDecodeTiming {
                compute_ms,
                runtime_lock_wait_ms,
                runtime_lock_hold_ms,
                proactive_eviction: &proactive_eviction,
                forwarded: Some(&forwarded),
            }),
        );
        return Ok(());
    }

    {
        let mut runtime = runtime.lock().expect("runtime lock poisoned");
        let record = maybe_record_binary_full_prefill(
            config,
            &mut runtime,
            kv,
            telemetry,
            session_id,
            &message,
            message.tokens.as_slice(),
        );
        add_binary_record_stats(&mut control_stats, config, &record);
    }
    emit_restore_decode_control(
        config,
        telemetry,
        wire_session_id,
        &message,
        control_started,
        true,
        Some(RestoreDecodeTiming {
            compute_ms,
            runtime_lock_wait_ms,
            runtime_lock_hold_ms,
            proactive_eviction: &proactive_eviction,
            forwarded: None,
        }),
    );
    send_restore_decode_direct_reply(
        config,
        topology,
        &message,
        wire_dtype,
        downstream_connect_timeout_secs,
        prediction_return_streams,
        StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted: predicted_token,
            predicted_tokens: vec![predicted_token],
            native_mtp_draft: None,
            window: Default::default(),
            stats: control_stats,
        },
    )
    .context("send restore-decode direct predicted reply")
}

struct RestoreDecodeTiming<'a> {
    compute_ms: f64,
    runtime_lock_wait_ms: f64,
    runtime_lock_hold_ms: f64,
    proactive_eviction: &'a BinaryProactiveEviction,
    forwarded: Option<&'a forwarding::ForwardedStageMessage>,
}

#[allow(clippy::too_many_arguments)]
fn emit_restore_decode_control(
    config: &StageConfig,
    telemetry: &Telemetry,
    wire_session_id: u64,
    message: &StageWireMessage,
    control_started: Instant,
    hit: bool,
    timing: Option<RestoreDecodeTiming<'_>>,
) {
    let mut attrs = binary_message_attrs(config, wire_session_id, message);
    attrs.insert("skippy.kv.control_hit".to_string(), json!(hit));
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(elapsed_ms(control_started)),
    );
    if let Some(timing) = timing {
        attrs.insert(
            "llama_stage.compute_ms".to_string(),
            json!(timing.compute_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(timing.runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(timing.runtime_lock_hold_ms),
        );
        timing.proactive_eviction.insert_attrs(&mut attrs);
        if let Some(forwarded) = timing.forwarded {
            attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(forwarded.message.activation.len()),
            );
            attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(forwarded.activation_encode_ms),
            );
        }
    }
    telemetry.emit_debug("stage.binary_prefix_cache_decode_control", attrs);
}

fn send_restore_decode_direct_reply(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    message: &StageWireMessage,
    wire_dtype: WireActivationDType,
    downstream_connect_timeout_secs: u64,
    prediction_return_streams: &mut BTreeMap<(u64, u64), TcpStream>,
    reply: StageReply,
) -> Result<()> {
    if let Some(stream) =
        prediction_return_streams.get_mut(&(message.request_id, message.session_id))
    {
        return direct_return::send_direct_prediction_return(stream, reply);
    }
    send_one_off_direct_return(
        config,
        topology,
        message,
        wire_dtype,
        downstream_connect_timeout_secs,
        reply,
    )
}

fn send_one_off_direct_return(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    message: &StageWireMessage,
    wire_dtype: WireActivationDType,
    downstream_connect_timeout_secs: u64,
    reply: StageReply,
) -> Result<()> {
    let mut stream = direct_return::open_prediction_return_stream(
        config,
        topology,
        message.request_id,
        message.session_id,
        wire_dtype,
        downstream_connect_timeout_secs,
    )?;
    direct_return::send_direct_prediction_return(&mut stream, reply)
}

fn restore_decode_sideband(message: &StageWireMessage) -> Result<(&[i32], i32)> {
    let Some((&current, prefix_tokens)) = message.tokens.split_last() else {
        bail!("restore-decode message requires prefix tokens plus current token");
    };
    if prefix_tokens.is_empty() {
        bail!("restore-decode message requires non-empty prefix tokens");
    }
    Ok((prefix_tokens, current))
}

pub(super) fn restore_prefill_decode_as_decode_message(
    message: &StageWireMessage,
    current_token: i32,
) -> StageWireMessage {
    let mut decode = message.clone();
    decode.kind = WireMessageKind::DecodeEmbd;
    decode.token_count = 1;
    decode.tokens = vec![current_token];
    decode.positions.clear();
    decode.activation.clear();
    decode.raw_bytes.clear();
    decode.state.phase = StageStateHeader::new(
        WireMessageKind::DecodeEmbd,
        message.state.dtype().unwrap_or(WireActivationDType::F32),
    )
    .phase;
    decode.state.current_token = current_token;
    decode
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::StageSamplingConfig;

    #[test]
    fn restore_decode_routes_misses_and_terminal_hits_directly() {
        assert_eq!(
            restore_prefill_decode_route(false, true),
            RestorePrefillDecodeRoute::DirectMiss
        );
        assert_eq!(
            restore_prefill_decode_route(false, false),
            RestorePrefillDecodeRoute::DirectMiss
        );
        assert_eq!(
            restore_prefill_decode_route(true, false),
            RestorePrefillDecodeRoute::DirectHit
        );
    }

    #[test]
    fn restore_decode_forwards_intermediate_hits_without_waiting_for_a_lane_reply() {
        assert_eq!(
            restore_prefill_decode_route(true, true),
            RestorePrefillDecodeRoute::ForwardHit
        );
    }

    #[test]
    fn restore_prefill_decode_as_decode_preserves_chat_metadata() {
        let metadata = r#"{"grammar":"chat"}"#;
        let sampling = StageSamplingConfig {
            flags: 1,
            seed: 42,
            ..StageSamplingConfig::default()
        };
        let mut state = StageStateHeader::new(
            WireMessageKind::TryRestorePrefillDecode,
            WireActivationDType::F16,
        );
        state.prompt_token_count = 4;
        state.decode_step = 0;
        state.current_token = 104;

        let message = StageWireMessage {
            kind: WireMessageKind::TryRestorePrefillDecode,
            pos_start: 3,
            token_count: 1,
            state,
            request_id: 11,
            session_id: 13,
            sampling: Some(sampling.clone()),
            chat_sampling_metadata: Some(metadata.to_string()),
            tokens: vec![101, 102, 103, 104],
            positions: Vec::new(),
            activation: vec![1, 2, 3, 4],
            raw_bytes: Vec::new(),
        };

        let decode = restore_prefill_decode_as_decode_message(&message, 104);

        assert_eq!(decode.kind, WireMessageKind::DecodeEmbd);
        assert_eq!(decode.token_count, 1);
        assert_eq!(decode.tokens, vec![104]);
        assert_eq!(decode.sampling, Some(sampling));
        assert_eq!(decode.chat_sampling_metadata.as_deref(), Some(metadata));
        assert!(decode.activation.is_empty());
        assert!(decode.positions.is_empty());
    }
}
