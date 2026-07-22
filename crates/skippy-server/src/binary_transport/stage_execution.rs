use std::{
    collections::BTreeMap,
    env,
    io::{self, Write},
    net::TcpStream,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use crate::{runtime_state::RuntimeState, telemetry::lifecycle_attrs};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use skippy_metrics::attr;
#[cfg(test)]
use skippy_protocol::{
    LoadMode, PeerConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
    binary::StageStateHeader,
};
use skippy_protocol::{
    MessageBase, SCHEMA_VERSION, StageConfig, StageTopology,
    binary::{
        READY_MAGIC, StageNativeMtpDraft, StageSamplingConfig, StageWireMessage,
        WireActivationDType, WireMessageKind, WireReplyKind,
        activation_frame_flags_from_state_flags, send_ready,
    },
};
use skippy_runtime::{
    ActivationDesc, ActivationFrame, LogitBias, MAX_LOGIT_BIAS, NativeMtpDraft,
    RuntimeActivationDType, RuntimeActivationLayout, SamplingConfig,
};

use super::socket::{connect_downstream_socket, downstream_source_ip, resolve_downstream_endpoint};

const CLIENT_READY_HELLO_ENV: &str = "SKIPPY_STAGE_CLIENT_READY_HELLO";
const CLIENT_READY_HELLO_OPT_IN_PEEK_MS: u64 = 500;

pub(in crate::binary_transport) fn warm_downstream_preconnect_enabled() -> bool {
    warm_downstream_preconnect_enabled_from(
        env::var("SKIPPY_BINARY_WARM_PRECONNECT").ok().as_deref(),
    )
}

fn warm_downstream_preconnect_enabled_from(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(in crate::binary_transport) fn take_warm_or_connect_downstream(
    config: &StageConfig,
    warm_downstream: &Arc<Mutex<Option<TcpStream>>>,
    timeout_secs: u64,
) -> Result<Option<TcpStream>> {
    if config.downstream.is_none() {
        return Ok(None);
    }
    let warm = warm_downstream
        .lock()
        .map_err(|_| anyhow!("warm downstream lock poisoned"))?
        .take();
    match warm {
        Some(stream) if warm_downstream_is_healthy(&stream)? => Ok(Some(stream)),
        Some(_) | None => connect_binary_downstream(config, timeout_secs),
    }
}

pub(in crate::binary_transport) fn warm_downstream_is_healthy(stream: &TcpStream) -> Result<bool> {
    let previous_timeout = stream
        .read_timeout()
        .context("read warm downstream timeout")?;
    stream
        .set_read_timeout(Some(Duration::from_millis(1)))
        .context("set warm downstream health-check timeout")?;
    let mut byte = [0_u8; 1];
    let peek_result = stream.peek(&mut byte);
    stream
        .set_read_timeout(previous_timeout)
        .context("restore warm downstream timeout")?;

    Ok(match peek_result {
        Ok(0) => false,
        Ok(_) => true,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
        {
            true
        }
        Err(_) => false,
    })
}

pub(in crate::binary_transport) fn prepare_binary_stage_connection(
    stream: &TcpStream,
) -> Result<()> {
    stream
        .set_nonblocking(false)
        .context("set binary stage connection blocking")?;
    stream.set_nodelay(true).ok();
    Ok(())
}

pub(crate) fn send_client_ready_hello_if_enabled(stream: &mut TcpStream) -> Result<()> {
    if !client_ready_hello_enabled() {
        return Ok(());
    }
    send_ready(&mut *stream).context("send client ready hello")?;
    stream.flush().ok();
    Ok(())
}

pub(in crate::binary_transport) fn consume_optional_client_ready_hello(
    stream: &mut TcpStream,
) -> Result<()> {
    if !client_ready_hello_enabled() {
        return Ok(());
    }
    let previous_timeout = stream
        .read_timeout()
        .context("read stage connection timeout")?;
    stream
        .set_read_timeout(Some(Duration::from_millis(
            CLIENT_READY_HELLO_OPT_IN_PEEK_MS,
        )))
        .context("set client ready hello peek timeout")?;
    let mut bytes = [0_u8; 4];
    let peek_result = stream.peek(&mut bytes);
    stream
        .set_read_timeout(previous_timeout)
        .context("restore stage connection timeout")?;

    match peek_result {
        Ok(4) if i32::from_le_bytes(bytes) == READY_MAGIC => {
            skippy_protocol::binary::recv_ready(&mut *stream)
                .context("consume client ready hello")?;
            eprintln!("binary consumed client ready hello");
        }
        Ok(_) => {}
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) => {}
        Err(error) => return Err(error).context("peek optional client ready hello"),
    }
    Ok(())
}

fn client_ready_hello_enabled() -> bool {
    env::var(CLIENT_READY_HELLO_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}
pub(in crate::binary_transport) fn insert_optional_unix_nanos(
    attrs: &mut BTreeMap<String, Value>,
    key: &str,
    value: Option<u64>,
) {
    if let Some(value) = value {
        attrs.insert(key.to_string(), json!(value));
    }
}

fn native_mtp_prediction_tokens(predicted: i32, draft: Option<NativeMtpDraft>) -> Vec<i32> {
    let Some(draft) = draft else {
        return vec![predicted];
    };
    let token_count = i32::try_from(draft.token_ids.len()).unwrap_or(i32::MAX);
    let mut tokens = Vec::with_capacity(draft.token_ids.len() + 3);
    tokens.push(predicted);
    tokens.push(token_count);
    tokens.extend(draft.token_ids);
    tokens.push(draft.proposal_compute_us.clamp(0, i64::from(i32::MAX)) as i32);
    tokens
}

/// Converts the temporary llama-stage sideband into the typed stage reply field.
///
/// The C ABI still returns the proposal as a trailer. The network boundary is
/// authoritative: consumers receive only target predictions plus a separate
/// native-MTP draft. A malformed trailer is rejected instead of being exposed
/// as target-model output.
pub(in crate::binary_transport) fn split_native_mtp_reply(
    message: &StageWireMessage,
    prediction_tokens: &mut Vec<i32>,
) -> Result<Option<StageNativeMtpDraft>> {
    let sideband_offset = match message.kind {
        WireMessageKind::DecodeEmbd
        | WireMessageKind::DecodeReadout
        | WireMessageKind::DecodeLightCtx
        | WireMessageKind::DecodeReplayEmbd
        | WireMessageKind::DecodeReplayFinalEmbd => 1,
        WireMessageKind::VerifyWindow => message.tokens.len(),
        _ => return Ok(None),
    };
    if prediction_tokens.len() <= sideband_offset {
        return Ok(None);
    }
    let draft_token_count = usize::try_from(prediction_tokens[sideband_offset])
        .map_err(|_| anyhow!("negative native MTP draft token count"))?;
    let draft_start = sideband_offset + 1;
    let draft_end = draft_start
        .checked_add(draft_token_count)
        .ok_or_else(|| anyhow!("native MTP draft token count overflow"))?;
    let compute_index = draft_end;
    if prediction_tokens.len() != compute_index + 1 {
        bail!(
            "malformed native MTP sideband: expected {} values, got {}",
            compute_index + 1,
            prediction_tokens.len()
        );
    }
    let proposal_compute_us = i64::from(prediction_tokens[compute_index].max(0));
    let token_ids = prediction_tokens[draft_start..draft_end].to_vec();
    prediction_tokens.truncate(sideband_offset);
    Ok((!token_ids.is_empty()).then_some(StageNativeMtpDraft {
        token_ids,
        proposal_compute_us,
    }))
}

pub(crate) fn stage_output_activation_capacity(
    config: &StageConfig,
    token_count: i32,
    activation_width: i32,
) -> Result<usize> {
    if config.downstream.is_none() || token_count <= 0 {
        return Ok(0);
    }
    skippy_protocol::binary::activation_wire_bytes(
        WireActivationDType::F32,
        token_count,
        activation_width,
    )
    .context("estimate output activation capacity")
}
pub(in crate::binary_transport) fn estimated_reply_wire_bytes(
    reply_kind: WireReplyKind,
    predicted_token_count: usize,
) -> usize {
    const REPLY_HEADER_BYTES: usize = 3 * std::mem::size_of::<i32>();
    const REPLY_STATS_BYTES: usize = 23 * std::mem::size_of::<i64>();
    let token_count = match reply_kind {
        WireReplyKind::Ack => 0,
        WireReplyKind::PredictedToken => 1,
        WireReplyKind::PredictedTokens => predicted_token_count,
    };
    REPLY_HEADER_BYTES + token_count * std::mem::size_of::<i32>() + REPLY_STATS_BYTES
}
pub(in crate::binary_transport) fn binary_message_attrs(
    config: &StageConfig,
    session_id: u64,
    message: &StageWireMessage,
) -> std::collections::BTreeMap<String, serde_json::Value> {
    let mut attrs = lifecycle_attrs(config);
    let epoch = message.request_epoch();
    attrs.insert(attr::SESSION_ID.to_string(), json!(session_id.to_string()));
    attrs.insert(
        attr::REQUEST_ID.to_string(),
        json!(binary_message_request_id(message)),
    );
    attrs.insert(attr::PROMPT_INDEX.to_string(), json!(message.state.seq_id));
    attrs.insert(
        attr::MESSAGE_KIND.to_string(),
        json!(format!("{:?}", message.kind)),
    );
    attrs.insert(attr::TOKEN_COUNT.to_string(), json!(message.token_count));
    attrs.insert(
        attr::CHECKPOINT_GENERATION.to_string(),
        json!(epoch.checkpoint_generation),
    );
    attrs.insert(
        attr::PROMPT_TOKEN_COUNT.to_string(),
        json!(epoch.prompt_token_count),
    );
    attrs.insert(attr::DECODE_STEP.to_string(), json!(epoch.decode_step));
    let layer_count = i64::from(config.layer_end.saturating_sub(config.layer_start));
    let kv_tokens_after = estimated_kv_tokens_after(message);
    attrs.insert("skippy.kv_tokens_after".to_string(), json!(kv_tokens_after));
    attrs.insert("skippy.kv_layer_count".to_string(), json!(layer_count));
    attrs.insert(
        "skippy.kv_token_layer_cells".to_string(),
        json!(kv_tokens_after.saturating_mul(layer_count)),
    );
    attrs
}

pub(in crate::binary_transport) fn estimated_kv_tokens_after(message: &StageWireMessage) -> i64 {
    if message.kind == WireMessageKind::Stop {
        return 0;
    }
    let pos_start = i64::from(message.pos_start.max(0));
    let token_count = i64::from(message.token_count.max(0));
    pos_start.saturating_add(token_count)
}
pub(in crate::binary_transport) fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

pub(in crate::binary_transport) fn nanos_delta_ms(
    start_unix_nanos: u64,
    end_unix_nanos: u64,
) -> f64 {
    end_unix_nanos.saturating_sub(start_unix_nanos) as f64 / 1_000_000.0
}

pub(in crate::binary_transport) fn ms_to_us(ms: f64) -> i64 {
    if !ms.is_finite() || ms <= 0.0 {
        0
    } else {
        (ms * 1000.0).round().min(i64::MAX as f64) as i64
    }
}

pub(in crate::binary_transport) fn stage_mask(stage_index: u32) -> i64 {
    if stage_index < 63 {
        1_i64 << stage_index
    } else {
        0
    }
}

pub(in crate::binary_transport) fn binary_message_base(
    config: &StageConfig,
    session_id: &str,
    message: &StageWireMessage,
) -> MessageBase {
    MessageBase {
        schema_version: SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        request_id: binary_message_request_id(message),
        session_id: session_id.to_string(),
        stage_id: "binary-driver".to_string(),
        stage_index: 0,
        topology_id: config.topology_id.clone(),
        model_id: Some(config.model_id.clone()),
        tokenizer_id: None,
        chat_template_id: None,
        seq: Some(message.state.seq_id.max(0) as u64),
    }
}

pub(in crate::binary_transport) fn binary_message_session_id(
    fallback: u64,
    message: &StageWireMessage,
) -> u64 {
    if message.session_id == 0 {
        fallback
    } else {
        message.session_id
    }
}

pub(in crate::binary_transport) fn binary_message_request_id(message: &StageWireMessage) -> String {
    if message.request_id == 0 {
        format!("prompt-{}", message.state.seq_id)
    } else {
        message.request_id.to_string()
    }
}
pub(crate) fn connect_binary_downstream(
    config: &StageConfig,
    timeout_secs: u64,
) -> Result<Option<TcpStream>> {
    let Some(peer) = config.downstream.as_ref() else {
        return Ok(None);
    };
    let endpoint = peer
        .endpoint
        .strip_prefix("tcp://")
        .unwrap_or(&peer.endpoint);
    let downstream_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match connect_downstream_socket(downstream_addr, source_ip, Duration::from_secs(2)) {
            Ok(stream) => {
                stream.set_nodelay(true).ok();
                return Ok(Some(stream));
            }
            Err(error) => {
                last_error = Some(anyhow!(error));
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out"))
        .context(format!(
            "connect downstream binary stage at {endpoint} ({downstream_addr})"
        )))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BinaryStageExecutionOptions {
    pub(crate) sample_final_prefill: bool,
    pub(crate) output_capacity: usize,
    pub(crate) native_mtp_enabled: bool,
    pub(crate) native_mtp_max_tokens: usize,
}

impl BinaryStageExecutionOptions {
    pub(crate) fn new(
        sample_final_prefill: bool,
        output_capacity: usize,
        native_mtp_enabled: bool,
    ) -> Self {
        Self {
            sample_final_prefill,
            output_capacity,
            native_mtp_enabled,
            native_mtp_max_tokens: 1,
        }
    }

    pub(crate) fn with_native_mtp_max_tokens(mut self, value: usize) -> Self {
        self.native_mtp_max_tokens = value.max(1);
        self
    }
}

pub(crate) fn run_binary_stage_message(
    runtime: &mut RuntimeState,
    session_id: &str,
    message: &StageWireMessage,
    token_ids: &[i32],
    input: Option<&ActivationFrame>,
    options: BinaryStageExecutionOptions,
) -> Result<(i32, Vec<i32>, ActivationFrame)> {
    match message.kind {
        WireMessageKind::PrefillEmbd => {
            let output = runtime.prefill_frame_with_positions(
                session_id,
                token_ids,
                &message.positions,
                input,
            )?;
            Ok((message.state.current_token, Vec::new(), output))
        }
        WireMessageKind::PrefillFinalEmbd if options.sample_final_prefill => {
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            let (predicted, output) = runtime.prefill_final_frame_sampled(
                session_id,
                token_ids,
                &message.positions,
                sampling.as_ref(),
                input,
            )?;
            Ok((predicted, Vec::new(), output))
        }
        WireMessageKind::PrefillFinalEmbd => {
            let output = runtime.prefill_frame_with_positions(
                session_id,
                token_ids,
                &message.positions,
                input,
            )?;
            Ok((message.state.current_token, Vec::new(), output))
        }
        WireMessageKind::DecodeEmbd
        | WireMessageKind::DecodeReadout
        | WireMessageKind::DecodeLightCtx
        | WireMessageKind::DecodeReplayEmbd
        | WireMessageKind::DecodeReplayFinalEmbd => {
            let token_id = token_ids
                .first()
                .copied()
                .unwrap_or(message.state.current_token);
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            if !options.native_mtp_enabled {
                let (predicted, output) = runtime.decode_frame_sampled(
                    session_id,
                    token_id,
                    sampling.as_ref(),
                    input,
                    options.output_capacity,
                )?;
                return Ok((predicted, vec![predicted], output));
            }
            let (predicted, native_mtp, output) = runtime.decode_frame_sampled_mtp(
                session_id,
                token_id,
                sampling.as_ref(),
                input,
                options.output_capacity,
                options.native_mtp_max_tokens,
            )?;
            Ok((
                predicted,
                native_mtp_prediction_tokens(predicted, native_mtp),
                output,
            ))
        }
        WireMessageKind::VerifyWindow => {
            let sampling = runtime_sampling_config(message.sampling.as_ref());
            let (predicted_tokens, output) = runtime.verify_frame_sampled(
                session_id,
                token_ids,
                sampling.as_ref(),
                input,
                options.output_capacity,
            )?;
            let predicted = predicted_tokens.first().copied().unwrap_or(0);
            Ok((predicted, predicted_tokens, output))
        }
        WireMessageKind::Stop
        | WireMessageKind::StateImport
        | WireMessageKind::StateExport
        | WireMessageKind::ConfigureGeneration
        | WireMessageKind::TrimSession
        | WireMessageKind::ProbePrefill
        | WireMessageKind::RestorePrefill
        | WireMessageKind::TryRestorePrefill
        | WireMessageKind::TryRestorePrefillDecode
        | WireMessageKind::PredictionReturnOpen => {
            bail!("message kind is not executable")
        }
    }
}

pub(in crate::binary_transport) fn is_decode_frame_batch_candidate(
    config: &StageConfig,
    message: &StageWireMessage,
    token_ids: &[i32],
) -> bool {
    if config.downstream.is_none() {
        return false;
    }

    matches!(
        message.kind,
        WireMessageKind::DecodeEmbd
            | WireMessageKind::DecodeReadout
            | WireMessageKind::DecodeLightCtx
            | WireMessageKind::DecodeReplayEmbd
            | WireMessageKind::DecodeReplayFinalEmbd
    ) && message.token_count == 1
        && token_ids.len() == 1
}

pub(in crate::binary_transport) fn runtime_sampling_config(
    sampling: Option<&StageSamplingConfig>,
) -> Option<SamplingConfig> {
    let sampling = sampling?;
    let mut config = SamplingConfig {
        enabled: true,
        seed: sampling.seed,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        repeat_penalty: sampling.repeat_penalty,
        penalty_last_n: sampling.penalty_last_n,
        ..SamplingConfig::default()
    };
    config.logit_bias = sampling
        .logit_bias
        .iter()
        .take(MAX_LOGIT_BIAS)
        .map(|source| LogitBias {
            token_id: source.token_id,
            bias: source.bias,
        })
        .collect();
    sampling.enabled().then_some(config)
}

pub(in crate::binary_transport) fn input_activation_frame(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    message: &mut StageWireMessage,
    activation_width: i32,
) -> Result<Option<ActivationFrame>> {
    if message.activation.is_empty() {
        return Ok(None);
    }
    let payload = message
        .take_activation_f32_payload(activation_width)
        .context("decode wire activation payload")?;
    let (layer_start, layer_end) = upstream_layer_range(config, topology, message);
    Ok(Some(ActivationFrame {
        desc: ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: message.state.source_stage_index,
            layer_start,
            layer_end,
            token_count: message.token_count.try_into().unwrap_or(0),
            sequence_count: if message.token_count > 0 { 1 } else { 0 },
            payload_bytes: payload.len() as u64,
            flags: activation_frame_flags_from_state_flags(message.state.flags),
        },
        payload,
    }))
}

pub(in crate::binary_transport) fn empty_activation_frame(
    config: &StageConfig,
    message: &StageWireMessage,
) -> ActivationFrame {
    ActivationFrame {
        desc: ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: config.stage_index as i32,
            layer_start: config.layer_start as i32,
            layer_end: config.layer_end as i32,
            token_count: message.token_count.try_into().unwrap_or(0),
            sequence_count: if message.token_count > 0 { 1 } else { 0 },
            payload_bytes: 0,
            flags: 0,
        },
        payload: Vec::new(),
    }
}

fn upstream_layer_range(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    message: &StageWireMessage,
) -> (i32, i32) {
    if let Some(topology) = topology
        && let Some(stage) = topology
            .stages
            .iter()
            .find(|stage| stage.stage_index as i32 == message.state.source_stage_index)
    {
        return (stage.layer_start as i32, stage.layer_end as i32);
    }
    (0, config.layer_start as i32)
}

pub(in crate::binary_transport) fn token_sideband_or_fill(
    message: &StageWireMessage,
) -> Result<Vec<i32>> {
    let token_count: usize = message
        .token_count
        .try_into()
        .context("negative token_count")?;
    if let Some(token) = decode_execution_token(message, token_count) {
        return Ok(vec![token]);
    }
    if message.tokens.len() == token_count {
        return Ok(message.tokens.clone());
    }
    if !message.tokens.is_empty() && token_count == 1 {
        return Ok(vec![message.tokens[0]]);
    }
    let fill = if message.state.current_token != skippy_protocol::binary::LLAMA_TOKEN_NULL {
        message.state.current_token
    } else {
        0
    };
    Ok(vec![fill; token_count])
}

fn decode_execution_token(message: &StageWireMessage, token_count: usize) -> Option<i32> {
    if !matches!(
        message.kind,
        WireMessageKind::DecodeEmbd
            | WireMessageKind::DecodeReadout
            | WireMessageKind::DecodeLightCtx
            | WireMessageKind::DecodeReplayEmbd
            | WireMessageKind::DecodeReplayFinalEmbd
    ) || token_count != 1
        || message.state.current_token == skippy_protocol::binary::LLAMA_TOKEN_NULL
    {
        return None;
    }
    Some(message.state.current_token)
}

pub(in crate::binary_transport) fn decode_record_tokens_sideband(
    message: &StageWireMessage,
) -> Option<&[i32]> {
    if message.kind != WireMessageKind::DecodeEmbd
        || message.token_count != 1
        || message.state.prompt_token_count <= 0
    {
        return None;
    }
    let prompt_token_count = usize::try_from(message.state.prompt_token_count).ok()?;
    let decode_step = usize::try_from(message.state.decode_step).ok()?;
    let expected_token_count = prompt_token_count.checked_add(decode_step)?;
    if message.tokens.len() != expected_token_count
        || message.tokens.last().copied() != Some(message.state.current_token)
    {
        return None;
    }
    Some(message.tokens.as_slice())
}

#[cfg(test)]
pub(in crate::binary_transport) fn prefix_cache_test_config() -> StageConfig {
    StageConfig {
        run_id: "run".to_string(),
        topology_id: "topology".to_string(),
        model_id: "org/model:Q4_K_M".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: None,
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 4,
        ctx_size: 8192,
        lane_count: 2,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: 0,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: Default::default(),
        filter_tensors_on_load: false,
        selected_device: None,
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            payload: StageKvCachePayload::ResidentKv,
            max_entries: 8,
            max_bytes: 0,
            min_tokens: 256,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: 2,
        }),
        native_mtp_enabled: true,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: Some(PeerConfig {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            endpoint: "127.0.0.1:0".to_string(),
        }),
    }
}

#[cfg(test)]
pub(in crate::binary_transport) fn first_decode_message_with_full_prompt_sideband()
-> StageWireMessage {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F16);
    state.prompt_token_count = 4;
    state.decode_step = 0;
    state.current_token = 104;
    StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 3,
        token_count: 1,
        state,
        request_id: 11,
        session_id: 13,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![101, 102, 103, 104],
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_record_tokens_sideband, first_decode_message_with_full_prompt_sideband,
        is_decode_frame_batch_candidate, prefix_cache_test_config, prepare_binary_stage_connection,
        split_native_mtp_reply, take_warm_or_connect_downstream, token_sideband_or_fill,
        warm_downstream_is_healthy, warm_downstream_preconnect_enabled_from,
    };
    use skippy_protocol::binary::{
        StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
    };
    use std::{
        io,
        net::{Shutdown, TcpListener, TcpStream},
        os::fd::AsRawFd,
        thread,
        time::Duration,
    };

    #[test]
    fn accepted_binary_stage_connection_is_blocking() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let client = thread::spawn(move || TcpStream::connect(addr).unwrap());

        let (stream, _) = loop {
            match listener.accept() {
                Ok(conn) => break conn,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("accept failed: {error}"),
            }
        };
        stream.set_nonblocking(true).unwrap();
        prepare_binary_stage_connection(&stream).unwrap();

        let flags = unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_GETFL) };
        assert_ne!(flags, -1);
        assert_eq!(flags & libc::O_NONBLOCK, 0);
        drop(client.join().unwrap());
    }

    #[test]
    fn warm_preconnect_is_opt_in() {
        assert!(!warm_downstream_preconnect_enabled_from(None));
        assert!(!warm_downstream_preconnect_enabled_from(Some("0")));
        assert!(warm_downstream_preconnect_enabled_from(Some("true")));
        assert!(warm_downstream_preconnect_enabled_from(Some(" ON ")));
    }

    #[test]
    fn warm_downstream_connection_is_consumed_before_connecting() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let warm = std::sync::Arc::new(std::sync::Mutex::new(Some(server)));

        let result = take_warm_or_connect_downstream(&prefix_cache_test_config(), &warm, 1)
            .unwrap()
            .unwrap();

        assert_eq!(result.peer_addr().unwrap(), client.local_addr().unwrap());
        assert!(warm.lock().unwrap().is_none());
    }

    #[test]
    fn stale_warm_downstream_connection_is_replaced() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let client = TcpStream::connect(&endpoint).unwrap();
        let (stale_server, _) = listener.accept().unwrap();
        client.shutdown(Shutdown::Both).unwrap();

        for _ in 0..20 {
            if !warm_downstream_is_healthy(&stale_server).unwrap() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(!warm_downstream_is_healthy(&stale_server).unwrap());

        let mut config = prefix_cache_test_config();
        config.downstream.as_mut().unwrap().endpoint = endpoint;
        let warm = std::sync::Arc::new(std::sync::Mutex::new(Some(stale_server)));
        let replacement = take_warm_or_connect_downstream(&config, &warm, 1)
            .unwrap()
            .unwrap();
        let (accepted, _) = listener.accept().unwrap();

        assert_eq!(
            accepted.peer_addr().unwrap(),
            replacement.local_addr().unwrap()
        );
        assert!(warm.lock().unwrap().is_none());
    }
    #[test]
    fn decode_record_tokens_sideband_records_metadata_without_changing_exec_token() {
        let message = first_decode_message_with_full_prompt_sideband();

        let exec_tokens = token_sideband_or_fill(&message).unwrap();
        let prompt_tokens = decode_record_tokens_sideband(&message).unwrap();

        assert_eq!(exec_tokens, vec![104]);
        assert_eq!(prompt_tokens, &[101, 102, 103, 104]);
    }

    #[test]
    fn decode_record_tokens_sideband_accepts_decode_checkpoint() {
        let mut message = first_decode_message_with_full_prompt_sideband();
        message.state.decode_step = 1;
        message.state.current_token = 201;
        message.tokens.push(201);

        assert_eq!(
            decode_record_tokens_sideband(&message).unwrap(),
            &[101, 102, 103, 104, 201]
        );
        assert_eq!(token_sideband_or_fill(&message).unwrap(), vec![201]);
    }

    #[test]
    fn native_mtp_sideband_is_removed_from_verify_predictions() {
        let mut message = test_message(WireMessageKind::VerifyWindow, 3);
        message.tokens = vec![10, 11, 12];
        let mut predictions = vec![11, 12, 13, 2, 14, 15, 123];

        let draft = split_native_mtp_reply(&message, &mut predictions).unwrap();

        assert_eq!(predictions, vec![11, 12, 13]);
        assert_eq!(
            draft,
            Some(skippy_protocol::binary::StageNativeMtpDraft {
                token_ids: vec![14, 15],
                proposal_compute_us: 123,
            })
        );
    }

    #[test]
    fn malformed_native_mtp_sideband_is_rejected() {
        let message = test_message(WireMessageKind::DecodeEmbd, 1);
        let mut predictions = vec![11, 2, 12];

        let error = split_native_mtp_reply(&message, &mut predictions).unwrap_err();

        assert!(error.to_string().contains("malformed native MTP sideband"));
    }

    fn test_message(kind: WireMessageKind, token_count: i32) -> StageWireMessage {
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count,
            state: StageStateHeader::new(kind, WireActivationDType::F16),
            request_id: 11,
            session_id: 13,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn decode_record_tokens_sideband_rejects_wrong_checkpoint_len() {
        let mut message = first_decode_message_with_full_prompt_sideband();
        message.state.decode_step = 1;

        assert!(decode_record_tokens_sideband(&message).is_none());
        assert_eq!(token_sideband_or_fill(&message).unwrap(), vec![104]);
    }

    #[test]
    fn decode_frame_batch_candidate_keeps_intermediate_decode_batching() {
        let config = prefix_cache_test_config();
        let message = first_decode_message_with_full_prompt_sideband();

        assert!(is_decode_frame_batch_candidate(&config, &message, &[104]));
    }

    #[test]
    fn decode_frame_batch_candidate_skips_final_output_stage() {
        let mut config = prefix_cache_test_config();
        config.downstream = None;
        let message = first_decode_message_with_full_prompt_sideband();

        assert!(!is_decode_frame_batch_candidate(&config, &message, &[104]));
    }
}
