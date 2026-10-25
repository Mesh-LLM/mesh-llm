pub(crate) use super::stage_connections::{
    connect_binary_downstream, connect_binary_downstream_cancellable,
    send_client_ready_hello_if_enabled,
};
pub(in crate::binary_transport) use super::stage_connections::{
    consume_optional_client_ready_hello, prepare_binary_stage_connection, take_ready_downstream,
    warm_downstream_preconnect_enabled,
};
#[cfg(test)]
use super::stage_connections::{
    take_warm_or_connect_downstream, warm_downstream_is_healthy,
    warm_downstream_preconnect_enabled_from,
};

use std::{collections::BTreeMap, time::Instant};

use crate::{runtime_state::RuntimeState, telemetry::lifecycle_attrs};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use skippy_metrics::attr;
#[cfg(test)]
use skippy_protocol::{
    LoadMode, PeerConfig, StageKvCacheConfig, StageKvCacheMode, binary::StageStateHeader,
};
use skippy_protocol::{
    MessageBase, SCHEMA_VERSION, StageConfig, StageTopology,
    binary::{StageNativeMtpDraft, StageWireMessage, WireMessageKind, WireReplyKind},
};
use skippy_runtime::{
    ACTIVATION_FRAME_VERSION, ACTIVATION_MAX_PARTS, ActivationDesc, ActivationFrame,
};

use super::native_mtp_reply::{
    native_mtp_draft_tokens, native_mtp_prediction_tokens, stage_native_mtp_draft,
    verify_reply_native_mtp_draft,
};
use super::peer_sampling::runtime_sampling_config;
use super::prefill_execution::runtime_activation_frame;

pub(in crate::binary_transport) fn insert_optional_unix_nanos(
    attrs: &mut BTreeMap<String, Value>,
    key: &str,
    value: Option<u64>,
) {
    if let Some(value) = value {
        attrs.insert(key.to_string(), json!(value));
    }
}

pub(crate) fn stage_output_activation_capacity(
    has_downstream: bool,
    token_count: i32,
    boundary: Option<skippy_runtime::ActivationBoundaryDesc>,
) -> Result<usize> {
    if !has_downstream || token_count <= 0 {
        return Ok(0);
    }
    let token_count = u32::try_from(token_count).context("output activation token count")?;
    let bytes = boundary
        .context("loaded stage does not expose an output activation boundary")?
        .payload_bytes_hint("output", token_count)?
        .unwrap_or(0);
    usize::try_from(bytes).context("output activation capacity exceeds usize")
}
pub(in crate::binary_transport) fn estimated_reply_wire_bytes(
    reply_kind: WireReplyKind,
    predicted_token_count: usize,
) -> usize {
    const REPLY_HEADER_BYTES: usize = 3 * std::mem::size_of::<i32>();
    const REPLY_STATS_BYTES: usize = 27 * std::mem::size_of::<i64>();
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
) -> Result<(i32, Vec<i32>, ActivationFrame, Option<StageNativeMtpDraft>)> {
    match message.kind {
        WireMessageKind::PrefillEmbd => {
            let output = runtime.prefill_frame_with_positions(
                session_id,
                token_ids,
                &message.positions,
                input,
            )?;
            Ok((message.state.current_token, Vec::new(), output, None))
        }
        WireMessageKind::PrefillFinalEmbd if options.sample_final_prefill => {
            let sampling =
                runtime_sampling_config(message.sampling.as_ref(), runtime.kv_pool_tokens());
            let (predicted, output) = runtime.prefill_final_frame_sampled(
                session_id,
                token_ids,
                &message.positions,
                sampling.as_ref(),
                input,
            )?;
            Ok((predicted, Vec::new(), output, None))
        }
        WireMessageKind::PrefillFinalEmbd => {
            let output = runtime.prefill_frame_with_positions(
                session_id,
                token_ids,
                &message.positions,
                input,
            )?;
            Ok((message.state.current_token, Vec::new(), output, None))
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
            let sampling =
                runtime_sampling_config(message.sampling.as_ref(), runtime.kv_pool_tokens());
            let draft_tokens = native_mtp_draft_tokens(message, options);
            if !options.native_mtp_enabled || draft_tokens == 0 {
                let (predicted, output) = runtime.decode_frame_sampled(
                    session_id,
                    token_id,
                    sampling.as_ref(),
                    input,
                    options.output_capacity,
                )?;
                return Ok((predicted, vec![predicted], output, None));
            }
            let (predicted, native_mtp, output) = runtime.decode_frame_sampled_mtp(
                session_id,
                token_id,
                sampling.as_ref(),
                input,
                options.output_capacity,
                draft_tokens,
            )?;
            Ok((
                predicted,
                native_mtp_prediction_tokens(predicted, native_mtp),
                output,
                None,
            ))
        }
        WireMessageKind::VerifyWindow => {
            let sampling =
                runtime_sampling_config(message.sampling.as_ref(), runtime.kv_pool_tokens());
            let (predicted_tokens, native_mtp, output) = runtime.verify_frame_sampled(
                session_id,
                token_ids,
                sampling.as_ref(),
                input,
                options.output_capacity,
                native_mtp_draft_tokens(message, options),
            )?;
            let predicted = predicted_tokens.first().copied().unwrap_or(0);
            let draft = verify_reply_native_mtp_draft(
                native_mtp.map(stage_native_mtp_draft),
                token_ids,
                &predicted_tokens,
                message.state.accepts_native_mtp_draft_after_correction(),
            );
            Ok((predicted, predicted_tokens, output, draft))
        }
        WireMessageKind::Stop
        | WireMessageKind::StateImport
        | WireMessageKind::StateExport
        | WireMessageKind::ConfigureGeneration
        | WireMessageKind::TrimSession
        | WireMessageKind::RetireVerifyWindow
        | WireMessageKind::DiscardStaleWindows
        | WireMessageKind::ProbePrefill
        | WireMessageKind::RestorePrefill
        | WireMessageKind::TryRestorePrefill
        | WireMessageKind::TryRestorePrefillDecode
        | WireMessageKind::PredictionReturnOpen => {
            bail!("message kind is not executable")
        }
    }
}

/// Environment switch: let the final stage batch single-token decode across
/// lanes. The batched path does not produce native MTP drafts, so the final
/// stage stays unbatched whenever native MTP is enabled.
pub(in crate::binary_transport) const LAST_STAGE_DECODE_BATCH_ENV: &str =
    "SKIPPY_LAST_STAGE_DECODE_BATCH";

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Whether the final stage batches decode across lanes.
///
/// `planned` is `throughput.last_stage_decode_batch`, delivered on the stage
/// load because the stage that owns the output layers is a different process -
/// on a mesh split a different machine - from the one the operator configured,
/// and `--auto-balance` can move which stage that is.
///
/// The environment variable stays a bench and incident-response override that
/// needs no replan, the same shape as `SKIPPY_KV_CACHE`. It is read first and
/// decides on its own: a set-but-unparseable value means unbatched rather than
/// deferring to the plan, so an operator's typo cannot look like a policy.
pub(in crate::binary_transport) fn resolve_last_stage_decode_batch(
    env_value: Option<&str>,
    planned: Option<bool>,
) -> bool {
    match env_value {
        Some(value) => truthy(value),
        None => planned.unwrap_or(false),
    }
}

pub(in crate::binary_transport) fn last_stage_decode_batch_enabled(planned: Option<bool>) -> bool {
    resolve_last_stage_decode_batch(
        std::env::var(LAST_STAGE_DECODE_BATCH_ENV).ok().as_deref(),
        planned,
    )
}

pub(in crate::binary_transport) fn is_decode_frame_batch_candidate(
    config: &StageConfig,
    message: &StageWireMessage,
    token_ids: &[i32],
    native_mtp_enabled: bool,
    last_stage_batching: bool,
) -> bool {
    if config.downstream.is_none() && (native_mtp_enabled || !last_stage_batching) {
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

pub(in crate::binary_transport) fn input_activation_frame(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    message: &mut StageWireMessage,
) -> Result<Option<ActivationFrame>> {
    if message.activation.is_empty() {
        return Ok(None);
    }
    let frame = message
        .take_activation_frame()
        .context("decode wire activation frame")?
        .context("activation payload did not contain a frame")?;
    let expected_range = upstream_layer_range(config, topology, message);
    if (frame.desc.layer_start, frame.desc.layer_end) != expected_range {
        bail!(
            "activation frame layer range {}..{} does not match upstream stage {}..{}",
            frame.desc.layer_start,
            frame.desc.layer_end,
            expected_range.0,
            expected_range.1,
        );
    }
    runtime_activation_frame(frame).map(Some)
}

pub(in crate::binary_transport) fn empty_activation_frame(
    config: &StageConfig,
    message: &StageWireMessage,
) -> ActivationFrame {
    ActivationFrame {
        desc: ActivationDesc {
            version: ACTIVATION_FRAME_VERSION,
            producer_stage_index: config.stage_index as i32,
            layer_start: config.layer_start as i32,
            layer_end: config.layer_end as i32,
            token_count: message.token_count.try_into().unwrap_or(0),
            sequence_count: if message.token_count > 0 { 1 } else { 0 },
            part_count: 0,
            payload_bytes: 0,
            frontier_identity: [0; skippy_runtime::ACTIVATION_IDENTITY_BYTES],
            parts: [skippy_runtime::ActivationPartDesc::default(); ACTIVATION_MAX_PARTS],
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
    // The upstream peer chooses token_count, and a missing sideband is filled
    // in below, so bound it by the sideband limit before allocating.
    if token_count > skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES {
        bail!("token_count {token_count} exceeds the stage message token limit");
    }
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
        model_id: "hugging-quants/Llama-3.2-1B-Instruct-GGUF:Q4_K_M".to_string(),
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
        repack: false,
        op_offload: None,
        no_host_buffer: false,
        check_tensors: false,
        direct_io: false,
        main_gpu: None,
        split_mode: skippy_protocol::SplitMode::Auto,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: Default::default(),
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
        cache_idle_slots: None,
        resident_tensor_names: Vec::new(),
        selected_device: None,
        kv_cache: Some(StageKvCacheConfig {
            exact_max_bytes: None,
            mode: StageKvCacheMode::LookupRecord,
            max_entries: 8,
            max_bytes: 0,
            l2_max_bytes: 0,
            codec: skippy_protocol::StageKvCacheCodec::Native,
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
        ..StageConfig::default()
    }
}

#[cfg(test)]
pub(in crate::binary_transport) fn first_decode_message_with_full_prompt_sideband()
-> StageWireMessage {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
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
        is_decode_frame_batch_candidate, prefix_cache_test_config,
        stage_output_activation_capacity, take_ready_downstream, take_warm_or_connect_downstream,
        token_sideband_or_fill, warm_downstream_is_healthy,
        warm_downstream_preconnect_enabled_from,
    };
    use crate::stage_transport::{StageConnection, StageListener};
    use skippy_protocol::binary::{StageStateHeader, StageWireMessage, WireMessageKind};
    use std::{
        net::Shutdown,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    #[cfg(unix)]
    #[test]
    fn accepted_binary_stage_connection_is_blocking() {
        use super::prepare_binary_stage_connection;
        use std::{io, os::fd::AsRawFd};

        let listener = StageListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let client = thread::spawn(move || StageConnection::connect(addr).unwrap());

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

        let flags = unsafe { libc::fcntl(stream.tcp().unwrap().as_raw_fd(), libc::F_GETFL) };
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
    fn dynamic_output_dimensions_defer_capacity_to_native_preparation() {
        let mut boundary = crate::test_activation::boundary_f32(4);
        boundary.parts[0].rank = 3;
        boundary.parts[0].token_axis = 2;
        boundary.parts[0].dimensions = [4, -1, -1, 0];

        assert_eq!(
            stage_output_activation_capacity(true, 2, Some(boundary)).unwrap(),
            0
        );
    }

    #[test]
    fn warm_downstream_connection_is_consumed_before_connecting() {
        let listener = StageListener::bind("127.0.0.1:0").unwrap();
        let client = StageConnection::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let warm = std::sync::Arc::new(std::sync::Mutex::new(Some(server)));

        let result = take_warm_or_connect_downstream(
            &prefix_cache_test_config(),
            None,
            &warm,
            Duration::from_secs(1),
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.peer_addr().unwrap(), client.local_addr().unwrap());
        assert!(warm.lock().unwrap().is_none());
    }

    #[test]
    fn stale_warm_downstream_connection_is_replaced() {
        let listener = StageListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let client = StageConnection::connect(&endpoint).unwrap();
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
        let replacement =
            take_warm_or_connect_downstream(&config, None, &warm, Duration::from_secs(1))
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
    fn downstream_ready_retries_after_failed_handshake() {
        let listener = StageListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = thread::spawn(move || {
            let (first, _) = listener.accept().unwrap();
            drop(first);
            let (mut second, _) = listener.accept().unwrap();
            skippy_protocol::binary::send_ready(&mut second).unwrap();
        });
        let mut config = prefix_cache_test_config();
        config.downstream.as_mut().unwrap().endpoint = endpoint;
        let warm = std::sync::Arc::new(std::sync::Mutex::new(None));

        let shutdown = AtomicBool::new(false);
        let ready = take_ready_downstream(&config, None, &warm, 2, &shutdown)
            .unwrap()
            .expect("downstream should be present");

        assert!(ready.peer_addr().is_ok());
        server.join().unwrap();
    }

    #[test]
    fn downstream_ready_honors_absolute_deadline() {
        let listener = StageListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            thread::sleep(Duration::from_secs(2));
        });
        let mut config = prefix_cache_test_config();
        config.downstream.as_mut().unwrap().endpoint = endpoint;
        let warm = std::sync::Arc::new(std::sync::Mutex::new(None));

        let started = Instant::now();
        let shutdown = AtomicBool::new(false);
        let error = take_ready_downstream(&config, None, &warm, 1, &shutdown).unwrap_err();
        let elapsed = started.elapsed();

        assert!(error.to_string().contains("did not become ready"));
        assert!(elapsed >= Duration::from_millis(800));
        assert!(elapsed < Duration::from_millis(1500));
        server.join().unwrap();
    }

    #[test]
    fn downstream_acquisition_stops_when_shutdown_is_requested() {
        let listener = StageListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let mut config = prefix_cache_test_config();
        config.downstream.as_mut().unwrap().endpoint = endpoint;
        let warm = Arc::new(Mutex::new(None));
        let shutdown = Arc::new(AtomicBool::new(false));
        let task_shutdown = shutdown.clone();
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let task = thread::spawn(move || {
            let result = take_ready_downstream(&config, None, &warm, 30, &task_shutdown);
            let _ = result_tx.send(result);
        });

        let (_unready_downstream, _) = listener.accept().unwrap();
        shutdown.store(true, Ordering::Release);
        let error = result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("downstream acquisition must stop within one second")
            .unwrap_err();

        assert!(error.to_string().contains("cancelled during shutdown"));
        task.join().unwrap();
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

    fn test_message(kind: WireMessageKind, token_count: i32) -> StageWireMessage {
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count,
            state: StageStateHeader::new(kind),
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
    fn token_fill_rejects_counts_beyond_the_sideband_limit() {
        // An upstream peer chooses token_count. Without a sideband the
        // tokens are filled in, so the count must be bounded first.
        let limit = skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES;
        let mut message = test_message(
            WireMessageKind::PrefillEmbd,
            i32::try_from(limit + 1).unwrap(),
        );
        message.state.current_token = 7;
        assert!(token_sideband_or_fill(&message).is_err());

        message.token_count = i32::try_from(limit).unwrap();
        assert_eq!(token_sideband_or_fill(&message).unwrap().len(), limit);
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

        assert!(is_decode_frame_batch_candidate(
            &config,
            &message,
            &[104],
            false,
            false
        ));
    }

    #[test]
    fn decode_frame_batch_candidate_skips_final_output_stage_by_default() {
        let mut config = prefix_cache_test_config();
        config.downstream = None;
        let message = first_decode_message_with_full_prompt_sideband();

        assert!(!is_decode_frame_batch_candidate(
            &config,
            &message,
            &[104],
            false,
            false
        ));
    }

    #[test]
    fn final_output_stage_batches_decode_when_enabled_without_native_mtp() {
        let mut config = prefix_cache_test_config();
        config.downstream = None;
        let message = first_decode_message_with_full_prompt_sideband();

        assert!(is_decode_frame_batch_candidate(
            &config,
            &message,
            &[104],
            false,
            true
        ));
        assert!(
            !is_decode_frame_batch_candidate(&config, &message, &[104], true, true),
            "native MTP drafts only come from the unbatched final-stage path"
        );
    }
}

#[cfg(test)]
mod last_stage_decode_batch_tests {
    use super::resolve_last_stage_decode_batch;

    #[test]
    fn the_planned_policy_applies_when_the_environment_is_unset() {
        assert!(resolve_last_stage_decode_batch(None, Some(true)));
        assert!(!resolve_last_stage_decode_batch(None, Some(false)));
    }

    /// An older peer leaves the field absent. Unbatched is the pre-existing
    /// behaviour, so absent must never read as enabled.
    #[test]
    fn an_absent_plan_stays_unbatched() {
        assert!(!resolve_last_stage_decode_batch(None, None));
    }

    #[test]
    fn the_environment_override_wins_in_both_directions() {
        assert!(resolve_last_stage_decode_batch(Some("1"), Some(false)));
        assert!(resolve_last_stage_decode_batch(Some("on"), None));
        assert!(!resolve_last_stage_decode_batch(Some("0"), Some(true)));
        assert!(!resolve_last_stage_decode_batch(Some("false"), Some(true)));
    }

    /// Deliberately not a fall-through to the plan: the variable decides once
    /// it is set at all, so a typo shows up as "batching did not happen"
    /// rather than quietly honouring a different policy.
    #[test]
    fn an_unparseable_override_does_not_fall_back_to_the_plan() {
        assert!(!resolve_last_stage_decode_batch(Some("yep"), Some(true)));
        assert!(!resolve_last_stage_decode_batch(Some(""), Some(true)));
    }
}
