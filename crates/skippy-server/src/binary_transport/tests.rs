use super::{
    binary_full_prefill_record_identities, decode_record_tokens_sideband,
    is_decode_frame_batch_candidate, native_mtp_enabled_from, prepare_binary_stage_connection,
    restore_prefill_decode_as_decode_message, token_sideband_or_fill,
};
use std::{
    io,
    net::{TcpListener, TcpStream},
    os::fd::AsRawFd,
    thread,
    time::Duration,
};

use crate::kv_integration::KvStageIntegration;
use crate::runtime_state::RuntimeState;
use skippy_protocol::binary::{
    StageReplyStats, StageSamplingConfig, StageStateHeader, StageWireMessage, WireActivationDType,
    WireMessageKind,
};
use skippy_protocol::{
    LoadMode, PeerConfig, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};

type BinaryEvictionFn = fn(
    &mut RuntimeState,
    Option<&std::sync::Arc<KvStageIntegration>>,
    &str,
    super::BinaryProactiveEvictionPlan,
) -> anyhow::Result<super::BinaryProactiveEviction>;

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
fn native_mtp_enabled_flag_defaults_on_and_accepts_false_values() {
    assert!(native_mtp_enabled_from(None));
    assert!(native_mtp_enabled_from(Some("1")));
    assert!(native_mtp_enabled_from(Some("true")));
    assert!(!native_mtp_enabled_from(Some("0")));
    assert!(!native_mtp_enabled_from(Some("false")));
    assert!(!native_mtp_enabled_from(Some(" disabled ")));
}

#[test]
fn request_summary_tracks_verify_span_compute_ms() {
    let config = prefix_cache_test_config();
    let mut summary = super::BinaryRequestSummary::default();
    let verify = test_message(WireMessageKind::VerifySpan, 2);
    let decode = test_message(WireMessageKind::DecodeEmbd, 1);

    summary.observe(summary_observation(&config, &verify, 12.5));
    summary.observe(summary_observation(&config, &decode, 7.0));

    assert_eq!(summary.verify_span_count, 1);
    assert_eq!(summary.verify_span_token_count, 2);
    assert_eq!(summary.verify_span_max_tokens, 2);
    assert_eq!(summary.verify_span_compute_ms, 12.5);
    assert_eq!(summary.verify_span_input_activation_decode_ms, 1.25);
    assert_eq!(summary.verify_span_runtime_lock_hold_ms, 2.5);
    assert_eq!(summary.verify_span_upstream_reply_ms, 0.75);
    assert_eq!(summary.compute_ms, 19.5);
    assert_eq!(summary.input_activation_decode_ms, 2.5);
    assert_eq!(summary.runtime_lock_hold_ms, 5.0);
    assert_eq!(summary.upstream_reply_ms, 1.5);
}

#[test]
fn request_summary_tracks_auto_align_totals() {
    let config = prefix_cache_test_config();
    let mut summary = super::BinaryRequestSummary::default();
    let verify = test_message(WireMessageKind::VerifySpan, 2);
    let decode = test_message(WireMessageKind::DecodeEmbd, 1);

    let mut verify_observation = summary_observation(&config, &verify, 12.5);
    verify_observation.session_auto_align_count = 1;
    verify_observation.session_auto_align_ms = 0.75;
    verify_observation.session_auto_align_trimmed_tokens = 1;
    summary.observe(verify_observation);

    let mut decode_observation = summary_observation(&config, &decode, 7.0);
    decode_observation.session_auto_align_count = 1;
    decode_observation.session_auto_align_ms = 1.25;
    decode_observation.session_auto_align_trimmed_tokens = 2;
    summary.observe(decode_observation);

    assert_eq!(summary.session_auto_align_count, 2);
    assert_eq!(summary.session_auto_align_ms, 2.0);
    assert_eq!(summary.session_auto_align_trimmed_tokens, 3);
    assert_eq!(summary.verify_span_session_auto_align_count, 1);
    assert_eq!(summary.verify_span_session_auto_align_ms, 0.75);
    assert_eq!(summary.verify_span_session_auto_align_trimmed_tokens, 1);
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

#[test]
fn binary_decode_work_requires_proactive_resident_eviction() {
    assert!(
        super::binary_proactive_eviction_plan(WireMessageKind::PrefillFinalEmbd, false, 128)
            .required
    );
    assert!(super::binary_proactive_eviction_plan(WireMessageKind::DecodeEmbd, false, 1).required);
    assert!(
        super::binary_proactive_eviction_plan(WireMessageKind::DecodeReplayEmbd, false, 64)
            .required
    );
    assert!(
        !super::binary_proactive_eviction_plan(WireMessageKind::PrefillEmbd, false, 128).required
    );
    assert!(!super::binary_proactive_eviction_plan(WireMessageKind::DecodeEmbd, true, 1).required);
    assert!(!super::binary_proactive_eviction_plan(WireMessageKind::DecodeEmbd, false, 0).required);
    assert!(
        !super::binary_proactive_eviction_plan(WireMessageKind::TryRestorePrefillDecode, false, 1)
            .required
    );
}

#[test]
fn one_chunk_prefill_final_admits_session_before_proactive_eviction() {
    let plan = super::binary_proactive_eviction_plan(WireMessageKind::PrefillFinalEmbd, false, 1);

    assert!(plan.required);
    assert!(plan.ensure_session_before_eviction);
}

#[test]
fn required_binary_proactive_eviction_is_fallible_before_decode() {
    fn accepts_fallible_eviction(_evict: BinaryEvictionFn) {}

    accepts_fallible_eviction(super::evict_binary_resident_prefix_for_decode);
}

fn prefix_cache_test_config() -> StageConfig {
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

fn summary_observation<'a>(
    config: &'a StageConfig,
    message: &'a StageWireMessage,
    compute_ms: f64,
) -> super::BinaryMessageObservation<'a> {
    super::BinaryMessageObservation {
        config,
        message,
        reply_stats: StageReplyStats::default(),
        compute_ms,
        forward_write_ms: 0.0,
        downstream_wait_ms: 0.0,
        upstream_reply_ms: 0.75,
        message_elapsed_ms: compute_ms,
        input_activation_decode_ms: 1.25,
        forward_activation_encode_ms: 0.0,
        runtime_lock_hold_ms: 2.5,
        input_activation_bytes: 0,
        output_activation_bytes: 0,
        prefill_credit_limit: 0,
        pending_prefill_replies_before: 0,
        pending_prefill_replies_after: 0,
        credit_wait_count: 0,
        deferred_prefill_replies_drained: 0,
        session_auto_align_count: 0,
        session_auto_align_ms: 0.0,
        session_auto_align_trimmed_tokens: 0,
        verify_span_pre_compute_ms: 0.25,
        verify_span_post_compute_ms: 0.5,
        verify_span_pre_reply_ms: 0.0,
        verify_span_after_reply_ms: 0.0,
        upstream_message_wait_ms: 0.0,
    }
}

fn prefill_message() -> StageWireMessage {
    StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32),
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

fn first_decode_message_with_full_prompt_sideband() -> StageWireMessage {
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

#[test]
fn binary_full_prefill_record_plan_includes_shared_prefix_candidate() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let message = prefill_message();
    let recorded_tokens = (0..2214).collect::<Vec<_>>();
    let mut lookup_tokens = recorded_tokens.clone();
    lookup_tokens.extend(100_000..100_017);

    let record_plan =
        binary_full_prefill_record_identities(&kv, &config, "session", &message, &recorded_tokens);
    let base = super::binary_message_base(&config, "session", &message);
    let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

    let record_counts = record_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();
    let lookup_counts = lookup_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();

    assert_eq!(record_counts, vec![2214, 2176]);
    assert!(lookup_counts.contains(&2176));

    let recorded_shared = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("binary full-prefill record plan should include shared grid prefix");
    let lookup_shared = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("lookup plan should probe shared grid prefix");
    let recorded_exact = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2214)
        .expect("binary full-prefill record plan should keep exact first prompt");
    let lookup_exact = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2231)
        .expect("lookup plan should probe exact second prompt");

    assert_eq!(recorded_shared.page_id, lookup_shared.page_id);
    assert_ne!(recorded_exact.page_id, lookup_exact.page_id);
}
