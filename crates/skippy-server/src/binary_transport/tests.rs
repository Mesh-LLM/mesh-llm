use super::{
    binary_full_prefill_record_identities, decode_record_tokens_sideband,
    prepare_binary_stage_connection, restore_prefill_decode_as_decode_message,
    token_sideband_or_fill,
};
use std::{
    io,
    net::{TcpListener, TcpStream},
    os::fd::AsRawFd,
    thread,
    time::Duration,
};

use crate::kv_integration::KvStageIntegration;
use skippy_protocol::binary::{
    StageSamplingConfig, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
};
use skippy_protocol::{
    LoadMode, PeerConfig, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
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
