use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::{thread, time::Duration};

use anyhow::{Result, bail};
use skippy_protocol::{LoadMode, StageConfig};
use skippy_runtime::SamplingConfig;
use tokio::sync::Semaphore;

use crate::binary_transport::DecodeFrameBatcher;
use crate::frontend::SpeculativeDecodeConfig;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::decode_batcher::DecodeBatcher;
use crate::frontend::generation::{
    LocalGeneration, OpenAiBackendMode, OpenAiCacheHints, OpenAiGenerationIds, StageOpenAiBackend,
    TokenControl,
};
use crate::frontend::local_generation::{
    native_mtp_dispatch_counts_for_test, prompt_fits_single_prefill_sample,
};
use crate::frontend::{
    EmbeddedOpenAiRequestDefaults, GenerationAbort, GenerationCommit, GenerationReceipt,
    GenerationReceiptConfig, GenerationReceiptSink, GenerationStart, GenerationTermination,
};
use crate::runtime_state::load_runtime;
use crate::telemetry::{Telemetry, TelemetryLevel};

#[derive(Default)]
struct RecordingReceiptSink {
    receipts: Mutex<Vec<GenerationReceipt>>,
    commits: Mutex<Vec<GenerationCommit>>,
    fail: AtomicBool,
}

impl GenerationReceiptSink for RecordingReceiptSink {
    fn begin(&self, _start: &GenerationStart) -> Result<()> {
        Ok(())
    }

    fn committed(&self, commit: &GenerationCommit) -> Result<()> {
        self.commits.lock().unwrap().push(commit.clone());
        Ok(())
    }

    fn abort(&self, _abort: &GenerationAbort) -> Result<()> {
        Ok(())
    }

    fn record(&self, receipt: &GenerationReceipt) -> Result<()> {
        self.receipts.lock().unwrap().push(receipt.clone());
        if self.fail.load(Ordering::Relaxed) {
            bail!("synthetic generation receipt sink failure");
        }
        Ok(())
    }
}

fn wait_for_receipts(sink: &RecordingReceiptSink, expected: usize) {
    for _ in 0..100 {
        if sink.receipts.lock().unwrap().len() >= expected {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!("timed out waiting for {expected} generation receipts");
}

#[test]
fn single_prefill_sample_requires_prompt_to_fit_session_batch() {
    assert!(!prompt_fits_single_prefill_sample(0, 2048));
    assert!(!prompt_fits_single_prefill_sample(1, 2048));
    assert!(prompt_fits_single_prefill_sample(2048, 2048));
    assert!(!prompt_fits_single_prefill_sample(2049, 2048));
}

#[test]
fn local_native_mtp_decode_uses_non_frame_runtime_api() {
    let (sampled_calls, frame_calls) = native_mtp_dispatch_counts_for_test();
    assert_eq!(sampled_calls, 1);
    assert_eq!(frame_calls, 0);
}

#[test]
fn local_generation_eventually_delivers_receipts_and_cleanup_survives_sink_errors() -> Result<()> {
    let Some(model_path) = std::env::var_os("SKIPPY_GENERATION_RECEIPT_MODEL") else {
        eprintln!("skipping: SKIPPY_GENERATION_RECEIPT_MODEL is not set");
        return Ok(());
    };
    let Some(layer_count) = std::env::var_os("SKIPPY_GENERATION_RECEIPT_MODEL_LAYERS") else {
        eprintln!("skipping: SKIPPY_GENERATION_RECEIPT_MODEL_LAYERS is not set");
        return Ok(());
    };
    let layer_count = layer_count
        .to_string_lossy()
        .parse::<u32>()
        .map_err(|error| anyhow::anyhow!("invalid receipt test layer count: {error}"))?;
    let config = StageConfig {
        run_id: "generation-receipt-test".to_string(),
        topology_id: "generation-receipt-test".to_string(),
        model_id: "generation-receipt-test".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(model_path.to_string_lossy().into_owned()),
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: layer_count,
        ctx_size: 128,
        lane_count: 1,
        n_batch: Some(32),
        n_ubatch: Some(32),
        n_gpu_layers: 0,
        mmap: Some(true),
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: Default::default(),
        filter_tensors_on_load: false,
        selected_device: None,
        kv_cache: None,
        native_mtp_enabled: false,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
    };
    let runtime = load_runtime(&config)?
        .ok_or_else(|| anyhow::anyhow!("receipt test runtime was not loaded"))?;
    let sink = Arc::new(RecordingReceiptSink::default());
    let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
    let speculative = SpeculativeDecodeConfig::default();
    let decode_batcher = DecodeBatcher::new(runtime.clone(), 1);
    let decode_frame_batcher = DecodeFrameBatcher::new(runtime.clone(), 1);
    let backend = StageOpenAiBackend {
        runtime: runtime.clone(),
        config,
        telemetry,
        model_id: "generation-receipt-test".to_string(),
        default_max_tokens: 1,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size: 128,
        mode: OpenAiBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: 0,
        speculative: speculative.clone(),
        generation_limit: Arc::new(Semaphore::new(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_session_locks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(128)),
        hook_policy: None,
        generation_receipt: Some(GenerationReceiptConfig::new(sink.clone())),
        linear_proposal_ingress: None,
        kv: None,
        decode_batcher,
        decode_frame_batcher,
    };
    let sampling = SamplingConfig::default();
    // A multi-token prompt takes the whole-prompt prefill path. Keep this
    // above one token so the test exercises a fresh runtime session before
    // its batch size is queried.
    let prompt_token_ids = [1, 2];
    let ids = OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false);
    let mut emitted = Vec::new();
    backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &prompt_token_ids,
            max_tokens: 1,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &speculative,
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &ids,
        },
        |token_id| {
            emitted.push(token_id);
            Ok(TokenControl::Continue)
        },
    )?;

    wait_for_receipts(&sink, 1);
    let commits = sink.commits.lock().unwrap();
    assert_eq!(commits.len(), emitted.len());
    let mut committed_tokens = Vec::new();
    for (index, commit) in commits.iter().enumerate() {
        assert_eq!(commit.request_id, ids.request_id);
        assert_eq!(commit.session_id, ids.session_id);
        assert_eq!(commit.generated_token_count, index + 1);
        assert_eq!(commit.token_ids.len(), 1);
        committed_tokens.extend_from_slice(&commit.token_ids);
    }
    assert_eq!(committed_tokens, emitted);
    drop(commits);
    let receipts = sink.receipts.lock().unwrap();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].request_id, ids.request_id);
    assert_eq!(receipts[0].session_id, ids.session_id);
    assert_eq!(receipts[0].prompt_token_count, prompt_token_ids.len());
    assert_eq!(receipts[0].generated_token_ids.as_ref(), emitted.as_slice());
    assert_eq!(receipts[0].termination, GenerationTermination::MaxTokens);
    assert!(receipts[0].final_session_position >= prompt_token_ids.len() as u64);
    drop(receipts);
    assert!(
        runtime
            .lock()
            .unwrap()
            .session_stats()
            .lanes
            .iter()
            .all(|lane| lane.session_id.as_deref() != Some(&ids.session_label))
    );

    sink.fail.store(true, Ordering::Relaxed);
    let failing_ids = OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false);
    backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &prompt_token_ids,
            max_tokens: 1,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &speculative,
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &failing_ids,
        },
        |_| Ok(TokenControl::Continue),
    )?;
    wait_for_receipts(&sink, 2);
    assert!(
        runtime
            .lock()
            .unwrap()
            .session_stats()
            .lanes
            .iter()
            .all(|lane| lane.session_id.as_deref() != Some(&failing_ids.session_label))
    );

    Ok(())
}
