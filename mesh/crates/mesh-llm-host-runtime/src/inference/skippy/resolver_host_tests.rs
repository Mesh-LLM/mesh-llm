use super::{SkippyModelHandle, SkippyModelLoadOptions};
use anyhow::Result;
use mesh_llm_skippy_adapter::config::{
    SkippyConfigResolveRequest, resolve_skippy_config_for_selector,
};
use skippy_inference_api::OpenAiBackend;
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

#[derive(Default)]
struct RecordingNativeProposalIngress {
    proposals: AtomicUsize,
    reports: Mutex<Vec<skippy_serving::LinearProposalReceipt>>,
}

impl skippy_serving::LinearProposalIngress for RecordingNativeProposalIngress {
    fn propose(
        &self,
        _query: skippy_serving::LinearProposalQuery,
    ) -> anyhow::Result<skippy_serving::LinearProposalSourceResponse> {
        self.proposals.fetch_add(1, Ordering::Relaxed);
        let decision_id = skippy_serving::OpaqueProposalDecisionId::new(vec![1])?;
        Ok(skippy_serving::LinearProposalSourceResponse::new(Some(
            skippy_serving::LinearProposal::new(decision_id, vec![0]),
        )))
    }

    fn report(&self, receipt: &skippy_serving::LinearProposalReceipt) -> anyhow::Result<()> {
        self.reports.lock().unwrap().push(receipt.clone());
        Ok(())
    }
}

struct NoopGenerationReceiptSink;

impl skippy_serving::frontend::GenerationReceiptSink for NoopGenerationReceiptSink {
    fn begin(&self, _start: &skippy_serving::frontend::GenerationStart) -> Result<()> {
        Ok(())
    }

    fn committed(&self, _commit: &skippy_serving::frontend::GenerationCommit) -> Result<()> {
        Ok(())
    }

    fn abort(&self, _abort: &skippy_serving::frontend::GenerationAbort) -> Result<()> {
        Ok(())
    }

    fn record(&self, _receipt: &skippy_serving::frontend::GenerationReceipt) -> Result<()> {
        Ok(())
    }
}

struct RecordingNativeHooksFactory {
    ingress: Arc<RecordingNativeProposalIngress>,
}

impl skippy_serving::serving_hooks::ModelServingHooksFactory for RecordingNativeHooksFactory {
    fn create(
        &self,
        _tokenizer: skippy_serving::TokenizerCapability,
        _extra_generation_sink: Option<
            Arc<dyn skippy_serving::frontend::GenerationLifecycleIngress>,
        >,
    ) -> Result<skippy_serving::serving_hooks::ModelServingHooks> {
        let source: Arc<dyn skippy_serving::LinearProposalIngress> = self.ingress.clone();
        let ingress = skippy_serving::frontend::LinearProposalIngressConfig::new(
            source,
            Duration::from_millis(25),
            1,
        )?;
        Ok(skippy_serving::serving_hooks::ModelServingHooks::new(
            skippy_serving::frontend::GenerationReceiptConfig::new(Arc::new(
                NoopGenerationReceiptSink,
            )),
            ingress,
        ))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_single_stage_serving_delivers_target_authoritative_native_receipts() -> Result<()> {
    let Some(model_path) = std::env::var_os("SKIPPY_NATIVE_PLUGIN_MODEL") else {
        eprintln!("skipping: SKIPPY_NATIVE_PLUGIN_MODEL is not set");
        return Ok(());
    };
    let model_path = Path::new(&model_path);
    let model_bytes = std::fs::metadata(model_path)?.len();
    let model_id = "Qwen/Qwen3-0.6B:Q4_K_M";
    let mesh_config: mesh_llm_config::MeshConfig = toml::from_str(
        r#"
[defaults.speculative]
strategy = "ngram-suffix"
ngram_min = 2
ngram_max = 8
ngram_max_proposal_tokens = 1
"#,
    )?;
    let resolved = resolve_skippy_config_for_selector(
        SkippyConfigResolveRequest {
            mesh_config: &mesh_config,
            model_id,
            model_path,
            model_bytes,
            allocatable_memory_bytes: None,
            request_defaults: None,
            package_generation: None,
            compact_meta: None,
        },
        Some(model_id),
    )?;
    let embedded_openai = resolved.to_embedded_openai_args(0, false)?;
    let ingress = Arc::new(RecordingNativeProposalIngress::default());
    let factory: skippy_serving::serving_hooks::SharedModelServingHooksFactory =
        Arc::new(RecordingNativeHooksFactory {
            ingress: Arc::clone(&ingress),
        });
    let mut options = SkippyModelLoadOptions::for_direct_gguf(model_id, model_path)
        .with_ctx_size(256)
        .with_embedded_openai(embedded_openai)
        .with_serving_hooks_factory(Some(factory));
    options.n_gpu_layers = 0;

    let handle = SkippyModelHandle::load_with_hooks(
        options,
        None,
        crate::runtime::survey::SurveyTelemetry::disabled(),
    )?;
    let request = serde_json::from_value(serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 2,
        "temperature": 0.0
    }))?;
    let response = handle.chat_completion(request).await;
    handle.shutdown();
    response?;

    for _ in 0..100 {
        if !ingress.reports.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(ingress.proposals.load(Ordering::Relaxed) > 0);
    let reports = ingress.reports.lock().unwrap();
    assert!(
        !reports.is_empty(),
        "native proposal report was not delivered"
    );
    for receipt in reports.iter() {
        assert_eq!(receipt.proposal_token_count, 1);
        assert!(receipt.verification_rows > 0);
        assert!(receipt.accepted_proposal_tokens <= receipt.proposal_token_count);
        assert!(!receipt.committed_tokens.is_empty());
        assert_eq!(
            receipt.canonical_prediction_count,
            receipt.committed_tokens.len()
        );
        assert_eq!(
            receipt.verification_rows,
            receipt.verification_row_predictions.len()
        );
        assert!(receipt.canonical_prediction_count <= receipt.verification_rows);
        assert!(receipt.canonical_position >= receipt.base_position);
        assert!(receipt.canonical_position <= receipt.position_after_verification);
    }
    Ok(())
}
