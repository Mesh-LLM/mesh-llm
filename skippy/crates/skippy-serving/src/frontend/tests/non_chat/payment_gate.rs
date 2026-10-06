use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires SKIPPY_WORKLOAD_CLASS=encoder_decoder, SKIPPY_WORKLOAD_MODEL, and a native runtime"]
async fn paid_encoder_decoder_obeys_denied_generation_gate() -> Result<()> {
    use crate::frontend::generation_gate::{GenerationGate, register};

    #[derive(Default)]
    struct RejectGate {
        authorization_calls: AtomicUsize,
        committed: AtomicU64,
    }

    impl GenerationGate for RejectGate {
        fn after_prefill(&self, _: usize, _: u32) -> OpenAiResult<()> {
            self.authorization_calls.fetch_add(1, Ordering::SeqCst);
            Err(OpenAiError::backend("test payment denied"))
        }

        fn before_token(&self) -> OpenAiResult<()> {
            Err(OpenAiError::backend("test payment denied"))
        }

        fn committed_token(&self) -> OpenAiResult<()> {
            self.committed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let fixture = workload_fixture()?.context("encoder-decoder fixture is required")?;
    assert_eq!(fixture.class, CertifiedWorkloadClass::EncoderDecoder);
    let backend =
        support::local_openai_backend(workload_stage_config(&fixture), fixture.model_id.clone())?;
    backend.ensure_local_workload(ModelWorkload::EncoderDecoder)?;
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": fixture.model_id,
        "prompt": "translate English to German: The house is wonderful.",
        "max_tokens": 8,
        "temperature": 0.0,
        "ignore_eos": true
    }))?;

    // Prove that this model and request can generate before testing the gate.
    let baseline = backend.completion(request.clone()).await?;
    assert!(baseline.usage.completion_tokens > 0);

    let gate = Arc::new(RejectGate::default());
    let request_id = uuid::Uuid::new_v4();
    let registration = register(*request_id.as_bytes(), gate.clone())?;
    let result = backend
        .completion_with_context(
            request,
            OpenAiRequestContext::with_request_id(request_id.into()),
        )
        .await;
    // Keep registration alive throughout generation, including its blocking worker.
    drop(registration);

    assert!(
        result.is_err(),
        "encoder-decoder completed despite a rejecting payment gate: authorization_calls={}, committed={}",
        gate.authorization_calls.load(Ordering::SeqCst),
        gate.committed.load(Ordering::SeqCst),
    );
    // A fix may reject paid encoder-decoder models outright or honor the gate.
    assert_eq!(gate.committed.load(Ordering::SeqCst), 0);
    Ok(())
}
