use super::*;
use crate::frontend::generation_gate::{GenerationGate, register};
use std::sync::atomic::AtomicBool;

struct TestGate {
    prefilled: tokio::sync::Notify,
    released: AtomicBool,
    expired: AtomicBool,
    input: AtomicUsize,
    output: AtomicU64,
}
impl GenerationGate for TestGate {
    fn after_prefill(&self, input: usize, _: u32) -> OpenAiResult<()> {
        self.input.store(input, Ordering::Release);
        self.prefilled.notify_one();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !self.released.load(Ordering::Acquire) {
            if std::time::Instant::now() > deadline {
                return Err(OpenAiError::backend("test approval timed out"));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if self.expired.load(Ordering::Acquire) {
            return Err(OpenAiError::backend("test invoice expired"));
        }
        Ok(())
    }
    fn committed_token(&self) -> OpenAiResult<()> {
        self.output.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
    fn committed_tokens(&self) -> u64 {
        self.output.load(Ordering::Acquire)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MESH_PAYMENT_TEST_MODEL and a CPU native runtime bundle"]
async fn payments_real_model_prefills_before_gate_and_streams_usage_after_release() -> Result<()> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        let directory = PathBuf::from(std::env::var("MESH_PAYMENT_TEST_RUNTIME")?);
        let manifest: Value = serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
        let libraries = manifest["runtime"]["libraries"]
            .as_array()
            .context("runtime library list")?
            .iter()
            .map(|path| {
                path.as_str()
                    .map(|p| directory.join(p))
                    .context("runtime library path")
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: this opt-in test requires a bundle built from this checkout.
        unsafe {
            skippy_runtime::load_native_runtime_libraries(libraries)?;
        }
    }
    let path = std::env::var("MESH_PAYMENT_TEST_MODEL")?;
    let config = StageConfig {
        model_id: "payment-smoke".into(),
        model_path: Some(path),
        layer_start: 0,
        layer_end: 30,
        ctx_size: 512,
        lane_count: 1,
        n_gpu_layers: 0,
        load_mode: LoadMode::RuntimeSlice,
        ..StageConfig::default()
    };
    let backend = Arc::new(multimodal::local_openai_backend(config)?);
    let gate = Arc::new(TestGate {
        prefilled: tokio::sync::Notify::new(),
        released: AtomicBool::new(false),
        expired: AtomicBool::new(false),
        input: AtomicUsize::new(0),
        output: AtomicU64::new(0),
    });
    let id = [47; 16];
    let _registration = register(id, gate.clone())?;
    let generating_backend = backend.clone();
    let generation =
        tokio::spawn(async move { collect_completion(generating_backend.as_ref(), id).await });
    tokio::time::timeout(Duration::from_secs(30), gate.prefilled.notified()).await?;
    assert!(gate.input.load(Ordering::Acquire) > 0);
    assert_eq!(gate.output.load(Ordering::Acquire), 0);
    assert!(!generation.is_finished());
    gate.released.store(true, Ordering::Release);
    let (text, tokens) = tokio::time::timeout(Duration::from_secs(30), generation).await???;
    assert!(!text.is_empty());
    assert!(tokens > 0 && tokens <= 8);
    assert_eq!(u64::from(tokens), gate.output.load(Ordering::Acquire));
    let expired = Arc::new(TestGate {
        prefilled: tokio::sync::Notify::new(),
        released: AtomicBool::new(false),
        expired: AtomicBool::new(true),
        input: AtomicUsize::new(0),
        output: AtomicU64::new(0),
    });
    let expired_id = [48; 16];
    let _expired_registration = register(expired_id, expired.clone())?;
    let expiring_backend = backend.clone();
    let generation =
        tokio::spawn(
            async move { collect_completion(expiring_backend.as_ref(), expired_id).await },
        );
    tokio::time::timeout(Duration::from_secs(30), expired.prefilled.notified()).await?;
    assert_eq!(expired.output.load(Ordering::Acquire), 0);
    expired.released.store(true, Ordering::Release);
    assert!(
        tokio::time::timeout(Duration::from_secs(30), generation)
            .await??
            .is_err()
    );
    assert_eq!(expired.output.load(Ordering::Acquire), 0);
    // The same backend has one generation lane. A new request proves the
    // rejected gate released admission and native generation state.
    let (text, tokens) = tokio::time::timeout(
        Duration::from_secs(30),
        collect_completion(backend.as_ref(), [49; 16]),
    )
    .await??;
    assert!(!text.is_empty());
    assert!(tokens > 0 && tokens <= 8);
    Ok(())
}

async fn collect_completion(backend: &impl OpenAiBackend, id: [u8; 16]) -> Result<(String, u32)> {
    use futures_util::StreamExt;
    let request: CompletionRequest = serde_json::from_value(
        json!({"model":"mm-smoke","prompt":"The sun is","max_tokens":8,"stream":true,"stream_options":{"include_usage":true},"temperature":0}),
    )?;
    let context = OpenAiRequestContext::with_request_id(uuid::Uuid::from_bytes(id).into());
    let mut stream = backend.completion_stream(request, context).await?;
    let mut tokens = 0;
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        for choice in chunk.choices {
            text.push_str(&choice.text);
        }
        if let Some(usage) = chunk.usage {
            tokens = usage.completion_tokens;
        }
    }
    Ok((text, tokens))
}
