use super::*;

use openai_frontend::{
    AudioFormat, AudioSpeechRequest, AudioTranscriptionRequest, EmbeddingInput, EmbeddingOutput,
    EmbeddingsRequest, RerankDocument, RerankRequest,
};
use skippy_runtime::ModelWorkload;

const MODEL_ENV: &str = "SKIPPY_WORKLOAD_MODEL";
const MODEL_ID_ENV: &str = "SKIPPY_WORKLOAD_MODEL_ID";
const CLASS_ENV: &str = "SKIPPY_WORKLOAD_CLASS";
const PROJECTOR_ENV: &str = "SKIPPY_WORKLOAD_PROJECTOR";
const MEDIA_ENV: &str = "SKIPPY_WORKLOAD_MEDIA";
const LAYER_END_ENV: &str = "SKIPPY_WORKLOAD_LAYER_END";
const CTX_SIZE_ENV: &str = "SKIPPY_WORKLOAD_CTX_SIZE";
const MAX_TOKENS_ENV: &str = "SKIPPY_WORKLOAD_MAX_TOKENS";
const N_GPU_LAYERS_ENV: &str = "SKIPPY_WORKLOAD_N_GPU_LAYERS";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CertifiedWorkloadClass {
    Embedding,
    Rerank,
    EncoderDecoder,
    Ocr,
    SpeechSynthesis,
    SpeechRecognition,
}

impl CertifiedWorkloadClass {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "embedding" => Ok(Self::Embedding),
            "rerank" => Ok(Self::Rerank),
            "encoder_decoder" => Ok(Self::EncoderDecoder),
            "ocr" => Ok(Self::Ocr),
            "speech_synthesis" => Ok(Self::SpeechSynthesis),
            "speech_recognition" => Ok(Self::SpeechRecognition),
            other => bail!("unsupported {CLASS_ENV} value {other:?}"),
        }
    }

    fn requires_projector(self) -> bool {
        matches!(
            self,
            Self::Ocr | Self::SpeechSynthesis | Self::SpeechRecognition
        )
    }

    fn requires_media(self) -> bool {
        matches!(self, Self::Ocr | Self::SpeechRecognition)
    }

    fn staging_label(self) -> Option<&'static str> {
        match self {
            Self::Embedding => Some("embedding"),
            Self::Rerank => Some("rerank"),
            Self::EncoderDecoder => Some("encoder_decoder"),
            Self::SpeechSynthesis => Some("speech_synthesis"),
            Self::Ocr | Self::SpeechRecognition => None,
        }
    }
}

struct WorkloadFixture {
    class: CertifiedWorkloadClass,
    model_id: String,
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    media_path: Option<PathBuf>,
    layer_end: u32,
    ctx_size: u32,
    max_tokens: u32,
    n_gpu_layers: i32,
}

fn required_file(name: &str) -> Result<PathBuf> {
    let path = PathBuf::from(env::var_os(name).context(format!("{name} is required"))?);
    if !path.is_file() {
        bail!("{name} does not point at a file: {}", path.display());
    }
    Ok(path)
}

fn parse_env<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    env::var(name).map_or(Ok(default), |value| {
        value
            .parse::<T>()
            .map_err(|error| anyhow!("parse {name}={value:?}: {error}"))
    })
}

fn workload_fixture() -> Result<Option<WorkloadFixture>> {
    let Some(class) = env::var(CLASS_ENV).ok() else {
        return Ok(None);
    };
    let class = CertifiedWorkloadClass::parse(&class)?;
    let projector_path = class
        .requires_projector()
        .then(|| required_file(PROJECTOR_ENV))
        .transpose()?;
    let media_path = class
        .requires_media()
        .then(|| required_file(MEDIA_ENV))
        .transpose()?;
    let model_path = required_file(MODEL_ENV)?;
    let layer_end = parse_env(LAYER_END_ENV, 1_u32)?;
    if layer_end == 0 {
        bail!("{LAYER_END_ENV} must be positive");
    }
    Ok(Some(WorkloadFixture {
        class,
        model_id: env::var(MODEL_ID_ENV).unwrap_or_else(|_| "workload-smoke".to_string()),
        model_path,
        projector_path,
        media_path,
        layer_end,
        ctx_size: parse_env(CTX_SIZE_ENV, 2048)?,
        max_tokens: parse_env(MAX_TOKENS_ENV, 32)?,
        n_gpu_layers: parse_env(N_GPU_LAYERS_ENV, 0)?,
    }))
}

fn workload_stage_config(fixture: &WorkloadFixture) -> StageConfig {
    StageConfig {
        run_id: "workload-certification".to_string(),
        topology_id: "workload-certification-local".to_string(),
        model_id: fixture.model_id.clone(),
        model_path: Some(fixture.model_path.to_string_lossy().to_string()),
        projector_path: fixture
            .projector_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: fixture.layer_end,
        ctx_size: fixture.ctx_size,
        lane_count: 1,
        n_batch: Some(fixture.ctx_size.min(2048)),
        n_ubatch: Some(fixture.ctx_size.min(2048)),
        n_gpu_layers: fixture.n_gpu_layers,
        kv_offload: (fixture.n_gpu_layers == 0).then_some(false),
        op_offload: (fixture.n_gpu_layers == 0).then_some(false),
        selected_device: (fixture.n_gpu_layers == 0).then(|| StageDevice {
            backend_device: "CPU".to_string(),
            stable_id: None,
            index: None,
            vram_bytes: None,
        }),
        filter_tensors_on_load: false,
        native_mtp_enabled: false,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        ..StageConfig::default()
    }
}

fn assert_vectors_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    let maximum_delta = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum_delta <= 1e-5, "embedding delta {maximum_delta}");
}

async fn certify_embedding(backend: &StageOpenAiBackend) -> Result<()> {
    let info = backend.ensure_local_workload(ModelWorkload::Embedding)?;
    assert!(info.output_dimensions > 0);
    let request = EmbeddingsRequest {
        model: backend.model_id.clone(),
        input: EmbeddingInput::Texts(vec![
            "search_query: distributed inference".to_string(),
            "search_document: GPUs collaborate over a mesh".to_string(),
        ]),
        encoding_format: "float".to_string(),
        dimensions: None,
        user: None,
    };
    let first = backend
        .embeddings(request.clone(), OpenAiRequestContext::new())
        .await?;
    let second = backend
        .embeddings(request, OpenAiRequestContext::new())
        .await?;
    assert_eq!(first.object, "list");
    assert_eq!(first.data.len(), 2);
    assert_eq!(first.data.len(), second.data.len());
    assert!(first.usage.prompt_tokens > 0);
    for (left, right) in first.data.iter().zip(&second.data) {
        let (EmbeddingOutput::Float(left), EmbeddingOutput::Float(right)) =
            (&left.embedding, &right.embedding)
        else {
            bail!("float embedding request returned a non-float payload");
        };
        assert_eq!(left.len(), info.output_dimensions as usize);
        let norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() <= 1e-4, "embedding norm {norm}");
        assert_vectors_close(left, right);
    }
    Ok(())
}

async fn certify_rerank(backend: &StageOpenAiBackend) -> Result<()> {
    let info = backend.ensure_local_workload(ModelWorkload::Rerank)?;
    assert_eq!(info.classifier_outputs, 1);
    let request = RerankRequest {
        model: backend.model_id.clone(),
        query: "distributed GPU inference".to_string(),
        documents: vec![
            RerankDocument::Text("GPUs share one language model over a mesh".to_string()),
            RerankDocument::Text("A recipe for tomato soup".to_string()),
        ],
        top_n: None,
        return_documents: true,
    };
    let first = backend
        .rerank(request.clone(), OpenAiRequestContext::new())
        .await?;
    let second = backend.rerank(request, OpenAiRequestContext::new()).await?;
    assert_eq!(first.results.len(), 2);
    assert_eq!(first.results.len(), second.results.len());
    assert!(first.usage.prompt_tokens > 0);
    for (left, right) in first.results.iter().zip(&second.results) {
        assert_eq!(left.index, right.index);
        assert!(left.relevance_score.is_finite());
        assert!((left.relevance_score - right.relevance_score).abs() <= 1e-6);
        assert!(left.document.is_some());
    }
    Ok(())
}

async fn certify_encoder_decoder(backend: &StageOpenAiBackend, max_tokens: u32) -> Result<()> {
    backend.ensure_local_workload(ModelWorkload::EncoderDecoder)?;
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": backend.model_id,
        "prompt": "translate English to German: The house is wonderful.",
        "max_tokens": max_tokens,
        "temperature": 0.0
    }))?;
    let first = backend.completion(request.clone()).await?;
    let second = backend.completion(request).await?;
    assert!(!first.choices[0].text.trim().is_empty());
    assert_eq!(first.choices[0].text, second.choices[0].text);
    assert!(first.usage.prompt_tokens > 0);
    assert!(first.usage.completion_tokens > 0);
    Ok(())
}

fn media_chat_request(fixture: &WorkloadFixture) -> Result<ChatCompletionRequest> {
    let path = fixture
        .media_path
        .as_ref()
        .context("media path is required")?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(fs::read(path)?);
    serde_json::from_value(json!({
        "model": fixture.model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Read all visible text. Return only the transcription."},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{encoded}")}}
            ]
        }],
        "max_tokens": fixture.max_tokens,
        "temperature": 0.0
    }))
    .context("build OCR request")
}

async fn certify_ocr(backend: &StageOpenAiBackend, fixture: &WorkloadFixture) -> Result<()> {
    let first = backend
        .chat_completion(media_chat_request(fixture)?)
        .await?;
    let second = backend
        .chat_completion(media_chat_request(fixture)?)
        .await?;
    let first_text = first.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or_default()
        .trim();
    let second_text = second.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or_default()
        .trim();
    assert!(!first_text.is_empty());
    assert_eq!(first_text, second_text);
    Ok(())
}

async fn certify_speech_synthesis(backend: &StageOpenAiBackend) -> Result<()> {
    assert!(
        backend
            .runtime
            .lock()
            .expect("runtime mutex poisoned")
            .supports_speech_synthesis()
    );
    let unsupported_voice = backend
        .audio_speech(
            AudioSpeechRequest {
                model: backend.model_id.clone(),
                input: "The mesh is ready.".to_string(),
                voice: "alloy".to_string(),
                response_format: AudioFormat::Wav,
                speed: 1.0,
            },
            OpenAiRequestContext::new(),
        )
        .await
        .expect_err("speaker selection must not be interpreted as a language");
    assert_eq!(
        unsupported_voice.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
    assert_eq!(
        unsupported_voice.body().error.param.as_deref(),
        Some("voice")
    );
    let response = backend
        .audio_speech(
            AudioSpeechRequest {
                model: backend.model_id.clone(),
                input: "The mesh is ready.".to_string(),
                voice: "default".to_string(),
                response_format: AudioFormat::Wav,
                speed: 1.0,
            },
            OpenAiRequestContext::new(),
        )
        .await?;
    assert_eq!(response.content_type, "audio/wav");
    assert!(response.bytes.len() > 44);
    assert_eq!(&response.bytes[..4], b"RIFF");
    assert_eq!(&response.bytes[8..12], b"WAVE");
    Ok(())
}

async fn certify_speech_recognition(
    backend: &StageOpenAiBackend,
    fixture: &WorkloadFixture,
) -> Result<()> {
    let media_path = fixture
        .media_path
        .as_ref()
        .context("media path is required")?;
    let response = backend
        .audio_transcription(
            AudioTranscriptionRequest {
                model: fixture.model_id.clone(),
                file: fs::read(media_path)?,
                filename: media_path
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string()),
                language: Some("en".to_string()),
                prompt: None,
                response_format: "json".to_string(),
                temperature: Some(0.0),
            },
            OpenAiRequestContext::new(),
        )
        .await?;
    assert!(!response.text.trim().is_empty());
    Ok(())
}

fn assert_unsupported_staging(
    backend: &StageOpenAiBackend,
    fixture: &WorkloadFixture,
    expected: &str,
) -> Result<()> {
    let mut config = workload_stage_config(fixture);
    config.filter_tensors_on_load = true;
    config.layer_end = (fixture.layer_end / 2).max(1);
    config.downstream = Some(PeerConfig {
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        endpoint: "127.0.0.1:1".to_string(),
    });
    let runtime = backend.runtime.lock().expect("runtime mutex poisoned");
    let error = match reject_unsupported_staged_workload(&config, &runtime.model) {
        Ok(()) => bail!("unsupported workload staging did not fail closed"),
        Err(error) => error,
    };
    let rendered = format!("{error:#}");
    assert!(
        rendered.contains("unsupported staged workload") && rendered.contains(expected),
        "unexpected staged workload error: {rendered}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_non_chat_class_smoke_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = workload_fixture()? else {
        return Ok(());
    };
    let backend =
        support::local_openai_backend(workload_stage_config(&fixture), fixture.model_id.clone())?;
    match fixture.class {
        CertifiedWorkloadClass::Embedding => certify_embedding(&backend).await?,
        CertifiedWorkloadClass::Rerank => certify_rerank(&backend).await?,
        CertifiedWorkloadClass::EncoderDecoder => {
            certify_encoder_decoder(&backend, fixture.max_tokens).await?
        }
        CertifiedWorkloadClass::Ocr => certify_ocr(&backend, &fixture).await?,
        CertifiedWorkloadClass::SpeechSynthesis => certify_speech_synthesis(&backend).await?,
        CertifiedWorkloadClass::SpeechRecognition => {
            certify_speech_recognition(&backend, &fixture).await?
        }
    }
    if let Some(staging_label) = fixture.class.staging_label() {
        assert_unsupported_staging(&backend, &fixture, staging_label)?;
    }
    Ok(())
}
