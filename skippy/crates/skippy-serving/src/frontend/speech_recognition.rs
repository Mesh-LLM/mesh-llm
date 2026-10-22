//! A speech-to-text backend for a model that serves nothing else, such as
//! Whisper.

use std::sync::Arc;

use async_trait::async_trait;
use skippy_inference_api::{
    AudioTranscriptionRequest, AudioTranscriptionResponse, AudioTranscriptionSegment,
    AudioTranscriptionTiming, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream,
    InferenceBackend, InferenceError, InferenceRequestContext, InferenceResult, ModelObject,
};
use skippy_runtime::{SpeechRecognitionModel, Transcription, TranscriptionOptions};
use tokio::sync::Semaphore;
use tokio::task;

/// What [`SpeechRecognitionBackend`] needs from a loaded model.
pub trait SpeechTranscriber: Send + Sync {
    /// Decodes an upload to the PCM [`SpeechTranscriber::transcribe`] takes.
    fn decode(&self, audio: &[u8]) -> anyhow::Result<Vec<f32>>;

    fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> anyhow::Result<Transcription>;
}

impl SpeechTranscriber for SpeechRecognitionModel {
    fn decode(&self, audio: &[u8]) -> anyhow::Result<Vec<f32>> {
        skippy_runtime::decode_audio(audio)
    }

    fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
        cancelled: &dyn Fn() -> bool,
    ) -> anyhow::Result<Transcription> {
        SpeechRecognitionModel::transcribe(self, samples, options, cancelled)
    }
}

/// Serves `/v1/audio/transcriptions` and `/v1/audio/translations` from a
/// loaded speech-recognition model. Every other OpenAI surface is refused:
/// the model generates no chat text.
#[derive(Clone)]
pub struct SpeechRecognitionBackend {
    model_id: String,
    model: Arc<dyn SpeechTranscriber>,
    /// Bounds concurrent transcriptions; each holds its own decoder state.
    lanes: Arc<Semaphore>,
}

impl SpeechRecognitionBackend {
    pub fn new(
        model_id: impl Into<String>,
        model: Arc<dyn SpeechTranscriber>,
        concurrency: usize,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model,
            lanes: Arc::new(Semaphore::new(concurrency.max(1))),
        }
    }

    async fn audio_to_text(
        &self,
        request: AudioTranscriptionRequest,
        translate: bool,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        if request.model != self.model_id {
            return Err(InferenceError::invalid_request(format!(
                "model {:?} is not loaded; use {:?}",
                request.model, self.model_id
            )));
        }
        let options = TranscriptionOptions {
            language: request
                .language
                .filter(|language| !language.trim().is_empty()),
            translate,
            prompt: request.prompt.filter(|prompt| !prompt.is_empty()),
            temperature: request.temperature.unwrap_or(0.0),
            threads: 0,
        };
        let _lane = self
            .lanes
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| InferenceError::backend("speech backend is shutting down"))?;
        let model = self.model.clone();
        let cancellation = context.cancellation_token();
        let transcription = task::spawn_blocking(move || {
            let samples = model
                .decode(&request.file)
                .map_err(|error| InferenceError::invalid_request(format!("{error:#}")))?;
            model
                .transcribe(&samples, &options, &|| cancellation.is_cancelled())
                .map_err(|error| InferenceError::backend(format!("{error:#}")))
        })
        .await
        .map_err(|error| {
            InferenceError::backend(format!("transcription task failed: {error}"))
        })??;
        Ok(response(transcription))
    }
}

fn response(transcription: Transcription) -> AudioTranscriptionResponse {
    let text = transcription.text();
    let segments = transcription
        .segments
        .into_iter()
        .enumerate()
        .map(|(id, segment)| AudioTranscriptionSegment {
            id,
            start: segment.start_ms as f64 / 1000.0,
            end: segment.end_ms as f64 / 1000.0,
            text: segment.text,
            no_speech_prob: segment.no_speech_prob,
        })
        .collect();
    AudioTranscriptionResponse {
        text,
        timing: Some(AudioTranscriptionTiming {
            language: transcription.language,
            duration: transcription.duration_seconds,
            segments,
        }),
    }
}

#[async_trait]
impl InferenceBackend for SpeechRecognitionBackend {
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.audio_to_text(request, false, context).await
    }

    async fn audio_translation(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.audio_to_text(request, true, context).await
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        Err(speech_only())
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        Err(speech_only())
    }
}

fn speech_only() -> InferenceError {
    InferenceError::unsupported(
        "this speech-recognition model only serves /v1/audio/transcriptions and /v1/audio/translations",
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use skippy_runtime::TranscriptSegment;
    use tower::ServiceExt;

    use super::*;

    /// Returns a fixed transcript and records the options it was given.
    #[derive(Default)]
    struct FakeModel {
        seen: Mutex<Vec<TranscriptionOptions>>,
    }

    impl SpeechTranscriber for FakeModel {
        fn decode(&self, audio: &[u8]) -> anyhow::Result<Vec<f32>> {
            if audio.starts_with(b"RIFF") {
                Ok(vec![0.0; 16_000])
            } else {
                Err(anyhow::anyhow!("unsupported audio format"))
            }
        }

        fn transcribe(
            &self,
            samples: &[f32],
            options: &TranscriptionOptions,
            cancelled: &dyn Fn() -> bool,
        ) -> anyhow::Result<Transcription> {
            assert!(!cancelled());
            self.seen.lock().unwrap().push(options.clone());
            Ok(Transcription {
                segments: vec![
                    TranscriptSegment {
                        start_ms: 0,
                        end_ms: 600,
                        text: " Hello".into(),
                        no_speech_prob: 0.0,
                    },
                    TranscriptSegment {
                        start_ms: 600,
                        end_ms: 1_000,
                        text: " there.".into(),
                        no_speech_prob: 0.0,
                    },
                ],
                language: options.language.clone().unwrap_or_else(|| "en".into()),
                duration_seconds: samples.len() as f64 / 16_000.0,
            })
        }
    }

    fn upload(model: &str, audio: &[u8], fields: &[(&str, &str)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, value) in [("model", model)].iter().chain(fields) {
            body.extend_from_slice(
                format!(
                    "--b\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(
            b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n",
        );
        body.extend_from_slice(audio);
        body.extend_from_slice(b"\r\n--b--\r\n");
        body
    }

    async fn post(model: Arc<FakeModel>, path: &str, body: Vec<u8>) -> (StatusCode, String) {
        let router = skippy_inference_api::router_for(Arc::new(SpeechRecognitionBackend::new(
            "whisper-test",
            model,
            1,
        )));
        let response = router
            .oneshot(
                Request::post(path)
                    .header("content-type", "multipart/form-data; boundary=b")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn transcription_reaches_the_model_and_joins_segments() {
        let model = Arc::new(FakeModel::default());
        let (status, body) = post(
            model.clone(),
            "/v1/audio/transcriptions",
            upload(
                "whisper-test",
                b"RIFFdata",
                &[
                    ("language", "de"),
                    ("prompt", "Mesh"),
                    ("temperature", "0.2"),
                ],
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["text"], "Hello there.");
        let seen = model.seen.lock().unwrap();
        assert_eq!(seen[0].language.as_deref(), Some("de"));
        assert_eq!(seen[0].prompt.as_deref(), Some("Mesh"));
        assert!(!seen[0].translate);
        assert!((seen[0].temperature - 0.2).abs() < 1e-6);
    }

    #[tokio::test]
    async fn translation_sets_translate_and_reports_timing() {
        let model = Arc::new(FakeModel::default());
        let (status, body) = post(
            model.clone(),
            "/v1/audio/translations",
            upload(
                "whisper-test",
                b"RIFFdata",
                &[("response_format", "verbose_json")],
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["task"], "translate");
        assert_eq!(body["duration"], 1.0);
        assert_eq!(body["segments"][1]["start"], 0.6);
        assert!(model.seen.lock().unwrap()[0].translate);
    }

    #[tokio::test]
    async fn undecodable_audio_is_a_client_error() {
        let (status, body) = post(
            Arc::new(FakeModel::default()),
            "/v1/audio/transcriptions",
            upload("whisper-test", b"OggSdata", &[]),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("unsupported audio format"), "{body}");
    }

    #[tokio::test]
    async fn other_models_and_chat_are_refused() {
        let (status, _) = post(
            Arc::new(FakeModel::default()),
            "/v1/audio/transcriptions",
            upload("other", b"RIFFdata", &[]),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let router = skippy_inference_api::router_for(Arc::new(SpeechRecognitionBackend::new(
            "whisper-test",
            Arc::new(FakeModel::default()),
            1,
        )));
        let response = router
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"whisper-test","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.status().is_client_error());
    }
}
