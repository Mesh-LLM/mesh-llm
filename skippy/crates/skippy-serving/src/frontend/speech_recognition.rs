//! A speech-to-text backend for a model that serves nothing else, such as
//! Whisper.

use std::sync::Arc;

use async_trait::async_trait;
use skippy_inference_api::realtime::{
    RealtimeInput, RealtimeOutput, RealtimeTranscriptionConfig, RealtimeTranscriptionSession,
};
use skippy_inference_api::{
    AudioTranscriptionRequest, AudioTranscriptionResponse, AudioTranscriptionSegment,
    AudioTranscriptionTiming, CancellationToken, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStream, InferenceBackend, InferenceError, InferenceRequestContext,
    InferenceResult, ModelObject,
};
use skippy_runtime::{
    PcmResampler, SpeechRecognitionModel, SpeechStream, SpeechStreamConfig, SpeechStreamEvent,
    Transcription, TranscriptionOptions, TurnDetection,
};
use tokio::sync::{Semaphore, mpsc};
use tokio::task;

/// What [`SpeechRecognitionBackend`] needs from a loaded model.
pub trait SpeechTranscriber: Send + Sync {
    /// Decodes an upload to the PCM [`SpeechTranscriber::transcribe`] takes.
    fn decode(&self, audio: &[u8]) -> anyhow::Result<Vec<f32>>;

    fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
        cancelled: &(dyn Fn() -> bool + Sync),
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
        cancelled: &(dyn Fn() -> bool + Sync),
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
        let lane = self
            .lanes
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| InferenceError::backend("speech backend is shutting down"))?;
        let model = self.model.clone();
        let cancellation = context.cancellation_token();
        let transcription = task::spawn_blocking(move || {
            // The lane is held for as long as the work runs, even when the
            // request future is dropped and nobody awaits the result.
            let _lane = lane;
            if cancellation.is_cancelled() {
                return Err(InferenceError::backend("transcription cancelled"));
            }
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

    async fn realtime_transcription(
        &self,
        config: RealtimeTranscriptionConfig,
        context: InferenceRequestContext,
    ) -> InferenceResult<RealtimeTranscriptionSession> {
        if config.model != self.model_id {
            return Err(InferenceError::invalid_request(format!(
                "model {:?} is not loaded; use {:?}",
                config.model, self.model_id
            )));
        }
        let (session, input, output) = RealtimeTranscriptionSession::channels();
        let live = LiveSession {
            model: self.model.clone(),
            lanes: self.lanes.clone(),
            runtime: tokio::runtime::Handle::current(),
            cancellation: context.cancellation_token(),
            output,
        };
        task::spawn_blocking(move || live.run(config, input));
        Ok(session)
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

/// One live transcription session, run on a blocking thread: it owns the
/// stream state and takes a model lane only while a pass runs.
struct LiveSession {
    model: Arc<dyn SpeechTranscriber>,
    lanes: Arc<Semaphore>,
    runtime: tokio::runtime::Handle,
    cancellation: CancellationToken,
    output: mpsc::Sender<RealtimeOutput>,
}

impl LiveSession {
    fn run(self, config: RealtimeTranscriptionConfig, mut input: mpsc::Receiver<RealtimeInput>) {
        let mut stream = SpeechStream::new(stream_config(&config));
        let mut resampler = PcmResampler::new(config.input_sample_rate);
        let mut next = input.blocking_recv();
        while let Some(command) = next.take() {
            let events = match command {
                RealtimeInput::Audio(samples) => {
                    // Batch audio that queued up behind the last pass.
                    let mut pcm = resampler.push(&samples);
                    loop {
                        match input.try_recv() {
                            Ok(RealtimeInput::Audio(more)) => pcm.extend(resampler.push(&more)),
                            Ok(other) => {
                                next = Some(other);
                                break;
                            }
                            Err(_) => break,
                        }
                    }
                    stream.push(&pcm, &mut |samples, options| {
                        self.transcribe(samples, options)
                    })
                }
                RealtimeInput::Commit => {
                    if !stream.has_pending_audio() {
                        self.send(RealtimeOutput::CommitEmpty);
                        next = next.or_else(|| input.blocking_recv());
                        continue;
                    }
                    self.send(RealtimeOutput::Committed {
                        turn: stream.current_turn(),
                    });
                    stream.commit(&mut |samples, options| self.transcribe(samples, options))
                }
                RealtimeInput::Clear => {
                    stream.clear();
                    Ok(Vec::new())
                }
                RealtimeInput::Configure(config) => {
                    stream.set_config(stream_config(&config));
                    if resampler.input_rate() != config.input_sample_rate {
                        resampler = PcmResampler::new(config.input_sample_rate);
                    }
                    Ok(Vec::new())
                }
            };
            match events {
                Ok(events) => events.into_iter().for_each(|event| self.report(event)),
                Err(error) => self.send(RealtimeOutput::Failed {
                    message: format!("{error:#}"),
                }),
            }
            if self.cancellation.is_cancelled() {
                break;
            }
            next = next.or_else(|| input.blocking_recv());
        }
    }

    fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
    ) -> anyhow::Result<Transcription> {
        let _lane = self
            .runtime
            .block_on(self.lanes.clone().acquire_owned())
            .map_err(|_| anyhow::anyhow!("speech backend is shutting down"))?;
        self.model
            .transcribe(samples, options, &|| self.cancellation.is_cancelled())
    }

    fn report(&self, event: SpeechStreamEvent) {
        match event {
            SpeechStreamEvent::SpeechStarted {
                turn,
                audio_start_ms,
            } => self.send(RealtimeOutput::SpeechStarted {
                turn,
                audio_start_ms,
            }),
            SpeechStreamEvent::SpeechStopped { turn, audio_end_ms } => {
                self.send(RealtimeOutput::SpeechStopped { turn, audio_end_ms });
                self.send(RealtimeOutput::Committed { turn });
            }
            SpeechStreamEvent::Delta { turn, text } => {
                self.send(RealtimeOutput::Delta { turn, text })
            }
            SpeechStreamEvent::Completed { turn, transcript } => {
                self.send(RealtimeOutput::Completed { turn, transcript })
            }
        }
    }

    fn send(&self, output: RealtimeOutput) {
        // A closed receiver means the client left; the input side ends the loop.
        let _ = self.output.blocking_send(output);
    }
}

fn stream_config(config: &RealtimeTranscriptionConfig) -> SpeechStreamConfig {
    SpeechStreamConfig {
        turn_detection: config.turn_detection.map(|detection| TurnDetection {
            threshold: detection.threshold,
            prefix_padding_ms: detection.prefix_padding_ms,
            silence_duration_ms: detection.silence_duration_ms,
        }),
        language: config.language.clone(),
        prompt: config.prompt.clone(),
        ..Default::default()
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
            cancelled: &(dyn Fn() -> bool + Sync),
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

    /// Holds each transcription open for a while and records how many ran
    /// at once.
    #[derive(Default)]
    struct SlowModel {
        active: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
        finished: std::sync::atomic::AtomicUsize,
    }

    impl SpeechTranscriber for SlowModel {
        fn decode(&self, _audio: &[u8]) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.0; 16_000])
        }

        fn transcribe(
            &self,
            _samples: &[f32],
            _options: &TranscriptionOptions,
            _cancelled: &(dyn Fn() -> bool + Sync),
        ) -> anyhow::Result<Transcription> {
            use std::sync::atomic::Ordering::SeqCst;
            let now = self.active.fetch_add(1, SeqCst) + 1;
            self.peak.fetch_max(now, SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(300));
            self.active.fetch_sub(1, SeqCst);
            self.finished.fetch_add(1, SeqCst);
            Ok(Transcription {
                segments: Vec::new(),
                language: "en".into(),
                duration_seconds: 1.0,
            })
        }
    }

    fn slow_request() -> AudioTranscriptionRequest {
        AudioTranscriptionRequest {
            model: "whisper-test".into(),
            file: b"RIFFdata".to_vec(),
            filename: None,
            language: None,
            prompt: None,
            response_format: "json".into(),
            temperature: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_abandoned_request_keeps_its_lane_until_its_work_ends() {
        use std::sync::atomic::Ordering::SeqCst;
        let model = Arc::new(SlowModel::default());
        let backend = Arc::new(SpeechRecognitionBackend::new(
            "whisper-test",
            model.clone(),
            1,
        ));
        let first = {
            let backend = backend.clone();
            tokio::spawn(async move {
                backend
                    .audio_transcription(slow_request(), InferenceRequestContext::new())
                    .await
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        // The client goes away while the first transcription is running.
        first.abort();
        backend
            .audio_transcription(slow_request(), InferenceRequestContext::new())
            .await
            .unwrap();
        assert_eq!(
            model.peak.load(SeqCst),
            1,
            "one lane admits one transcription"
        );
        assert_eq!(model.finished.load(SeqCst), 2);
    }

    #[tokio::test]
    async fn text_responses_come_back_as_plain_text() {
        let (status, body) = post(
            Arc::new(FakeModel::default()),
            "/v1/audio/transcriptions",
            upload("whisper-test", b"RIFFdata", &[("response_format", "text")]),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, "Hello there.");
    }

    async fn next_output(session: &mut RealtimeTranscriptionSession) -> RealtimeOutput {
        tokio::time::timeout(std::time::Duration::from_secs(5), session.output.recv())
            .await
            .expect("live session answers in time")
            .expect("live session stays open")
    }

    #[tokio::test]
    async fn live_sessions_confirm_words_and_complete_on_commit() {
        let backend =
            SpeechRecognitionBackend::new("whisper-test", Arc::new(FakeModel::default()), 1);
        let config = RealtimeTranscriptionConfig {
            turn_detection: None,
            input_sample_rate: 16_000,
            ..RealtimeTranscriptionConfig::new("whisper-test")
        };
        let mut session = backend
            .realtime_transcription(config, InferenceRequestContext::new())
            .await
            .unwrap();
        let second = vec![8_000i16; 16_000];
        // Audio that queues behind a pass is batched into the next one, so
        // how the words split across deltas depends on timing; their sum
        // does not.
        for _ in 0..2 {
            session
                .input
                .send(RealtimeInput::Audio(second.clone()))
                .await
                .unwrap();
        }
        session.input.send(RealtimeInput::Commit).await.unwrap();
        let mut streamed = String::new();
        let mut committed = false;
        let transcript = loop {
            match next_output(&mut session).await {
                RealtimeOutput::Delta { turn: 0, text } => streamed.push_str(&text),
                RealtimeOutput::Committed { turn: 0 } => committed = true,
                RealtimeOutput::Completed {
                    turn: 0,
                    transcript,
                } => break transcript,
                other => panic!("unexpected {other:?}"),
            }
        };
        assert!(committed);
        assert_eq!(streamed, "Hello there.");
        assert_eq!(transcript, "Hello there.");
        session.input.send(RealtimeInput::Commit).await.unwrap();
        assert_eq!(next_output(&mut session).await, RealtimeOutput::CommitEmpty);
    }

    #[tokio::test]
    async fn live_sessions_for_other_models_are_refused() {
        let backend =
            SpeechRecognitionBackend::new("whisper-test", Arc::new(FakeModel::default()), 1);
        let refused = backend
            .realtime_transcription(
                RealtimeTranscriptionConfig::new("other"),
                InferenceRequestContext::new(),
            )
            .await;
        assert!(refused.is_err());
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
