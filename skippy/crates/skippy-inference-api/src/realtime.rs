//! Live transcription over `GET /v1/realtime`, in OpenAI's realtime
//! transcription protocol.
//!
//! A backend exposes a live session as a pair of channels
//! ([`RealtimeTranscriptionSession`]); this module owns the websocket and the
//! JSON events. It accepts both the beta (`transcription_session.update`,
//! `input_audio_format`) and GA (`session.update` with `audio.input`) shapes
//! of the session configuration, and reports transcripts as
//! `conversation.item.input_audio_transcription.delta` / `.completed`.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::response::Response;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::backend::{InferenceRequestContext, SharedBackend};
use crate::router::FrontendState;

/// OpenAI's `pcm16` input: 16-bit little-endian mono at 24 kHz.
pub const DEFAULT_INPUT_SAMPLE_RATE: u32 = 24_000;
/// Commands a session buffers while a pass runs.
const INPUT_QUEUE: usize = 256;

/// Server-side turn detection, as configured by the client.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealtimeTurnDetection {
    pub threshold: f32,
    pub prefix_padding_ms: u32,
    pub silence_duration_ms: u32,
}

impl Default for RealtimeTurnDetection {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
        }
    }
}

/// How a live transcription session transcribes.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeTranscriptionConfig {
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    /// `None` leaves turns to `input_audio_buffer.commit`.
    pub turn_detection: Option<RealtimeTurnDetection>,
    /// Rate of the 16-bit PCM the client appends.
    pub input_sample_rate: u32,
}

impl RealtimeTranscriptionConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            language: None,
            prompt: None,
            turn_detection: Some(RealtimeTurnDetection::default()),
            input_sample_rate: DEFAULT_INPUT_SAMPLE_RATE,
        }
    }
}

/// What the client sends a live session.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeInput {
    /// 16-bit mono PCM at the configured input rate.
    Audio(Vec<i16>),
    /// End the current turn now.
    Commit,
    /// Drop the current turn's audio.
    Clear,
    /// New settings; they apply from the next turn.
    Configure(RealtimeTranscriptionConfig),
}

/// What a live session reports; `turn` numbers utterances from 0.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeOutput {
    SpeechStarted {
        turn: u64,
        audio_start_ms: u64,
    },
    SpeechStopped {
        turn: u64,
        audio_end_ms: u64,
    },
    /// The turn's audio is complete, after a stop or a client commit.
    Committed {
        turn: u64,
    },
    Delta {
        turn: u64,
        text: String,
    },
    Completed {
        turn: u64,
        transcript: String,
    },
    /// A commit arrived with no audio to transcribe.
    CommitEmpty,
    Failed {
        message: String,
    },
}

/// A running live session: send it input, read its output. Dropping the
/// input sender ends it.
pub struct RealtimeTranscriptionSession {
    pub input: mpsc::Sender<RealtimeInput>,
    pub output: mpsc::Receiver<RealtimeOutput>,
}

impl RealtimeTranscriptionSession {
    /// Channels sized for a session; the backend keeps the other ends.
    pub fn channels() -> (
        Self,
        mpsc::Receiver<RealtimeInput>,
        mpsc::Sender<RealtimeOutput>,
    ) {
        let (input, input_rx) = mpsc::channel(INPUT_QUEUE);
        let (output_tx, output) = mpsc::channel(INPUT_QUEUE);
        (Self { input, output }, input_rx, output_tx)
    }
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct RealtimeQuery {
    model: Option<String>,
}

pub(crate) async fn realtime(
    State(state): State<FrontendState>,
    Query(query): Query<RealtimeQuery>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade.on_upgrade(move |socket| async move {
        Connection::new(state.backend, query.model, socket)
            .run()
            .await;
    })
}

/// One websocket's protocol state.
struct Connection {
    backend: SharedBackend,
    socket: WebSocket,
    session_id: String,
    config: Option<RealtimeTranscriptionConfig>,
    /// The client sent the GA `session.update` shape; replies match it.
    ga: bool,
    session: Option<RealtimeTranscriptionSession>,
    context: InferenceRequestContext,
    next_event: u64,
    last_item: Option<String>,
}

enum Flow {
    Continue,
    Close,
}

impl Connection {
    fn new(backend: SharedBackend, model: Option<String>, socket: WebSocket) -> Self {
        Self {
            backend,
            socket,
            session_id: format!("sess_{}", uuid::Uuid::new_v4().simple()),
            config: model.map(RealtimeTranscriptionConfig::new),
            ga: false,
            session: None,
            context: InferenceRequestContext::new(),
            next_event: 0,
            last_item: None,
        }
    }

    async fn run(mut self) {
        if self.config.is_none() {
            let model = self
                .backend
                .models()
                .await
                .ok()
                .and_then(|models| models.into_iter().next())
                .map(|model| model.id);
            self.config = model.map(RealtimeTranscriptionConfig::new);
        }
        let created = self.session_event("transcription_session.created");
        if self.send(created).await.is_err() {
            return;
        }
        loop {
            let flow = tokio::select! {
                message = self.socket.recv() => match message {
                    Some(Ok(message)) => self.client_message(message).await,
                    _ => Flow::Close,
                },
                output = next_output(&mut self.session) => match output {
                    Some(output) => self.server_output(output).await,
                    None => {
                        self.session = None;
                        let error = self.error("server_error", "the transcription session ended");
                        let _ = self.send(error).await;
                        Flow::Close
                    }
                },
            };
            if matches!(flow, Flow::Close) {
                break;
            }
        }
        self.context.cancel();
    }

    async fn client_message(&mut self, message: Message) -> Flow {
        let event = match message {
            Message::Text(text) => match serde_json::from_str::<Value>(&text) {
                Ok(event) => event,
                Err(error) => {
                    return self
                        .reply_error("invalid_request_error", &format!("invalid JSON: {error}"))
                        .await;
                }
            },
            // Raw PCM frames: a lighter alternative to base64 appends.
            Message::Binary(bytes) => return self.append(pcm16(&bytes)).await,
            Message::Close(_) => return Flow::Close,
            Message::Ping(_) | Message::Pong(_) => return Flow::Continue,
        };
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "transcription_session.update" | "session.update" => {
                self.ga = event["type"] == "session.update";
                self.update(&event["session"]).await
            }
            "input_audio_buffer.append" => {
                let Some(audio) = event.get("audio").and_then(Value::as_str) else {
                    return self
                        .reply_error("invalid_request_error", "append needs base64 `audio`")
                        .await;
                };
                match base64::engine::general_purpose::STANDARD.decode(audio) {
                    Ok(bytes) => self.append(pcm16(&bytes)).await,
                    Err(error) => {
                        self.reply_error(
                            "invalid_request_error",
                            &format!("audio is not base64: {error}"),
                        )
                        .await
                    }
                }
            }
            "input_audio_buffer.commit" => self.forward(RealtimeInput::Commit).await,
            "input_audio_buffer.clear" => {
                let flow = self.forward(RealtimeInput::Clear).await;
                let cleared = self.event("input_audio_buffer.cleared", json!({}));
                let _ = self.send(cleared).await;
                flow
            }
            other => {
                self.reply_error(
                    "invalid_request_error",
                    &format!("unsupported client event type {other:?}"),
                )
                .await
            }
        }
    }

    async fn update(&mut self, session: &Value) -> Flow {
        let base = self
            .config
            .clone()
            .unwrap_or_else(|| RealtimeTranscriptionConfig::new(""));
        let updated = match apply_session_update(base, session) {
            Ok(updated) => updated,
            Err(message) => return self.reply_error("invalid_request_error", &message).await,
        };
        if let (Some(_), Some(current)) = (&self.session, &self.config)
            && current.model != updated.model
        {
            return self
                .reply_error(
                    "invalid_request_error",
                    "the model cannot change once audio has arrived",
                )
                .await;
        }
        self.config = Some(updated.clone());
        if self.session.is_some() {
            let flow = self.forward(RealtimeInput::Configure(updated)).await;
            if matches!(flow, Flow::Close) {
                return flow;
            }
        }
        let kind = if self.ga {
            "session.updated"
        } else {
            "transcription_session.updated"
        };
        let event = self.session_event(kind);
        self.reply(event).await
    }

    async fn append(&mut self, samples: Vec<i16>) -> Flow {
        if samples.is_empty() {
            return Flow::Continue;
        }
        self.forward(RealtimeInput::Audio(samples)).await
    }

    /// Sends input to the backend session, opening it on first use.
    async fn forward(&mut self, input: RealtimeInput) -> Flow {
        if self.session.is_none() {
            let Some(config) = self
                .config
                .clone()
                .filter(|config| !config.model.is_empty())
            else {
                return self
                    .reply_error(
                        "invalid_request_error",
                        "no model: pass ?model= or set input_audio_transcription.model",
                    )
                    .await;
            };
            match self
                .backend
                .realtime_transcription(config, self.context.clone())
                .await
            {
                Ok(session) => self.session = Some(session),
                Err(error) => {
                    let message = error.body().error.message.clone();
                    let _ = self.reply_error("invalid_request_error", &message).await;
                    return Flow::Close;
                }
            }
        }
        let Some(session) = &self.session else {
            return Flow::Close;
        };
        if session.input.send(input).await.is_err() {
            return self
                .reply_error("server_error", "the transcription session ended")
                .await;
        }
        Flow::Continue
    }

    async fn server_output(&mut self, output: RealtimeOutput) -> Flow {
        let events = match output {
            RealtimeOutput::SpeechStarted {
                turn,
                audio_start_ms,
            } => vec![self.event(
                "input_audio_buffer.speech_started",
                json!({"audio_start_ms": audio_start_ms, "item_id": self.item_id(turn)}),
            )],
            RealtimeOutput::SpeechStopped { turn, audio_end_ms } => vec![self.event(
                "input_audio_buffer.speech_stopped",
                json!({"audio_end_ms": audio_end_ms, "item_id": self.item_id(turn)}),
            )],
            RealtimeOutput::Committed { turn } => {
                let item_id = self.item_id(turn);
                let previous = self.last_item.replace(item_id.clone());
                vec![self.event(
                    "input_audio_buffer.committed",
                    json!({"item_id": item_id, "previous_item_id": previous}),
                )]
            }
            RealtimeOutput::Delta { turn, text } => vec![self.event(
                "conversation.item.input_audio_transcription.delta",
                json!({"item_id": self.item_id(turn), "content_index": 0, "delta": text}),
            )],
            RealtimeOutput::Completed { turn, transcript } => vec![self.event(
                "conversation.item.input_audio_transcription.completed",
                json!({"item_id": self.item_id(turn), "content_index": 0, "transcript": transcript}),
            )],
            RealtimeOutput::CommitEmpty => vec![self.error(
                "invalid_request_error",
                "input_audio_buffer_commit_empty: the buffer holds no audio",
            )],
            RealtimeOutput::Failed { message } => vec![self.error("server_error", &message)],
        };
        for event in events {
            if self.send(event).await.is_err() {
                return Flow::Close;
            }
        }
        Flow::Continue
    }

    fn item_id(&self, turn: u64) -> String {
        format!("item_{}_{turn}", &self.session_id[5..17])
    }

    fn session_event(&mut self, kind: &str) -> Value {
        let session = self
            .config
            .as_ref()
            .map(|config| session_json(&self.session_id, config, self.ga))
            .unwrap_or(json!({"id": self.session_id}));
        self.event(kind, json!({"session": session}))
    }

    fn event(&mut self, kind: &str, body: Value) -> Value {
        self.next_event += 1;
        let mut event = json!({"type": kind, "event_id": format!("event_{}", self.next_event)});
        if let (Value::Object(event), Value::Object(body)) = (&mut event, body) {
            event.extend(body);
        }
        event
    }

    fn error(&mut self, kind: &str, message: &str) -> Value {
        self.event(
            "error",
            json!({"error": {"type": kind, "message": message}}),
        )
    }

    async fn reply_error(&mut self, kind: &str, message: &str) -> Flow {
        let error = self.error(kind, message);
        self.reply(error).await
    }

    async fn reply(&mut self, event: Value) -> Flow {
        match self.send(event).await {
            Ok(()) => Flow::Continue,
            Err(()) => Flow::Close,
        }
    }

    async fn send(&mut self, event: Value) -> Result<(), ()> {
        self.socket
            .send(Message::Text(event.to_string().into()))
            .await
            .map_err(|_| ())
    }
}

async fn next_output(session: &mut Option<RealtimeTranscriptionSession>) -> Option<RealtimeOutput> {
    match session {
        Some(session) => session.output.recv().await,
        None => std::future::pending().await,
    }
}

/// Little-endian 16-bit samples; a trailing odd byte is dropped.
fn pcm16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect()
}

/// Applies a beta or GA session object to the current configuration. Fields
/// the client leaves out keep their values; `turn_detection: null` turns
/// server turn detection off.
pub(crate) fn apply_session_update(
    mut config: RealtimeTranscriptionConfig,
    session: &Value,
) -> Result<RealtimeTranscriptionConfig, String> {
    // GA nests everything under `audio.input`.
    let input = session.pointer("/audio/input").unwrap_or(session);
    let transcription = input
        .get("transcription")
        .or_else(|| session.get("input_audio_transcription"));
    if let Some(transcription) = transcription.filter(|value| !value.is_null()) {
        if let Some(model) = transcription.get("model").and_then(Value::as_str) {
            config.model = model.to_string();
        }
        if let Some(language) = transcription.get("language") {
            config.language = language
                .as_str()
                .filter(|language| !language.is_empty())
                .map(str::to_string);
        }
        if let Some(prompt) = transcription.get("prompt") {
            config.prompt = prompt
                .as_str()
                .filter(|prompt| !prompt.is_empty())
                .map(str::to_string);
        }
    }
    if let Some(format) = input
        .get("format")
        .or_else(|| session.get("input_audio_format"))
    {
        config.input_sample_rate = input_rate(format)?;
    }
    let turn_detection = input
        .get("turn_detection")
        .or_else(|| session.get("turn_detection"));
    if let Some(turn_detection) = turn_detection {
        config.turn_detection = match turn_detection {
            Value::Null => None,
            detection => {
                let kind = detection
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("server_vad");
                if !matches!(kind, "server_vad" | "semantic_vad") {
                    return Err(format!("unsupported turn_detection type {kind:?}"));
                }
                let defaults = config.turn_detection.unwrap_or_default();
                let number = |key: &str| detection.get(key).and_then(Value::as_f64);
                Some(RealtimeTurnDetection {
                    threshold: number("threshold").map_or(defaults.threshold, |value| value as f32),
                    prefix_padding_ms: number("prefix_padding_ms")
                        .map_or(defaults.prefix_padding_ms, |value| value as u32),
                    silence_duration_ms: number("silence_duration_ms")
                        .map_or(defaults.silence_duration_ms, |value| value as u32),
                })
            }
        };
    }
    Ok(config)
}

fn input_rate(format: &Value) -> Result<u32, String> {
    match format {
        Value::String(name) if name == "pcm16" => Ok(DEFAULT_INPUT_SAMPLE_RATE),
        Value::Object(_) if format["type"] == "audio/pcm" => Ok(format
            .get("rate")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_INPUT_SAMPLE_RATE, |rate| rate as u32)),
        other => Err(format!(
            "unsupported input audio format {other}; send pcm16 (16-bit little-endian mono)"
        )),
    }
}

fn session_json(id: &str, config: &RealtimeTranscriptionConfig, ga: bool) -> Value {
    let turn_detection = config.turn_detection.map(|detection| {
        json!({
            "type": "server_vad",
            "threshold": detection.threshold,
            "prefix_padding_ms": detection.prefix_padding_ms,
            "silence_duration_ms": detection.silence_duration_ms,
        })
    });
    let transcription = json!({
        "model": config.model,
        "language": config.language,
        "prompt": config.prompt,
    });
    if ga {
        json!({
            "id": id,
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": config.input_sample_rate},
                "transcription": transcription,
                "turn_detection": turn_detection,
            }},
        })
    } else {
        json!({
            "id": id,
            "object": "realtime.transcription_session",
            "input_audio_format": "pcm16",
            "input_audio_transcription": transcription,
            "turn_detection": turn_detection,
        })
    }
}

#[cfg(test)]
mod tests;
