//! Live transcription over `GET /v1/realtime`, in OpenAI's realtime
//! transcription protocol.
//!
//! A backend exposes a live session as a pair of channels
//! ([`RealtimeTranscriptionSession`]); this module owns the websocket and the
//! JSON events. It accepts both the beta (`transcription_session.update`,
//! `input_audio_format`) and GA (`session.update` with `audio.input`) shapes
//! of the session configuration, and reports transcripts as
//! `conversation.item.input_audio_transcription.delta` / `.completed`.

use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::response::Response;
use base64::Engine as _;
use futures_util::stream::SplitStream;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::backend::{InferenceRequestContext, SharedBackend};
use crate::router::FrontendState;

/// OpenAI's `pcm16` input: 16-bit little-endian mono at 24 kHz.
pub const DEFAULT_INPUT_SAMPLE_RATE: u32 = 24_000;
/// Client commands a backend session buffers while a pass runs.
const INPUT_QUEUE: usize = 256;
/// Server events buffered for the socket writer.
const OUTPUT_QUEUE: usize = 256;
/// Largest websocket message or frame accepted: about 16 s of 24 kHz audio as
/// base64, well above what a streaming client sends at once.
const MAX_MESSAGE_BYTES: usize = 1 << 20;
/// Bounds on client-supplied turn detection and audio rates.
const MAX_PREFIX_PADDING_MS: f64 = 5_000.0;
const SILENCE_DURATION_MS: std::ops::RangeInclusive<f64> = 100.0..=30_000.0;
const INPUT_SAMPLE_RATES: std::ops::RangeInclusive<u64> = 8_000..=96_000;
/// How long a closing socket waits for the client to finish the handshake.
const CLOSE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

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
    /// New settings; they apply immediately, mid-turn included. A new
    /// input rate restarts resampling, which drops about a millisecond of
    /// filter history.
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
        let (output_tx, output) = mpsc::channel(OUTPUT_QUEUE);
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
    upgrade
        .max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| async move {
            Connection::start(state.backend, query.model, socket)
                .await
                .run()
                .await;
        })
}

/// Numbers the server's events and remembers the last committed item, for
/// both the request loop and the output pump.
struct Events {
    session_id: String,
    /// Short form of the session id that names items.
    item_prefix: String,
    state: Mutex<EventState>,
}

#[derive(Default)]
struct EventState {
    next_event: u64,
    last_item: Option<String>,
}

impl Events {
    fn new() -> Self {
        let id = uuid::Uuid::new_v4().simple().to_string();
        Self {
            session_id: format!("sess_{id}"),
            item_prefix: id[..12].to_string(),
            state: Mutex::default(),
        }
    }

    fn item_id(&self, turn: u64) -> String {
        format!("item_{}_{turn}", self.item_prefix)
    }

    fn event(&self, kind: &str, body: Value) -> Value {
        let id = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            state.next_event += 1;
            state.next_event
        };
        let mut event = json!({"type": kind, "event_id": format!("event_{id}")});
        if let (Value::Object(event), Value::Object(body)) = (&mut event, body) {
            event.extend(body);
        }
        event
    }

    fn error(&self, kind: &str, message: &str) -> Value {
        self.event(
            "error",
            json!({"error": {"type": kind, "message": message}}),
        )
    }

    fn output(&self, output: RealtimeOutput) -> Value {
        match output {
            RealtimeOutput::SpeechStarted {
                turn,
                audio_start_ms,
            } => self.event(
                "input_audio_buffer.speech_started",
                json!({"audio_start_ms": audio_start_ms, "item_id": self.item_id(turn)}),
            ),
            RealtimeOutput::SpeechStopped { turn, audio_end_ms } => self.event(
                "input_audio_buffer.speech_stopped",
                json!({"audio_end_ms": audio_end_ms, "item_id": self.item_id(turn)}),
            ),
            RealtimeOutput::Committed { turn } => {
                let item_id = self.item_id(turn);
                let previous = self
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .last_item
                    .replace(item_id.clone());
                self.event(
                    "input_audio_buffer.committed",
                    json!({"item_id": item_id, "previous_item_id": previous}),
                )
            }
            RealtimeOutput::Delta { turn, text } => self.event(
                "conversation.item.input_audio_transcription.delta",
                json!({"item_id": self.item_id(turn), "content_index": 0, "delta": text}),
            ),
            RealtimeOutput::Completed { turn, transcript } => self.event(
                "conversation.item.input_audio_transcription.completed",
                json!({"item_id": self.item_id(turn), "content_index": 0, "transcript": transcript}),
            ),
            RealtimeOutput::CommitEmpty => self.error(
                "invalid_request_error",
                "input_audio_buffer_commit_empty: the buffer holds no audio",
            ),
            RealtimeOutput::Failed { message } => self.error("server_error", &message),
        }
    }
}

/// One websocket. Reading client events, forwarding the backend's output,
/// and writing to the socket run independently, so a slow backend can never
/// stop its own output from draining.
struct Connection {
    backend: SharedBackend,
    incoming: SplitStream<WebSocket>,
    writer: mpsc::Sender<Message>,
    events: Arc<Events>,
    config: Option<RealtimeTranscriptionConfig>,
    /// The client sent the GA `session.update` shape; replies match it.
    ga: bool,
    /// Input to the backend session once audio has arrived.
    session: Option<mpsc::Sender<RealtimeInput>>,
    context: InferenceRequestContext,
    carry: PcmCarry,
}

/// Turns audio frames into 16-bit little-endian samples, carrying a trailing
/// odd byte into the next frame so a sample split across frames survives.
#[derive(Debug, Default)]
struct PcmCarry {
    pending: Option<u8>,
}

impl PcmCarry {
    fn samples(&mut self, bytes: &[u8]) -> Vec<i16> {
        let mut joined;
        let bytes = match self.pending.take() {
            Some(first) => {
                joined = Vec::with_capacity(bytes.len() + 1);
                joined.push(first);
                joined.extend_from_slice(bytes);
                joined.as_slice()
            }
            None => bytes,
        };
        let (pairs, rest) = bytes.as_chunks::<2>();
        self.pending = rest.first().copied();
        pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect()
    }

    fn clear(&mut self) {
        self.pending = None;
    }
}

enum Flow {
    Continue,
    Close,
}

impl Connection {
    /// Splits the socket, starts its writer, and picks the default model.
    async fn start(backend: SharedBackend, model: Option<String>, socket: WebSocket) -> Self {
        let (mut sink, incoming) = socket.split();
        let (writer, mut outgoing) = mpsc::channel::<Message>(OUTPUT_QUEUE);
        tokio::spawn(async move {
            while let Some(message) = outgoing.recv().await {
                let close = matches!(message, Message::Close(_));
                if sink.send(message).await.is_err() || close {
                    break;
                }
            }
        });
        let model = match model {
            Some(model) => Some(model),
            None => backend
                .models()
                .await
                .ok()
                .and_then(|models| models.into_iter().next())
                .map(|model| model.id),
        };
        Self {
            backend,
            incoming,
            writer,
            events: Arc::new(Events::new()),
            config: model.map(RealtimeTranscriptionConfig::new),
            ga: false,
            session: None,
            context: InferenceRequestContext::new(),
            carry: PcmCarry::default(),
        }
    }

    async fn run(mut self) {
        // Refuse a model that cannot stream before the client sends audio.
        if let Some(config) = self.config.clone()
            && let Err(message) = self.probe(&config).await
        {
            let error = self.events.error("invalid_request_error", &message);
            let _ = self.send(error).await;
            let _ = self.writer.send(Message::Close(None)).await;
            return;
        }
        let created = self.session_event("transcription_session.created");
        if self.send(created).await.is_err() {
            return;
        }
        while let Some(Ok(message)) = self.incoming.next().await {
            if matches!(self.client_message(message).await, Flow::Close) {
                break;
            }
        }
        // Ends the backend session: its input closes and its work is cancelled.
        self.context.cancel();
        // The reply to a client's close frame goes out on the next read, so
        // read on until the socket ends; without that the client sees a
        // reset. Bounded, for a client that never answers our own close.
        let _ = self.writer.send(Message::Close(None)).await;
        let _ = tokio::time::timeout(CLOSE_GRACE, async {
            while let Some(Ok(_)) = self.incoming.next().await {}
        })
        .await;
    }

    /// Opens and immediately drops a session, to learn whether the backend
    /// streams this model. Backends start no work until input arrives.
    async fn probe(&self, config: &RealtimeTranscriptionConfig) -> Result<(), String> {
        self.backend
            .realtime_transcription(config.clone(), InferenceRequestContext::new())
            .await
            .map(drop)
            .map_err(|error| error.body().error.message.clone())
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
            Message::Binary(bytes) => {
                let samples = self.carry.samples(&bytes);
                return self.append(samples).await;
            }
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
                    Ok(bytes) => {
                        let samples = self.carry.samples(&bytes);
                        self.append(samples).await
                    }
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
                self.carry.clear();
                // Before any audio there is nothing to clear.
                if self.session.is_some()
                    && matches!(self.forward(RealtimeInput::Clear).await, Flow::Close)
                {
                    return Flow::Close;
                }
                let cleared = self.events.event("input_audio_buffer.cleared", json!({}));
                self.reply(cleared).await
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
        let model_changed = self
            .config
            .as_ref()
            .is_none_or(|current| current.model != updated.model);
        if model_changed {
            if self.session.is_some() {
                return self
                    .reply_error(
                        "invalid_request_error",
                        "the model cannot change once audio has arrived",
                    )
                    .await;
            }
            if let Err(message) = self.probe(&updated).await {
                return self.reply_error("invalid_request_error", &message).await;
            }
        }
        self.config = Some(updated.clone());
        if self.session.is_some()
            && matches!(
                self.forward(RealtimeInput::Configure(updated)).await,
                Flow::Close
            )
        {
            return Flow::Close;
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

    /// Sends input to the backend session, opening it on first use. Waiting
    /// here for input capacity never blocks output, which a separate task
    /// drains.
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
                Ok(session) => {
                    self.session = Some(session.input);
                    tokio::spawn(pump_output(
                        session.output,
                        self.events.clone(),
                        self.writer.clone(),
                    ));
                }
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
        if session.send(input).await.is_err() {
            return self
                .reply_error("server_error", "the transcription session ended")
                .await;
        }
        Flow::Continue
    }

    fn session_event(&self, kind: &str) -> Value {
        let session = self
            .config
            .as_ref()
            .map(|config| session_json(&self.events.session_id, config, self.ga))
            .unwrap_or(json!({"id": self.events.session_id}));
        self.events.event(kind, json!({"session": session}))
    }

    async fn reply_error(&mut self, kind: &str, message: &str) -> Flow {
        let error = self.events.error(kind, message);
        self.reply(error).await
    }

    async fn reply(&mut self, event: Value) -> Flow {
        match self.send(event).await {
            Ok(()) => Flow::Continue,
            Err(()) => Flow::Close,
        }
    }

    async fn send(&self, event: Value) -> Result<(), ()> {
        self.writer
            .send(Message::Text(event.to_string().into()))
            .await
            .map_err(|_| ())
    }
}

/// Forwards a backend session's output to the socket writer. When the
/// session ends on its own, the client is told and the socket closes.
async fn pump_output(
    mut output: mpsc::Receiver<RealtimeOutput>,
    events: Arc<Events>,
    writer: mpsc::Sender<Message>,
) {
    while let Some(event) = output.recv().await {
        let message = Message::Text(events.output(event).to_string().into());
        if writer.send(message).await.is_err() {
            return;
        }
    }
    let ended = events.error("server_error", "the transcription session ended");
    let _ = writer.send(Message::Text(ended.to_string().into())).await;
    let _ = writer.send(Message::Close(None)).await;
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
                let number =
                    |key: &str, range: std::ops::RangeInclusive<f64>| match detection.get(key) {
                        None | Some(Value::Null) => Ok(None),
                        Some(value) => value
                            .as_f64()
                            .filter(|value| range.contains(value))
                            .map(Some)
                            .ok_or_else(|| {
                                format!(
                                    "turn_detection.{key} must be a number from {} to {}",
                                    range.start(),
                                    range.end()
                                )
                            }),
                    };
                Some(RealtimeTurnDetection {
                    threshold: number("threshold", 0.0..=1.0)?
                        .map_or(defaults.threshold, |value| value as f32),
                    prefix_padding_ms: number("prefix_padding_ms", 0.0..=MAX_PREFIX_PADDING_MS)?
                        .map_or(defaults.prefix_padding_ms, |value| value as u32),
                    silence_duration_ms: number("silence_duration_ms", SILENCE_DURATION_MS)?
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
        Value::Object(_) if format["type"] == "audio/pcm" => match format.get("rate") {
            None | Some(Value::Null) => Ok(DEFAULT_INPUT_SAMPLE_RATE),
            Some(rate) => rate
                .as_u64()
                .filter(|rate| INPUT_SAMPLE_RATES.contains(rate))
                .map(|rate| rate as u32)
                .ok_or_else(|| {
                    format!(
                        "audio input rate must be an integer from {} to {} Hz",
                        INPUT_SAMPLE_RATES.start(),
                        INPUT_SAMPLE_RATES.end()
                    )
                }),
        },
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
