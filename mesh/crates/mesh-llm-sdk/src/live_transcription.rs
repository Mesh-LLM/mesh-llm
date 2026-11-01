//! Live transcription through a node's `/v1/realtime` endpoint.
//!
//! The caller supplies audio (16-bit mono PCM at any sample rate, for example
//! from a microphone) and reads transcript events as the speaker talks. Any
//! node can host the session: the mesh relays it to a node serving a
//! speech-recognition model.
//!
//! ```no_run
//! # async fn example(node: &mesh_llm_sdk::MeshNode, frames: Vec<Vec<i16>>) -> anyhow::Result<()> {
//! use mesh_llm_sdk::live_transcription::{LiveTranscriptionEvent, LiveTranscriptionOptions};
//!
//! let mut live = node
//!     .start_live_transcription(LiveTranscriptionOptions {
//!         sample_rate: 48_000,
//!         ..Default::default()
//!     })
//!     .await?;
//! for frame in frames {
//!     live.send_audio(&frame).await?;
//! }
//! while let Some(event) = live.next_event().await {
//!     if let LiveTranscriptionEvent::Completed { transcript, .. } = event {
//!         let _ = transcript; // hand the finished utterance to the app
//!         break;
//!     }
//! }
//! let _remaining = live.stop().await?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::Duration;

pub mod session;

use anyhow::{Context, Result, anyhow, bail};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::Connector;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;

/// How long [`LiveTranscription::stop`] waits for the final transcript.
const STOP_TIMEOUT: Duration = Duration::from_secs(15);
/// How long starting a session waits for the node to accept it.
const START_TIMEOUT: Duration = Duration::from_secs(30);
const QUEUE: usize = 256;

/// How a live transcription session transcribes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LiveTranscriptionOptions {
    /// Model to use; `None` picks the first model on the mesh that supports
    /// live transcription.
    pub model: Option<String>,
    /// ISO 639-1 code such as `"en"`; `None` detects the language.
    pub language: Option<String>,
    /// Text that conditions transcription, such as names or jargon.
    pub prompt: Option<String>,
    /// Sample rate of the audio passed to [`LiveTranscription::send_audio`];
    /// the node accepts 8000 to 96000 Hz.
    pub sample_rate: u32,
    /// Server-side turn detection; `None` ends turns only on
    /// [`LiveTranscription::commit`] (push to talk).
    pub turn_detection: Option<LiveTurnDetection>,
}

impl Default for LiveTranscriptionOptions {
    fn default() -> Self {
        Self {
            model: None,
            language: None,
            prompt: None,
            sample_rate: 16_000,
            turn_detection: Some(LiveTurnDetection::default()),
        }
    }
}

/// When the server ends a turn on its own.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LiveTurnDetection {
    /// 0.0 to 1.0; higher needs louder audio to count as speech.
    pub threshold: f32,
    /// Audio kept from before speech starts.
    pub prefix_padding_ms: u32,
    /// Silence that ends a turn.
    pub silence_duration_ms: u32,
}

impl Default for LiveTurnDetection {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
        }
    }
}

/// What a session reports. `item_id` names one turn (one utterance).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveTranscriptionEvent {
    SpeechStarted {
        item_id: String,
        audio_start_ms: u64,
    },
    SpeechStopped {
        item_id: String,
        audio_end_ms: u64,
    },
    /// Newly confirmed text to append to the turn; it never changes.
    Delta {
        item_id: String,
        text: String,
    },
    /// The turn's audio is complete, ended by a pause or by a commit; its
    /// [`Completed`](Self::Completed) transcript follows.
    Committed {
        item_id: String,
    },
    /// The turn's whole transcript; no more deltas follow for it.
    Completed {
        item_id: String,
        transcript: String,
    },
    /// The session reported a problem. After a lost connection this is the
    /// last event; otherwise the session may continue.
    Error {
        message: String,
    },
}

enum Outbound {
    Audio(Vec<u8>),
    Event(String),
    Close,
}

/// A running live transcription session. Send audio, read events, and
/// [`stop`](Self::stop) when done. [`split`](Self::split) separates the two
/// halves for use from different tasks. Dropping the session, or the
/// [`LiveTranscriptionAudio`] half, closes it.
pub struct LiveTranscription {
    audio: LiveTranscriptionAudio,
    events: LiveTranscriptionEvents,
}

/// The sending half of a session. Dropping it closes the session.
pub struct LiveTranscriptionAudio {
    outbound: mpsc::Sender<Outbound>,
}

/// The receiving half of a session.
pub struct LiveTranscriptionEvents {
    inbound: mpsc::Receiver<LiveTranscriptionEvent>,
    /// The last turn a pause ended. The node reports such a turn as stopped
    /// and then committed, and a turn ended by a commit as committed alone.
    paused_turn: Option<String>,
}

impl LiveTranscription {
    /// Opens a session on `/v1/realtime` under `api_base_url`
    /// (for example `http://127.0.0.1:9337/v1`).
    pub async fn start(api_base_url: &str, options: LiveTranscriptionOptions) -> Result<Self> {
        Self::start_with_api_key(api_base_url, None, options).await
    }

    /// Like [`start`](Self::start), sending `api_key` as a bearer token.
    pub async fn start_with_api_key(
        api_base_url: &str,
        api_key: Option<&str>,
        options: LiveTranscriptionOptions,
    ) -> Result<Self> {
        if options.sample_rate == 0 {
            bail!("live transcription needs a sample rate above zero");
        }
        let url = realtime_url(api_base_url, options.model.as_deref())?;
        let mut request = url.as_str().into_client_request()?;
        if let Some(api_key) = api_key {
            let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("api key is not a valid header value")?;
            request.headers_mut().insert("authorization", value);
        }
        let connector = url.starts_with("wss://").then(tls_connector).transpose()?;
        let (socket, _) = tokio::time::timeout(
            START_TIMEOUT,
            tokio_tungstenite::connect_async_tls_with_config(request, None, false, connector),
        )
        .await
        .map_err(|_| anyhow!("timed out connecting to {url}"))?
        .with_context(|| format!("open live transcription session at {url}"))?;
        let (mut sink, mut stream) = socket.split();

        // The node announces the session, then confirms the configuration.
        let update = session_update(&options).to_string();
        sink.send(Message::Text(update.into()))
            .await
            .context("configure live transcription session")?;
        tokio::time::timeout(START_TIMEOUT, wait_until_configured(&mut stream))
            .await
            .map_err(|_| anyhow!("live transcription session was not confirmed"))??;

        let (outbound, mut outbound_rx) = mpsc::channel::<Outbound>(QUEUE);
        let (inbound_tx, inbound) = mpsc::channel(QUEUE);
        // Ends with a close frame, whether asked to or because every sender
        // was dropped.
        tokio::spawn(async move {
            while let Some(message) = outbound_rx.recv().await {
                let frame = match message {
                    Outbound::Audio(bytes) => Message::Binary(bytes.into()),
                    Outbound::Event(text) => Message::Text(text.into()),
                    Outbound::Close => break,
                };
                if sink.send(frame).await.is_err() {
                    return;
                }
            }
            let _ = sink.send(Message::Close(None)).await;
        });
        tokio::spawn(async move {
            while let Some(message) = stream.next().await {
                let event = match message {
                    Ok(Message::Text(text)) => serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|event| parse_event(&event)),
                    Ok(Message::Close(_)) => break,
                    Ok(_) => None,
                    Err(error) => {
                        let message = format!("live transcription connection lost: {error}");
                        let _ = inbound_tx
                            .send(LiveTranscriptionEvent::Error { message })
                            .await;
                        break;
                    }
                };
                if let Some(event) = event
                    && inbound_tx.send(event).await.is_err()
                {
                    break;
                }
            }
        });
        Ok(Self {
            audio: LiveTranscriptionAudio { outbound },
            events: LiveTranscriptionEvents {
                inbound,
                paused_turn: None,
            },
        })
    }

    /// Sends 16-bit mono PCM at the session's sample rate.
    pub async fn send_audio(&self, samples: &[i16]) -> Result<()> {
        self.audio.send_audio(samples).await
    }

    /// Sends float PCM at the session's sample rate. Samples outside
    /// -1.0..=1.0 are clipped.
    pub async fn send_audio_f32(&self, samples: &[f32]) -> Result<()> {
        self.audio.send_audio_f32(samples).await
    }

    /// Ends the current turn now; its final transcript follows as a
    /// [`LiveTranscriptionEvent::Completed`].
    pub async fn commit(&self) -> Result<()> {
        self.audio.commit().await
    }

    /// The next event, or `None` once the session has closed.
    pub async fn next_event(&mut self) -> Option<LiveTranscriptionEvent> {
        self.events.next_event().await
    }

    /// Finishes the current turn, waits for its transcript, and closes the
    /// session. Returns the events that arrived meanwhile, including earlier
    /// turns still finishing, and ending with the last turn's
    /// [`LiveTranscriptionEvent::Completed`] when there was audio left to
    /// transcribe. Fails if the transcript does not arrive in time.
    pub async fn stop(self) -> Result<Vec<LiveTranscriptionEvent>> {
        let Self { audio, mut events } = self;
        audio.commit().await?;
        let mut remaining = Vec::new();
        let finished = tokio::time::timeout(STOP_TIMEOUT, async {
            // The turn our commit ended: the first committed turn that a
            // pause did not end. Earlier turns may still complete first.
            let mut last_turn = None;
            while let Some(event) = events.next_event().await {
                match &event {
                    LiveTranscriptionEvent::Committed { item_id }
                        if last_turn.is_none() && events.paused_turn.as_ref() != Some(item_id) =>
                    {
                        last_turn = Some(item_id.clone());
                    }
                    // Nothing was left to transcribe.
                    LiveTranscriptionEvent::Error { message }
                        if last_turn.is_none() && is_commit_empty(message) =>
                    {
                        return;
                    }
                    LiveTranscriptionEvent::Completed { item_id, .. }
                        if last_turn.as_ref() == Some(item_id) =>
                    {
                        remaining.push(event);
                        return;
                    }
                    LiveTranscriptionEvent::Error { .. } => {
                        remaining.push(event);
                        return;
                    }
                    _ => {}
                }
                remaining.push(event);
            }
        })
        .await;
        audio.close().await;
        finished
            .map_err(|_| anyhow!("live transcription did not finish within {STOP_TIMEOUT:?}"))?;
        Ok(remaining)
    }

    /// Separates sending audio from reading events, for use from two tasks.
    pub fn split(self) -> (LiveTranscriptionAudio, LiveTranscriptionEvents) {
        (self.audio, self.events)
    }
}

impl LiveTranscriptionAudio {
    /// Sends 16-bit mono PCM at the session's sample rate.
    pub async fn send_audio(&self, samples: &[i16]) -> Result<()> {
        let bytes = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        self.send(Outbound::Audio(bytes)).await
    }

    /// Sends float PCM at the session's sample rate. Samples outside
    /// -1.0..=1.0 are clipped.
    pub async fn send_audio_f32(&self, samples: &[f32]) -> Result<()> {
        let pcm = samples
            .iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
            .collect::<Vec<_>>();
        self.send_audio(&pcm).await
    }

    /// Ends the current turn now.
    pub async fn commit(&self) -> Result<()> {
        let commit = json!({"type": "input_audio_buffer.commit"}).to_string();
        self.send(Outbound::Event(commit)).await
    }

    /// Closes the session without waiting for a final transcript. Dropping
    /// the sender does the same.
    pub async fn close(self) {
        let _ = self.outbound.send(Outbound::Close).await;
    }

    async fn send(&self, message: Outbound) -> Result<()> {
        self.outbound
            .send(message)
            .await
            .map_err(|_| anyhow!("live transcription session has closed"))
    }
}

impl LiveTranscriptionEvents {
    /// The next event, or `None` once the session has closed.
    pub async fn next_event(&mut self) -> Option<LiveTranscriptionEvent> {
        let event = self.inbound.recv().await;
        if let Some(LiveTranscriptionEvent::SpeechStopped { item_id, .. }) = &event {
            self.paused_turn = Some(item_id.clone());
        }
        event
    }
}

/// Reads the handshake: the session announcement, then the reply to our
/// configuration. An error event or a closed socket fails the start.
async fn wait_until_configured<S>(stream: &mut S) -> Result<()>
where
    S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while let Some(message) = stream.next().await {
        let Message::Text(text) = message.context("read live transcription session")? else {
            continue;
        };
        let event: Value = serde_json::from_str(&text).context("parse session event")?;
        match event["type"].as_str().unwrap_or_default() {
            "session.updated" | "transcription_session.updated" => return Ok(()),
            "error" => {
                return Err(anyhow!(
                    "live transcription refused: {}",
                    event["error"]["message"]
                        .as_str()
                        .unwrap_or("unknown error")
                ));
            }
            _ => {}
        }
    }
    Err(anyhow!("live transcription session closed during setup"))
}

/// The node's reply to a commit with no audio behind it.
fn is_commit_empty(message: &str) -> bool {
    message.starts_with("input_audio_buffer_commit_empty")
}

/// TLS for `wss://` nodes, trusting the platform's roots. The provider is the
/// process default when one is installed, ring otherwise; the SDK never
/// installs one itself.
fn tls_connector() -> Result<Connector> {
    let provider = rustls::crypto::CryptoProvider::get_default()
        .cloned()
        .unwrap_or_else(|| Arc::new(rustls::crypto::ring::default_provider()));
    let mut roots = rustls::RootCertStore::empty();
    let native = rustls_native_certs::load_native_certs();
    let (added, _) = roots.add_parsable_certificates(native.certs);
    if added == 0 {
        bail!("no trusted root certificates found for wss");
    }
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("configure TLS for wss")?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Connector::Rustls(Arc::new(config)))
}

fn realtime_url(api_base_url: &str, model: Option<&str>) -> Result<String> {
    let base = api_base_url.trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    let base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        return Err(anyhow!(
            "api base URL must be http or https: {api_base_url}"
        ));
    };
    let mut url = format!("{base}/v1/realtime?intent=transcription");
    if let Some(model) = model {
        url.push_str("&model=");
        url.push_str(&percent_encode(model));
    }
    Ok(url)
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                char::from(byte).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

fn session_update(options: &LiveTranscriptionOptions) -> Value {
    let turn_detection = options.turn_detection.map(|detection| {
        json!({
            "type": "server_vad",
            "threshold": detection.threshold,
            "prefix_padding_ms": detection.prefix_padding_ms,
            "silence_duration_ms": detection.silence_duration_ms,
        })
    });
    // The model is chosen by the handshake's query, where the node resolves
    // public ids and profiles; naming it here again would replace the
    // resolved model with the client's name for it.
    let transcription = json!({
        "language": options.language,
        "prompt": options.prompt,
    });
    json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": options.sample_rate},
                "transcription": transcription,
                "turn_detection": turn_detection,
            }},
        },
    })
}

fn parse_event(event: &Value) -> Option<LiveTranscriptionEvent> {
    let item_id = || event["item_id"].as_str().unwrap_or_default().to_string();
    let millis = |key: &str| event[key].as_u64().unwrap_or_default();
    Some(match event["type"].as_str()? {
        "input_audio_buffer.speech_started" => LiveTranscriptionEvent::SpeechStarted {
            item_id: item_id(),
            audio_start_ms: millis("audio_start_ms"),
        },
        "input_audio_buffer.speech_stopped" => LiveTranscriptionEvent::SpeechStopped {
            item_id: item_id(),
            audio_end_ms: millis("audio_end_ms"),
        },
        "input_audio_buffer.committed" => LiveTranscriptionEvent::Committed { item_id: item_id() },
        "conversation.item.input_audio_transcription.delta" => LiveTranscriptionEvent::Delta {
            item_id: item_id(),
            text: event["delta"].as_str().unwrap_or_default().to_string(),
        },
        "conversation.item.input_audio_transcription.completed" => {
            LiveTranscriptionEvent::Completed {
                item_id: item_id(),
                transcript: event["transcript"].as_str().unwrap_or_default().to_string(),
            }
        }
        "error" => LiveTranscriptionEvent::Error {
            message: event["error"]["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_string(),
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use tokio_tungstenite::WebSocketStream;

    type ServerSocket = WebSocketStream<tokio::net::TcpStream>;

    /// A one-connection node that confirms the session (or refuses it with
    /// `refusal`) and hands the socket and its authorization header to `then`.
    // The handshake callback's error type is tungstenite's.
    #[allow(clippy::result_large_err)]
    async fn fake_node<F, Fut, T>(
        refusal: Option<&'static str>,
        then: F,
    ) -> (String, tokio::task::JoinHandle<T>)
    where
        F: FnOnce(ServerSocket, Option<String>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = T> + Send,
        T: Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut authorization = None;
            let mut socket = tokio_tungstenite::accept_hdr_async(
                tcp,
                |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
                    authorization = request
                        .headers()
                        .get("authorization")
                        .map(|value| value.to_str().unwrap().to_string());
                    Ok(response)
                },
            )
            .await
            .unwrap();
            let Some(Ok(Message::Text(update))) = socket.next().await else {
                panic!("expected session.update");
            };
            assert!(update.contains("session.update"));
            let reply = match refusal {
                Some(message) => json!({"type": "error", "error": {"message": message}}),
                None => json!({"type": "session.updated"}),
            };
            socket
                .send(Message::Text(reply.to_string().into()))
                .await
                .unwrap();
            then(socket, authorization).await
        });
        (base, server)
    }

    /// Reads until the client's close frame, returning the binary frames
    /// seen on the way.
    async fn read_until_close(mut socket: ServerSocket) -> Vec<Vec<u8>> {
        let mut audio = Vec::new();
        let read = async {
            while let Some(message) = socket.next().await {
                match message.unwrap() {
                    Message::Binary(bytes) => audio.push(bytes.to_vec()),
                    Message::Close(_) => return,
                    _ => {}
                }
            }
            panic!("socket ended without a close frame");
        };
        tokio::time::timeout(Duration::from_secs(5), read)
            .await
            .expect("client closes the session");
        audio
    }

    #[tokio::test]
    async fn dropping_a_session_closes_it() {
        let (base, server) = fake_node(None, |socket, _| read_until_close(socket)).await;
        let live = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .unwrap();
        drop(live);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn dropping_the_audio_half_closes_a_split_session() {
        let (base, server) = fake_node(None, |socket, _| read_until_close(socket)).await;
        let live = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .unwrap();
        let (audio, _events) = live.split();
        drop(audio);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn audio_goes_out_as_little_endian_pcm_and_close_ends_it() {
        let (base, server) = fake_node(None, |socket, _| read_until_close(socket)).await;
        let live = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .unwrap();
        let (audio, _events) = live.split();
        audio.send_audio(&[0x0201, -2]).await.unwrap();
        audio.send_audio_f32(&[2.0, -2.0]).await.unwrap();
        audio.close().await;
        let frames = server.await.unwrap();
        assert_eq!(frames[0], [0x01, 0x02, 0xfe, 0xff]);
        assert_eq!(frames[1], [0xff, 0x7f, 0x01, 0x80]);
    }

    #[tokio::test]
    async fn a_refused_session_fails_to_start_with_the_reason() {
        let (base, _server) = fake_node(Some("no speech model on the mesh"), |socket, _| async {
            socket
        })
        .await;
        let error = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("no speech model on the mesh"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn api_keys_go_out_as_bearer_tokens() {
        let (base, server) = fake_node(None, |_, authorization| async { authorization }).await;
        let _live = LiveTranscription::start_with_api_key(
            &base,
            Some("k1"),
            LiveTranscriptionOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(server.await.unwrap().as_deref(), Some("Bearer k1"));
    }

    #[tokio::test]
    async fn a_lost_connection_is_reported_before_the_events_end() {
        // Dropping the socket without a close frame resets the connection.
        let (base, server) = fake_node(None, |socket, _| async { drop(socket) }).await;
        let mut live = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .unwrap();
        server.await.unwrap();
        let event = tokio::time::timeout(Duration::from_secs(5), live.next_event())
            .await
            .unwrap();
        assert!(
            matches!(&event, Some(LiveTranscriptionEvent::Error { message }) if message.contains("connection lost")),
            "{event:?}"
        );
        assert_eq!(live.next_event().await, None);
    }

    #[tokio::test]
    async fn stopping_with_nothing_buffered_returns_no_events() {
        let (base, server) = fake_node(None, |mut socket, _| async move {
            let Some(Ok(Message::Text(commit))) = socket.next().await else {
                panic!("expected a commit");
            };
            assert!(commit.contains("input_audio_buffer.commit"));
            let empty = json!({"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "input_audio_buffer_commit_empty: the buffer holds no audio",
            }});
            socket
                .send(Message::Text(empty.to_string().into()))
                .await
                .unwrap();
            read_until_close(socket).await
        })
        .await;
        let live = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .unwrap();
        assert_eq!(live.stop().await.unwrap(), Vec::new());
        server.await.unwrap();
    }

    /// Sends `events` as server events, in order.
    async fn send_events(socket: &mut ServerSocket, events: &[Value]) {
        for event in events {
            socket
                .send(Message::Text(event.to_string().into()))
                .await
                .unwrap();
        }
    }

    fn turn_event(kind: &str, item_id: &str) -> Value {
        match kind {
            "stopped" => json!({"type": "input_audio_buffer.speech_stopped", "item_id": item_id}),
            "committed" => json!({"type": "input_audio_buffer.committed", "item_id": item_id}),
            _ => json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": item_id,
            }),
        }
    }

    /// The node answers a commit after a pause ended `item_1`: that turn
    /// completes first, then the turn the commit ended.
    async fn finish_two_turns(mut socket: ServerSocket, pause_before_commit: bool) -> Vec<Vec<u8>> {
        if pause_before_commit {
            send_events(&mut socket, &[turn_event("stopped", "item_1")]).await;
        }
        let Some(Ok(Message::Text(commit))) = socket.next().await else {
            panic!("expected a commit");
        };
        assert!(commit.contains("input_audio_buffer.commit"));
        let mut events = Vec::new();
        if !pause_before_commit {
            events.push(turn_event("stopped", "item_1"));
        }
        events.extend([
            turn_event("committed", "item_1"),
            turn_event("committed", "item_2"),
            turn_event("completed", "item_1"),
            turn_event("completed", "item_2"),
        ]);
        send_events(&mut socket, &events).await;
        read_until_close(socket).await
    }

    fn completed_turns(events: &[LiveTranscriptionEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                LiveTranscriptionEvent::Completed { item_id, .. } => Some(item_id.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn stop_waits_for_the_turn_it_ended_not_an_earlier_one() {
        let (base, server) = fake_node(None, |socket, _| finish_two_turns(socket, false)).await;
        let live = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .unwrap();
        let remaining = live.stop().await.unwrap();
        assert_eq!(completed_turns(&remaining), ["item_1", "item_2"]);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn stop_knows_a_paused_turn_the_caller_already_saw_stop() {
        let (base, server) = fake_node(None, |socket, _| finish_two_turns(socket, true)).await;
        let mut live = LiveTranscription::start(&base, LiveTranscriptionOptions::default())
            .await
            .unwrap();
        assert!(matches!(
            live.next_event().await,
            Some(LiveTranscriptionEvent::SpeechStopped { .. })
        ));
        let remaining = live.stop().await.unwrap();
        assert_eq!(completed_turns(&remaining), ["item_1", "item_2"]);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn a_zero_sample_rate_is_refused() {
        let options = LiveTranscriptionOptions {
            sample_rate: 0,
            ..Default::default()
        };
        assert!(
            LiveTranscription::start("http://127.0.0.1:9/v1", options)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn wss_connects_through_tls_and_fails_cleanly() {
        // Nothing listens here; the point is reaching the connect with a TLS
        // configuration rather than panicking over the crypto provider.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let result = LiveTranscription::start(
            &format!("https://{address}/v1"),
            LiveTranscriptionOptions::default(),
        )
        .await;
        assert!(result.is_err());
    }

    #[test]
    fn realtime_urls_come_from_the_api_base_url() {
        assert_eq!(
            realtime_url("http://127.0.0.1:9337/v1", None).unwrap(),
            "ws://127.0.0.1:9337/v1/realtime?intent=transcription"
        );
        assert_eq!(
            realtime_url(
                "https://max.localhost/v1/",
                Some("ggerganov/whisper.cpp:ggml-base.en.bin")
            )
            .unwrap(),
            "wss://max.localhost/v1/realtime?intent=transcription&model=ggerganov%2Fwhisper.cpp%3Aggml-base.en.bin"
        );
        assert!(realtime_url("ftp://x/v1", None).is_err());
    }

    #[test]
    fn session_updates_carry_rate_language_and_push_to_talk_but_not_the_model() {
        let update = session_update(&LiveTranscriptionOptions {
            model: Some("public-whisper#fast".into()),
            language: Some("de".into()),
            sample_rate: 48_000,
            turn_detection: None,
            ..Default::default()
        });
        let input = &update["session"]["audio"]["input"];
        assert_eq!(input["format"]["rate"], 48_000);
        assert_eq!(input["transcription"]["language"], "de");
        assert_eq!(input["turn_detection"], Value::Null);
        assert!(input["transcription"].get("model").is_none());
    }

    #[test]
    fn server_events_map_to_typed_events() {
        let delta = json!({"type": "conversation.item.input_audio_transcription.delta", "item_id": "item_1", "delta": "And so"});
        assert_eq!(
            parse_event(&delta),
            Some(LiveTranscriptionEvent::Delta {
                item_id: "item_1".into(),
                text: "And so".into()
            })
        );
        let error = json!({"type": "error", "error": {"message": "nope"}});
        assert_eq!(
            parse_event(&error),
            Some(LiveTranscriptionEvent::Error {
                message: "nope".into()
            })
        );
        assert_eq!(
            parse_event(&json!({"type": "input_audio_buffer.committed", "item_id": "item_2"})),
            Some(LiveTranscriptionEvent::Committed {
                item_id: "item_2".into()
            })
        );
        assert_eq!(
            parse_event(&json!({"type": "input_audio_buffer.cleared"})),
            None
        );
    }
}
