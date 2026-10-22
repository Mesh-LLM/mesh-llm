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

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

/// How long [`LiveTranscription::stop`] waits for the final transcript.
const STOP_TIMEOUT: Duration = Duration::from_secs(15);
/// How long starting a session waits for the node to accept it.
const START_TIMEOUT: Duration = Duration::from_secs(30);
const QUEUE: usize = 256;

/// How a live transcription session transcribes.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveTranscriptionOptions {
    /// Model to use; `None` picks the first model on the mesh that supports
    /// live transcription.
    pub model: Option<String>,
    /// ISO 639-1 code such as `"en"`; `None` detects the language.
    pub language: Option<String>,
    /// Text that conditions transcription, such as names or jargon.
    pub prompt: Option<String>,
    /// Sample rate of the audio passed to [`LiveTranscription::send_audio`].
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
    /// The turn's whole transcript; no more deltas follow for it.
    Completed {
        item_id: String,
        transcript: String,
    },
    /// The session reported a problem; it may still continue.
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
/// halves for use from different tasks.
pub struct LiveTranscription {
    audio: LiveTranscriptionAudio,
    events: LiveTranscriptionEvents,
}

/// The sending half of a session.
#[derive(Clone)]
pub struct LiveTranscriptionAudio {
    outbound: mpsc::Sender<Outbound>,
}

/// The receiving half of a session.
pub struct LiveTranscriptionEvents {
    inbound: mpsc::Receiver<LiveTranscriptionEvent>,
}

impl LiveTranscription {
    /// Opens a session on `/v1/realtime` under `api_base_url`
    /// (for example `http://127.0.0.1:9337/v1`).
    pub async fn start(api_base_url: &str, options: LiveTranscriptionOptions) -> Result<Self> {
        let url = realtime_url(api_base_url, options.model.as_deref())?;
        let (socket, _) =
            tokio::time::timeout(START_TIMEOUT, tokio_tungstenite::connect_async(&url))
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
        tokio::spawn(async move {
            while let Some(message) = outbound_rx.recv().await {
                let frame = match message {
                    Outbound::Audio(bytes) => Message::Binary(bytes.into()),
                    Outbound::Event(text) => Message::Text(text.into()),
                    Outbound::Close => {
                        let _ = sink.send(Message::Close(None)).await;
                        break;
                    }
                };
                if sink.send(frame).await.is_err() {
                    break;
                }
            }
        });
        tokio::spawn(async move {
            while let Some(Ok(message)) = stream.next().await {
                let Message::Text(text) = message else {
                    continue;
                };
                let Some(event) = serde_json::from_str::<Value>(&text)
                    .ok()
                    .and_then(|event| parse_event(&event))
                else {
                    continue;
                };
                if inbound_tx.send(event).await.is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            audio: LiveTranscriptionAudio { outbound },
            events: LiveTranscriptionEvents { inbound },
        })
    }

    /// Sends 16-bit mono PCM at the session's sample rate.
    pub async fn send_audio(&self, samples: &[i16]) -> Result<()> {
        self.audio.send_audio(samples).await
    }

    /// Sends float PCM in -1.0..=1.0 at the session's sample rate.
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
    /// session. Returns the events that arrived meanwhile.
    pub async fn stop(self) -> Result<Vec<LiveTranscriptionEvent>> {
        let Self { audio, mut events } = self;
        audio.commit().await?;
        let mut remaining = Vec::new();
        let _ = tokio::time::timeout(STOP_TIMEOUT, async {
            while let Some(event) = events.next_event().await {
                let last = matches!(
                    event,
                    LiveTranscriptionEvent::Completed { .. } | LiveTranscriptionEvent::Error { .. }
                );
                remaining.push(event);
                if last {
                    break;
                }
            }
        })
        .await;
        audio.close().await;
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

    /// Sends float PCM in -1.0..=1.0 at the session's sample rate.
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

    /// Closes the session without waiting for a final transcript.
    pub async fn close(&self) {
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
        self.inbound.recv().await
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
    let mut transcription = json!({
        "language": options.language,
        "prompt": options.prompt,
    });
    if let Some(model) = &options.model {
        transcription["model"] = json!(model);
    }
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
    fn session_updates_carry_rate_language_and_push_to_talk() {
        let update = session_update(&LiveTranscriptionOptions {
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
            parse_event(&json!({"type": "input_audio_buffer.committed"})),
            None
        );
    }
}
