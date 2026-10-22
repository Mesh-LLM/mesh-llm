use std::sync::Arc;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message as ClientMessage;

use super::*;
use crate::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, InferenceBackend,
    InferenceResult, ModelObject,
};

#[test]
fn beta_session_updates_set_model_language_and_vad() {
    let config = apply_session_update(
        RealtimeTranscriptionConfig::new(""),
        &json!({
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper", "language": "de", "prompt": ""},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 800},
        }),
    )
    .unwrap();
    assert_eq!(config.model, "whisper");
    assert_eq!(config.language.as_deref(), Some("de"));
    assert_eq!(config.prompt, None);
    assert_eq!(config.input_sample_rate, 24_000);
    let detection = config.turn_detection.unwrap();
    assert_eq!(detection.silence_duration_ms, 800);
    assert_eq!(detection.prefix_padding_ms, 300);
}

#[test]
fn ga_session_updates_read_audio_input_and_can_disable_vad() {
    let config = apply_session_update(
        RealtimeTranscriptionConfig::new("whisper"),
        &json!({
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 16000},
                "transcription": {"prompt": "Mesh, Skippy"},
                "turn_detection": null,
            }},
        }),
    )
    .unwrap();
    assert_eq!(config.model, "whisper");
    assert_eq!(config.prompt.as_deref(), Some("Mesh, Skippy"));
    assert_eq!(config.input_sample_rate, 16_000);
    assert_eq!(config.turn_detection, None);
}

#[test]
fn unsupported_formats_and_turn_detection_are_refused() {
    let base = RealtimeTranscriptionConfig::new("whisper");
    assert!(
        apply_session_update(base.clone(), &json!({"input_audio_format": "g711_ulaw"})).is_err()
    );
    assert!(
        apply_session_update(base, &json!({"turn_detection": {"type": "push_to_talk"}})).is_err()
    );
}

#[test]
fn pcm16_is_little_endian_and_drops_a_trailing_byte() {
    assert_eq!(pcm16(&[0x01, 0x00, 0xff, 0xff, 0x7f]), vec![1, -1]);
}

/// Echoes audio as deltas of its sample count and completes on commit.
struct EchoBackend;

#[async_trait]
impl InferenceBackend for EchoBackend {
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("echo")])
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        unreachable!()
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        unreachable!()
    }

    async fn realtime_transcription(
        &self,
        config: RealtimeTranscriptionConfig,
        _context: InferenceRequestContext,
    ) -> InferenceResult<RealtimeTranscriptionSession> {
        assert_eq!(config.model, "echo");
        let (session, mut input, output) = RealtimeTranscriptionSession::channels();
        tokio::spawn(async move {
            let mut heard = 0usize;
            while let Some(command) = input.recv().await {
                match command {
                    RealtimeInput::Audio(samples) => {
                        heard += samples.len();
                        let _ = output
                            .send(RealtimeOutput::Delta {
                                turn: 0,
                                text: samples.len().to_string(),
                            })
                            .await;
                    }
                    RealtimeInput::Commit if heard == 0 => {
                        let _ = output.send(RealtimeOutput::CommitEmpty).await;
                    }
                    RealtimeInput::Commit => {
                        let _ = output.send(RealtimeOutput::Committed { turn: 0 }).await;
                        let _ = output
                            .send(RealtimeOutput::Completed {
                                turn: 0,
                                transcript: format!("{heard} samples"),
                            })
                            .await;
                    }
                    RealtimeInput::Clear | RealtimeInput::Configure(_) => {}
                }
            }
        });
        Ok(session)
    }
}

async fn connect(
    query: &str,
) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = crate::router_for(Arc::new(EchoBackend));
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let (socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{address}/v1/realtime{query}"))
            .await
            .unwrap();
    socket
}

async fn next_event(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Value {
    loop {
        match socket.next().await.unwrap().unwrap() {
            ClientMessage::Text(text) => return serde_json::from_str(&text).unwrap(),
            _ => continue,
        }
    }
}

async fn send(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    event: Value,
) {
    socket
        .send(ClientMessage::Text(event.to_string().into()))
        .await
        .unwrap();
}

#[tokio::test]
async fn a_session_streams_deltas_and_completes_on_commit() {
    let mut socket = connect("?intent=transcription").await;
    let created = next_event(&mut socket).await;
    assert_eq!(created["type"], "transcription_session.created");
    // With no ?model= the backend's only model is used.
    assert_eq!(
        created["session"]["input_audio_transcription"]["model"],
        "echo"
    );

    send(
        &mut socket,
        json!({"type": "transcription_session.update", "session": {"turn_detection": null}}),
    )
    .await;
    let updated = next_event(&mut socket).await;
    assert_eq!(updated["type"], "transcription_session.updated");
    assert_eq!(updated["session"]["turn_detection"], Value::Null);

    let audio = base64::engine::general_purpose::STANDARD.encode([0u8; 960]);
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": audio}),
    )
    .await;
    let delta = next_event(&mut socket).await;
    assert_eq!(
        delta["type"],
        "conversation.item.input_audio_transcription.delta"
    );
    assert_eq!(delta["delta"], "480");
    let item = delta["item_id"].as_str().unwrap().to_string();

    // Binary frames carry raw PCM without base64.
    socket
        .send(ClientMessage::Binary(vec![0u8; 320].into()))
        .await
        .unwrap();
    assert_eq!(next_event(&mut socket).await["delta"], "160");

    send(&mut socket, json!({"type": "input_audio_buffer.commit"})).await;
    let committed = next_event(&mut socket).await;
    assert_eq!(committed["type"], "input_audio_buffer.committed");
    assert_eq!(committed["item_id"], item.as_str());
    assert_eq!(committed["previous_item_id"], Value::Null);
    let completed = next_event(&mut socket).await;
    assert_eq!(
        completed["type"],
        "conversation.item.input_audio_transcription.completed"
    );
    assert_eq!(completed["transcript"], "640 samples");
}

#[tokio::test]
async fn ga_updates_are_answered_in_the_ga_shape_and_bad_events_are_errors() {
    let mut socket = connect("?model=echo").await;
    next_event(&mut socket).await;
    send(
        &mut socket,
        json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}}),
    )
    .await;
    let updated = next_event(&mut socket).await;
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(
        updated["session"]["audio"]["input"]["format"]["rate"],
        16000
    );

    send(&mut socket, json!({"type": "response.create"})).await;
    let error = next_event(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "invalid_request_error");

    send(&mut socket, json!({"type": "input_audio_buffer.commit"})).await;
    let empty = next_event(&mut socket).await;
    assert_eq!(empty["type"], "error");
    assert!(
        empty["error"]["message"]
            .as_str()
            .unwrap()
            .contains("commit_empty")
    );
}
