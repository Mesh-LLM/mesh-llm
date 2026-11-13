use super::*;
pub(super) use mesh_llm_embedded_runtime::native::transcription::RealtimeInput;
use mesh_llm_embedded_runtime::native::transcription::{
    RealtimeOutput, RealtimeTranscriptionConfig, RealtimeTurnDetection, TranscriptionAudio,
    TranscriptionEvents, TranscriptionSession,
};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) enum AudioTransport {
    Socket(mpsc::Sender<Outbound>),
    Native(TranscriptionAudio),
}

pub(super) enum EventTransport {
    Socket(mpsc::Receiver<LiveTranscriptionEvent>),
    Native {
        events: TranscriptionEvents,
        prefix: String,
    },
}

impl LiveTranscription {
    pub(crate) async fn start_native(
        node: &mesh_llm_embedded_runtime::EmbeddedMeshNodeHandle,
        options: LiveTranscriptionOptions,
    ) -> Result<Self> {
        let config = RealtimeTranscriptionConfig {
            model: options.model.unwrap_or_default(),
            language: options.language,
            prompt: options.prompt,
            input_sample_rate: options.sample_rate,
            turn_detection: options.turn_detection.map(|vad| RealtimeTurnDetection {
                threshold: vad.threshold,
                prefix_padding_ms: vad.prefix_padding_ms,
                silence_duration_ms: vad.silence_duration_ms,
            }),
            step_ms: options.step_ms,
            tentative: options.tentative,
        };
        Ok(Self::from_native(node.live_transcription(config).await?))
    }

    fn from_native(session: TranscriptionSession) -> Self {
        static NEXT_SESSION: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        Self {
            audio: LiveTranscriptionAudio {
                outbound: AudioTransport::Native(session.audio),
            },
            events: LiveTranscriptionEvents {
                inbound: EventTransport::Native {
                    events: session.events,
                    prefix: format!("item_native_{id}"),
                },
                paused_turn: None,
            },
        }
    }
}

impl EventTransport {
    pub(super) async fn next_event(&mut self) -> Option<LiveTranscriptionEvent> {
        match self {
            Self::Socket(receiver) => receiver.recv().await,
            Self::Native { events, prefix } => events
                .next_output()
                .await
                .map(|event| typed_event(prefix, event)),
        }
    }
}

fn typed_event(prefix: &str, event: RealtimeOutput) -> LiveTranscriptionEvent {
    let item_id = |turn| format!("{prefix}_{turn}");
    match event {
        RealtimeOutput::SpeechStarted {
            turn,
            audio_start_ms,
        } => LiveTranscriptionEvent::SpeechStarted {
            item_id: item_id(turn),
            audio_start_ms,
        },
        RealtimeOutput::SpeechStopped { turn, audio_end_ms } => {
            LiveTranscriptionEvent::SpeechStopped {
                item_id: item_id(turn),
                audio_end_ms,
            }
        }
        RealtimeOutput::Committed { turn } => LiveTranscriptionEvent::Committed {
            item_id: item_id(turn),
        },
        RealtimeOutput::Delta { turn, text } => LiveTranscriptionEvent::Delta {
            item_id: item_id(turn),
            text,
        },
        RealtimeOutput::Completed { turn, transcript } => LiveTranscriptionEvent::Completed {
            item_id: item_id(turn),
            transcript,
        },
        RealtimeOutput::Tentative { turn, text } => LiveTranscriptionEvent::Tentative {
            item_id: item_id(turn),
            text,
        },
        RealtimeOutput::CommitEmpty => LiveTranscriptionEvent::Error {
            message: "input_audio_buffer_commit_empty: the buffer holds no audio".into(),
        },
        RealtimeOutput::Failed { message } => LiveTranscriptionEvent::Error { message },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_results_keep_the_typed_event_contract_and_turn_identity() {
        let cases = [
            (
                RealtimeOutput::SpeechStarted {
                    turn: 2,
                    audio_start_ms: 10,
                },
                LiveTranscriptionEvent::SpeechStarted {
                    item_id: "item_session_2".into(),
                    audio_start_ms: 10,
                },
            ),
            (
                RealtimeOutput::SpeechStopped {
                    turn: 2,
                    audio_end_ms: 20,
                },
                LiveTranscriptionEvent::SpeechStopped {
                    item_id: "item_session_2".into(),
                    audio_end_ms: 20,
                },
            ),
            (
                RealtimeOutput::Committed { turn: 2 },
                LiveTranscriptionEvent::Committed {
                    item_id: "item_session_2".into(),
                },
            ),
            (
                RealtimeOutput::Delta {
                    turn: 2,
                    text: "hello".into(),
                },
                LiveTranscriptionEvent::Delta {
                    item_id: "item_session_2".into(),
                    text: "hello".into(),
                },
            ),
            (
                RealtimeOutput::Tentative {
                    turn: 2,
                    text: "world".into(),
                },
                LiveTranscriptionEvent::Tentative {
                    item_id: "item_session_2".into(),
                    text: "world".into(),
                },
            ),
            (
                RealtimeOutput::Completed {
                    turn: 2,
                    transcript: "hello world".into(),
                },
                LiveTranscriptionEvent::Completed {
                    item_id: "item_session_2".into(),
                    transcript: "hello world".into(),
                },
            ),
            (
                RealtimeOutput::CommitEmpty,
                LiveTranscriptionEvent::Error {
                    message: "input_audio_buffer_commit_empty: the buffer holds no audio".into(),
                },
            ),
            (
                RealtimeOutput::Failed {
                    message: "no decoder".into(),
                },
                LiveTranscriptionEvent::Error {
                    message: "no decoder".into(),
                },
            ),
        ];
        for (native, expected) in cases {
            assert_eq!(typed_event("item_session", native), expected);
        }
    }
}
