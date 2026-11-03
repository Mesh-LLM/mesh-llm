//! Concurrent session handle shared by language bindings.
//!
//! Audio commands and event reads use independent queues. A waiting event read
//! cannot hold a lock needed to send the audio that produces the next event.

use super::{LiveTranscription, LiveTranscriptionEvent};
use anyhow::{Context, Result};
use tokio::sync::{Mutex, mpsc, oneshot};

enum Command {
    Audio(Vec<i16>, oneshot::Sender<Result<()>>),
    FloatAudio(Vec<f32>, oneshot::Sender<Result<()>>),
    Commit(oneshot::Sender<Result<()>>),
    Stop(oneshot::Sender<Result<Vec<LiveTranscriptionEvent>>>),
    Close(oneshot::Sender<()>),
}

/// A thread-safe live session with independent audio and event operations.
/// Dropping the last handle closes the underlying session.
pub struct LiveTranscriptionSession {
    commands: mpsc::Sender<Command>,
    events: Mutex<mpsc::Receiver<LiveTranscriptionEvent>>,
}

impl LiveTranscriptionSession {
    /// Own a running session and start its bounded command/event pump.
    pub fn new(live: LiveTranscription) -> Self {
        let (commands, receiver) = mpsc::channel(32);
        let (events, inbound) = mpsc::channel(256);
        tokio::spawn(run_session(live, receiver, events));
        Self {
            commands,
            events: Mutex::new(inbound),
        }
    }

    /// Send mono signed 16-bit PCM at the configured sample rate.
    pub async fn send_audio(&self, samples: Vec<i16>) -> Result<()> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Audio(samples, reply)).await?;
        result
            .await
            .context("live transcription command response dropped")?
    }

    /// Send float PCM; values outside -1..=1 are clipped by the Rust SDK.
    pub async fn send_audio_f32(&self, samples: Vec<f32>) -> Result<()> {
        let (reply, result) = oneshot::channel();
        self.send(Command::FloatAudio(samples, reply)).await?;
        result
            .await
            .context("live transcription command response dropped")?
    }

    /// End the current turn, retaining subsequent events for the reader.
    pub async fn commit(&self) -> Result<()> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Commit(reply)).await?;
        result
            .await
            .context("live transcription command response dropped")?
    }

    /// Read the next event, or `None` once the session and queued events finish.
    pub async fn next_event(&self) -> Option<LiveTranscriptionEvent> {
        self.events.lock().await.recv().await
    }

    /// Finish transcription and return events not already consumed by a reader.
    /// Once submitted, the stop command closes the session even if finishing
    /// fails or the caller stops waiting for its result.
    pub async fn stop(&self) -> Result<Vec<LiveTranscriptionEvent>> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Stop(reply)).await?;
        let remaining = result
            .await
            .context("live transcription stop response dropped")?;
        let mut events = self.events.lock().await;
        let mut buffered = Vec::new();
        while let Ok(event) = events.try_recv() {
            buffered.push(event);
        }
        buffered.extend(remaining?);
        Ok(buffered)
    }

    /// Close without waiting for a transcript. Calling this after closure is safe.
    pub async fn close(&self) {
        let (reply, closed) = oneshot::channel();
        if self.commands.send(Command::Close(reply)).await.is_ok() {
            let _ = closed.await;
        }
    }

    async fn send(&self, command: Command) -> Result<()> {
        self.commands
            .send(command)
            .await
            .map_err(|_| anyhow::anyhow!("live transcription session has closed"))
    }
}

async fn run_session(
    mut live: LiveTranscription,
    mut commands: mpsc::Receiver<Command>,
    events: mpsc::Sender<LiveTranscriptionEvent>,
) {
    let mut pending = None;
    loop {
        tokio::select! {
            _ = events.closed() => break,
            command = commands.recv() => match command {
                Some(Command::Audio(samples, reply)) => { let _ = reply.send(live.send_audio(&samples).await); }
                Some(Command::FloatAudio(samples, reply)) => { let _ = reply.send(live.send_audio_f32(&samples).await); }
                Some(Command::Commit(reply)) => { let _ = reply.send(live.commit().await); }
                Some(Command::Stop(reply)) => {
                    let result = live.stop().await.map(|mut remaining| {
                        if let Some(event) = pending.take() { remaining.insert(0, event); }
                        remaining
                    });
                    let _ = reply.send(result);
                    return;
                }
                Some(Command::Close(reply)) => { let _ = reply.send(()); break; }
                None => break,
            },
            permit = events.reserve(), if pending.is_some() => match permit {
                Ok(permit) => permit.send(pending.take().expect("pending event guarded above")),
                Err(_) => break,
            },
            event = live.next_event(), if pending.is_none() => match event {
                Some(event) => pending = Some(event),
                None => break,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_transcription::{
        LiveTranscriptionAudio, LiveTranscriptionEvents, LiveTranscriptionOptions, Outbound,
    };
    use std::{sync::Arc, time::Duration};

    fn fake_live() -> (
        LiveTranscription,
        mpsc::Receiver<Outbound>,
        mpsc::Sender<LiveTranscriptionEvent>,
    ) {
        let (outbound, writes) = mpsc::channel(32);
        let (input, inbound) = mpsc::channel(400);
        (
            LiveTranscription {
                audio: LiveTranscriptionAudio { outbound },
                events: LiveTranscriptionEvents {
                    inbound,
                    paused_turn: None,
                },
            },
            writes,
            input,
        )
    }

    #[tokio::test]
    async fn event_read_does_not_block_audio_and_stop_preserves_remaining_transcript() {
        let (live, mut writes, input) = fake_live();
        let session = Arc::new(LiveTranscriptionSession::new(live));
        let reader = {
            let session = session.clone();
            tokio::spawn(async move { session.next_event().await })
        };
        tokio::time::timeout(Duration::from_secs(1), session.send_audio(vec![1, -2]))
            .await
            .unwrap()
            .unwrap();
        match writes.recv().await.unwrap() {
            Outbound::Audio(bytes) => assert_eq!(bytes, [1, 0, 254, 255]),
            _ => panic!("expected audio"),
        }
        input
            .send(LiveTranscriptionEvent::Delta {
                item_id: "turn".into(),
                text: "hello".into(),
            })
            .await
            .unwrap();
        assert!(matches!(
            reader.await.unwrap(),
            Some(LiveTranscriptionEvent::Delta { .. })
        ));
        let server = tokio::spawn(async move {
            assert!(matches!(writes.recv().await, Some(Outbound::Event(_))));
            input
                .send(LiveTranscriptionEvent::Committed {
                    item_id: "turn".into(),
                })
                .await
                .unwrap();
            input
                .send(LiveTranscriptionEvent::Completed {
                    item_id: "turn".into(),
                    transcript: "hello".into(),
                })
                .await
                .unwrap();
        });
        let events = session.stop().await.unwrap();
        assert!(
            matches!(events.last(), Some(LiveTranscriptionEvent::Completed { transcript, .. }) if transcript == "hello")
        );
        server.await.unwrap();
        assert!(session.next_event().await.is_none());
        assert!(session.commit().await.is_err());
    }

    #[tokio::test]
    async fn full_event_queue_keeps_commands_responsive_and_resumes_when_drained() {
        let (live, mut writes, input) = fake_live();
        let session = LiveTranscriptionSession::new(live);
        for index in 0..300 {
            input
                .send(LiveTranscriptionEvent::Delta {
                    item_id: "turn".into(),
                    text: index.to_string(),
                })
                .await
                .unwrap();
        }
        tokio::time::timeout(Duration::from_secs(1), session.commit())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(writes.recv().await, Some(Outbound::Event(_))));
        for index in 0..300 {
            let event = tokio::time::timeout(Duration::from_secs(1), session.next_event())
                .await
                .unwrap();
            assert!(
                matches!(event, Some(LiveTranscriptionEvent::Delta { text, .. }) if text == index.to_string())
            );
        }
        session.close().await;
        assert!(session.next_event().await.is_none());
    }

    #[test]
    fn options_preserve_defaults_and_explicit_push_to_talk() {
        let defaults: LiveTranscriptionOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults, LiveTranscriptionOptions::default());
        let manual: LiveTranscriptionOptions =
            serde_json::from_str(r#"{"turn_detection":null,"sample_rate":48000}"#).unwrap();
        assert!(manual.turn_detection.is_none());
        assert_eq!(manual.sample_rate, 48000);
        assert!(
            serde_json::from_str::<LiveTranscriptionOptions>(r#"{"sampleRate":48000}"#).is_err()
        );
    }

    #[tokio::test]
    async fn close_unblocks_a_waiting_reader_and_releases_audio_without_more_input() {
        let (live, mut writes, _input) = fake_live();
        let session = Arc::new(LiveTranscriptionSession::new(live));
        let reader = {
            let session = session.clone();
            tokio::spawn(async move { session.next_event().await })
        };
        tokio::time::timeout(Duration::from_secs(1), session.close())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), reader)
                .await
                .unwrap()
                .unwrap()
                .is_none()
        );
        assert!(writes.recv().await.is_none());
        session.close().await;
    }
}
