//! In-process live transcription for embedded nodes.

use super::EmbeddedServeHandle;
use anyhow::Result;
use skippy_inference_api::InferenceRequestContext;
use tokio::sync::{mpsc, watch};

pub use skippy_inference_api::realtime::{
    RealtimeInput, RealtimeOutput, RealtimeTranscriptionConfig, RealtimeTurnDetection,
};

/// Native audio and events backed by the loaded speech model's channels.
pub struct TranscriptionSession {
    pub audio: TranscriptionAudio,
    pub events: TranscriptionEvents,
}

/// Dropping either half cancels the backend and releases runtime admission.
pub struct TranscriptionAudio {
    input: mpsc::Sender<RealtimeInput>,
    context: InferenceRequestContext,
}

pub struct TranscriptionEvents {
    output: mpsc::Receiver<RealtimeOutput>,
    context: InferenceRequestContext,
    finished: bool,
}

impl TranscriptionSession {
    pub(crate) fn new(
        session: skippy_inference_api::realtime::RealtimeTranscriptionSession,
        context: InferenceRequestContext,
        lifetime: impl Send + 'static,
    ) -> Self {
        let cancellation = context.cancellation_token();
        tokio::spawn(async move {
            let _lifetime = lifetime;
            cancellation.cancelled().await;
        });
        Self {
            audio: TranscriptionAudio {
                input: session.input,
                context: context.clone(),
            },
            events: TranscriptionEvents {
                output: session.output,
                context,
                finished: false,
            },
        }
    }

    fn cancel_on_shutdown(self, mut shutdown: watch::Receiver<bool>) -> Self {
        let context = self.audio.context.clone();
        tokio::spawn(async move {
            let cancellation = context.cancellation_token();
            if !*shutdown.borrow_and_update() {
                tokio::select! {
                    _ = cancellation.cancelled() => return,
                    _ = shutdown.changed() => {},
                }
            }
            context.cancel();
        });
        self
    }
}

impl TranscriptionAudio {
    pub async fn send(&self, input: RealtimeInput) -> Result<()> {
        let cancellation = self.context.cancellation_token();
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => anyhow::bail!("live transcription session has closed"),
            sent = self.input.send(input) => sent.map_err(|_| anyhow::anyhow!("live transcription session has closed")),
        }
    }
}

impl Drop for TranscriptionAudio {
    fn drop(&mut self) {
        self.context.cancel();
    }
}

impl TranscriptionEvents {
    pub async fn next_output(&mut self) -> Option<RealtimeOutput> {
        if self.finished {
            return None;
        }
        let cancellation = self.context.cancellation_token();
        let output = tokio::select! {
            biased;
            _ = cancellation.cancelled() => None,
            output = self.output.recv() => match output {
                Some(output) => Some(output),
                None => Some(RealtimeOutput::Failed { message: "native transcription backend closed".into() }),
            },
        };
        if output.is_none() || matches!(output, Some(RealtimeOutput::Failed { .. })) {
            self.finished = true;
            self.context.cancel();
        }
        output
    }
}

impl Drop for TranscriptionEvents {
    fn drop(&mut self) {
        self.context.cancel();
    }
}

impl EmbeddedServeHandle {
    /// Open a local native speech session. Remote-only models fail explicitly;
    /// this path never connects to an HTTP or WebSocket endpoint.
    pub async fn live_transcription(
        &self,
        config: RealtimeTranscriptionConfig,
    ) -> Result<TranscriptionSession> {
        let mut shutdown = self.shutdown.subscribe();
        anyhow::ensure!(
            !*shutdown.borrow_and_update(),
            "embedded runtime is shutting down"
        );
        let node = self.native_node()?;
        tokio::select! {
            biased;
            _ = shutdown.changed() => anyhow::bail!("embedded runtime is shutting down"),
            session = node.native_live_transcription(config) => {
                Ok(session?.cancel_on_shutdown(self.shutdown.subscribe()))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::oneshot;

    struct Released(Option<oneshot::Sender<()>>);
    impl Drop for Released {
        fn drop(&mut self) {
            let _ = self.0.take().unwrap().send(());
        }
    }

    fn fake() -> (
        TranscriptionSession,
        mpsc::Receiver<RealtimeInput>,
        mpsc::Sender<RealtimeOutput>,
        oneshot::Receiver<()>,
    ) {
        let (backend, input, output) =
            skippy_inference_api::realtime::RealtimeTranscriptionSession::channels();
        let (released, observed) = oneshot::channel();
        (
            TranscriptionSession::new(
                backend,
                InferenceRequestContext::new(),
                Released(Some(released)),
            ),
            input,
            output,
            observed,
        )
    }

    #[tokio::test]
    async fn native_channels_forward_audio_commands_and_results_without_a_socket() {
        let (mut live, mut input, output, released) = fake();
        live.audio
            .send(RealtimeInput::Audio(vec![1, -2]))
            .await
            .unwrap();
        assert_eq!(input.recv().await, Some(RealtimeInput::Audio(vec![1, -2])));
        live.audio.send(RealtimeInput::Commit).await.unwrap();
        assert_eq!(input.recv().await, Some(RealtimeInput::Commit));
        for event in [
            RealtimeOutput::Delta {
                turn: 0,
                text: "Hello".into(),
            },
            RealtimeOutput::Completed {
                turn: 0,
                transcript: "Hello world".into(),
            },
            RealtimeOutput::Failed {
                message: "decode failed".into(),
            },
        ] {
            output.send(event.clone()).await.unwrap();
            assert_eq!(live.events.next_output().await, Some(event));
        }
        drop(live.audio);
        tokio::time::timeout(Duration::from_secs(5), released)
            .await
            .unwrap()
            .unwrap();
        assert!(live.events.next_output().await.is_none());
        assert!(live.events.next_output().await.is_none());
    }

    #[tokio::test]
    async fn dropping_the_reader_cancels_a_blocked_audio_send_and_releases_admission() {
        let (live, _input, _output, released) = fake();
        for _ in 0..256 {
            live.audio.send(RealtimeInput::Commit).await.unwrap();
        }
        let pending = tokio::spawn(async move { live.audio.send(RealtimeInput::Commit).await });
        drop(live.events);
        assert!(
            tokio::time::timeout(Duration::from_secs(5), pending)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        tokio::time::timeout(Duration::from_secs(5), released)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn runtime_shutdown_cancels_unpolled_audio_and_releases_admission() {
        let (live, _input, _output, released) = fake();
        let (shutdown, watch) = watch::channel(false);
        let mut live = live.cancel_on_shutdown(watch);
        shutdown.send_replace(true);
        tokio::time::timeout(Duration::from_secs(5), released)
            .await
            .unwrap()
            .unwrap();
        assert!(live.audio.send(RealtimeInput::Commit).await.is_err());
        assert!(live.events.next_output().await.is_none());
    }
    #[tokio::test]
    async fn a_lost_backend_reports_one_error_and_releases_admission() {
        let (mut live, _input, output, released) = fake();
        drop(output);
        assert!(
            matches!(live.events.next_output().await, Some(RealtimeOutput::Failed { message }) if message.contains("backend closed"))
        );
        assert!(live.events.next_output().await.is_none());
        assert!(live.audio.send(RealtimeInput::Commit).await.is_err());
        tokio::time::timeout(Duration::from_secs(5), released)
            .await
            .unwrap()
            .unwrap();
    }
}
