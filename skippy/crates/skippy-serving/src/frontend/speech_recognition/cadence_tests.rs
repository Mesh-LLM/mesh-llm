//! Queue coalescing must preserve the stream's audio-clock recognition windows.

use std::sync::{Condvar, Mutex};

use skippy_runtime::TranscriptSegment;

use super::*;

#[derive(Default)]
struct FirstPassGate {
    released: Mutex<bool>,
    wake: Condvar,
}

impl FirstPassGate {
    fn wait(&self) {
        let (released, timeout) = self
            .wake
            .wait_timeout_while(
                self.released.lock().unwrap(),
                Duration::from_secs(5),
                |released| !*released,
            )
            .unwrap();
        assert!(
            *released && !timeout.timed_out(),
            "the test must release its first pass"
        );
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_one();
    }
}

struct ReleaseOnDrop(Arc<FirstPassGate>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct WindowModel {
    seen: Mutex<Vec<(Vec<f32>, TranscriptionOptions)>>,
    called: mpsc::UnboundedSender<usize>,
    first: Option<Arc<FirstPassGate>>,
    fail_at: Option<usize>,
}

impl SpeechTranscriber for WindowModel {
    fn decode(&self, _audio: &[u8]) -> anyhow::Result<Vec<f32>> {
        unreachable!("live input already contains PCM")
    }

    fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
        _cancelled: &(dyn Fn() -> bool + Sync),
    ) -> anyhow::Result<Transcription> {
        let count = {
            let mut seen = self.seen.lock().unwrap();
            seen.push((samples.to_vec(), options.clone()));
            seen.len()
        };
        self.called.send(count).unwrap();
        if count == 1
            && let Some(gate) = &self.first
        {
            gate.wait();
        }
        if self.fail_at == Some(count) {
            anyhow::bail!("injected decoder failure");
        }
        Ok(Transcription {
            segments: vec![TranscriptSegment {
                start_ms: 0,
                end_ms: samples.len() as i64 / 16,
                text: " Heard audio".into(),
                no_speech_prob: 0.0,
            }],
            language: "en".into(),
            duration_seconds: samples.len() as f64 / 16_000.0,
        })
    }
}

type WindowObservation = (Vec<(Vec<f32>, TranscriptionOptions)>, Vec<RealtimeOutput>);

fn assert_same_windows(actual: &WindowObservation, expected: &WindowObservation) {
    assert_eq!(actual.0.len(), expected.0.len(), "native pass count");
    for (index, ((samples, options), (expected_samples, expected_options))) in
        actual.0.iter().zip(&expected.0).enumerate()
    {
        assert_eq!(samples.len(), expected_samples.len(), "pass {index} length");
        let first_difference = samples
            .iter()
            .zip(expected_samples)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected);
        assert!(
            samples == expected_samples,
            "pass {index} first sample difference: {first_difference:?}"
        );
        assert_eq!(options, expected_options, "pass {index} options");
    }
}

async fn run_session(backlogged: bool) -> WindowObservation {
    let (called, mut calls) = mpsc::unbounded_channel();
    let gate = Arc::new(FirstPassGate::default());
    let release = ReleaseOnDrop(gate.clone());
    let model = Arc::new(WindowModel {
        seen: Mutex::new(Vec::new()),
        called,
        first: backlogged.then_some(gate),
        fail_at: None,
    });
    let backend = SpeechRecognitionBackend::new("window-test", model.clone(), 1);
    let config = RealtimeTranscriptionConfig {
        input_sample_rate: 16_000,
        turn_detection: None,
        tentative: true,
        ..RealtimeTranscriptionConfig::new("window-test")
    };
    let mut session = backend
        .realtime_transcription(config, InferenceRequestContext::new())
        .await
        .unwrap();
    for index in 0..3 {
        session
            .input
            .send(RealtimeInput::Audio(vec![8_000 + index * 1_000; 16_000]))
            .await
            .unwrap();
        if index == 0 || !backlogged {
            let observed = tokio::time::timeout(Duration::from_secs(5), calls.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(observed, index as usize + 1);
        }
    }
    session.input.send(RealtimeInput::Commit).await.unwrap();
    drop(release); // All queued audio and its commit are present before release.
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(5), session.output.recv())
            .await
            .unwrap()
            .unwrap();
        let done = matches!(event, RealtimeOutput::Completed { .. });
        assert!(!matches!(event, RealtimeOutput::Failed { .. }), "{event:?}");
        events.push(event);
        if done {
            break;
        }
    }
    drop(session);
    let seen = model.seen.lock().unwrap().clone();
    (seen, events)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_queued_backlog_preserves_every_audio_clock_pass() {
    let control = run_session(false).await;
    assert_eq!(
        control
            .0
            .iter()
            .map(|(samples, _)| samples.len())
            .collect::<Vec<_>>(),
        [16_000, 32_000, 48_000, 48_000]
    );
    let delayed = run_session(true).await;
    assert_same_windows(&delayed, &control);
    assert_eq!(
        delayed.1, control.1,
        "deltas, tentative text, and completion stay ordered"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_mid_batch_decoder_error_ends_the_session_without_reusing_confirmation() {
    let (called, mut calls) = mpsc::unbounded_channel();
    let gate = Arc::new(FirstPassGate::default());
    let release = ReleaseOnDrop(gate.clone());
    let model = Arc::new(WindowModel {
        seen: Mutex::new(Vec::new()),
        called,
        first: Some(gate),
        fail_at: Some(3),
    });
    let backend = SpeechRecognitionBackend::new("window-test", model.clone(), 1);
    let config = RealtimeTranscriptionConfig {
        input_sample_rate: 16_000,
        turn_detection: None,
        tentative: true,
        ..RealtimeTranscriptionConfig::new("window-test")
    };
    let mut session = backend
        .realtime_transcription(config, InferenceRequestContext::new())
        .await
        .unwrap();
    session
        .input
        .send(RealtimeInput::Audio(vec![8_000; 16_000]))
        .await
        .unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), calls.recv())
            .await
            .unwrap(),
        Some(1)
    );
    // The next step-sized pushes contain a successful agreeing pass,
    // whose delta is delivered, followed by an error. The failed pass's
    // mutation cannot be resumed independently.
    session
        .input
        .send(RealtimeInput::Audio(vec![8_000; 32_000]))
        .await
        .unwrap();
    session.input.send(RealtimeInput::Commit).await.unwrap();
    drop(release);
    let mut failed = false;
    while let Some(event) = tokio::time::timeout(Duration::from_secs(5), session.output.recv())
        .await
        .unwrap()
    {
        match event {
            RealtimeOutput::Tentative { .. } if !failed => {}
            RealtimeOutput::Delta { .. } if !failed => {}
            RealtimeOutput::Failed { message } if !failed => {
                assert_eq!(message, "injected decoder failure");
                failed = true;
            }
            other => panic!("failed native turn must not emit or complete later: {other:?}"),
        }
    }
    assert!(failed);
    assert!(
        session
            .input
            .send(RealtimeInput::Audio(vec![8_000; 16_000]))
            .await
            .is_err()
    );
    assert_eq!(
        model.seen.lock().unwrap().len(),
        3,
        "queued commit and later input cannot transcribe again"
    );
}
