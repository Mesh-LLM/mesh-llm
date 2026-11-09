//! Opt-in observation of cold native recognition through the actual live backend.

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;

/// Renders a byte slice as lowercase hexadecimal.
fn digest_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct LiveTrace {
    file: Mutex<File>,
    started: Instant,
    pass: AtomicUsize,
    sent_samples: AtomicUsize,
}

impl LiveTrace {
    /// Appends one timestamped observation to the trace file.
    fn record(&self, kind: &str, detail: Value) {
        let mut file = self.file.lock().unwrap();
        let record = json!({
            "kind": kind,
            "elapsed_ms": self.started.elapsed().as_secs_f64() * 1_000.0,
            "pass": self.pass.load(Ordering::SeqCst),
            "sent_samples": self.sent_samples.load(Ordering::SeqCst),
            "detail": detail,
        });
        serde_json::to_writer(&mut *file, &record).unwrap();
        file.write_all(b"\n").unwrap();
        file.flush().unwrap();
    }
}

struct RecordingTranscriber {
    model: SpeechRecognitionModel,
    trace: Arc<LiveTrace>,
    backlog: Option<Arc<ExactBacklog>>,
}

/// Hold callback 28 until exactly frames 281..=300 are queued. The producer
/// cannot enqueue frame 301 until callback 29 has begun consuming that backlog.
#[derive(Default)]
struct ExactBacklog {
    sent_frames: Mutex<usize>,
    frames_arrived: Condvar,
    backlog_consumed: tokio::sync::Notify,
}

impl ExactBacklog {
    /// Holds callback 28 until the exact backlog frames have arrived.
    fn hold_callback(&self, pass: usize, trace: &LiveTrace) {
        if pass != 28 {
            return;
        }
        let started = Instant::now();
        let (frames, timeout) = self
            .frames_arrived
            .wait_timeout_while(
                self.sent_frames.lock().unwrap(),
                Duration::from_secs(10),
                |frames| *frames < 300,
            )
            .unwrap();
        assert!(!timeout.timed_out(), "producer must reach frame 300");
        assert_eq!(
            *frames, 300,
            "producer pauses at the exact backlog boundary"
        );
        trace.record(
            "backlog_released",
            json!({
                "held_pass": pass,
                "sent_frames": *frames,
                "additional_hold_ms": started.elapsed().as_secs_f64() * 1_000.0,
            }),
        );
    }

    /// Waits for the expected frame to leave the paced producer.
    async fn frame_sent(&self, frame: usize) {
        *self.sent_frames.lock().unwrap() = frame;
        self.frames_arrived.notify_one();
        if frame == 300 {
            tokio::time::timeout(Duration::from_secs(10), self.backlog_consumed.notified())
                .await
                .expect("the next callback must consume the held backlog");
        }
    }
}

impl SpeechTranscriber for RecordingTranscriber {
    /// Rejects decoded input; the live session already carries PCM.
    fn decode(&self, audio: &[u8]) -> anyhow::Result<Vec<f32>> {
        skippy_runtime::decode_audio(audio)
    }

    /// Records the native pass and answers the scripted transcript.
    fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> anyhow::Result<Transcription> {
        let pass = self.trace.pass.fetch_add(1, Ordering::SeqCst) + 1;
        let mut digest = Sha256::new();
        for sample in samples {
            digest.update(sample.to_le_bytes());
        }
        self.trace.record(
            "pass_start",
            json!({
                "samples": samples.len(),
                "samples_f32_sha256": digest_hex(&digest.finalize()),
                "language": options.language,
                "prompt": options.prompt,
                "temperature": options.temperature,
                "threads": options.threads,
            }),
        );
        if let Some(backlog) = &self.backlog {
            if pass == 28 {
                assert_eq!(
                    samples.len(),
                    96_000,
                    "delay the established 28-second window"
                );
            } else if pass == 29 {
                assert_eq!(
                    samples.len(),
                    112_000,
                    "the intermediate one-second window must survive the queued backlog"
                );
                backlog.backlog_consumed.notify_one();
            }
        }
        let started = Instant::now();
        let result = self.model.transcribe(samples, options, cancelled);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        match &result {
            Ok(transcript) => self.trace.record(
                "pass_result",
                json!({
                    "transcribe_ms": elapsed_ms,
                    "samples": samples.len(),
                    "segments": transcript.segments.iter().map(|segment| json!({
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "text": segment.text,
                        "no_speech_prob": segment.no_speech_prob,
                    })).collect::<Vec<_>>(),
                    "text": transcript.text(),
                }),
            ),
            Err(error) => self.trace.record(
                "pass_error",
                json!({
                    "transcribe_ms": elapsed_ms,
                    "message": format!("{error:#}"),
                }),
            ),
        }
        if let Some(backlog) = &self.backlog {
            backlog.hold_callback(pass, &self.trace);
        }
        result
    }
}

/// Builds the fixture PCM the transcript phrases are asserted against.
fn pinned_pcm() -> Vec<i16> {
    let path = std::env::var_os("SKIPPY_ASR_TEST_PCM").expect("set SKIPPY_ASR_TEST_PCM");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        bytes.len(),
        1_408_000,
        "four copies of the 11-second JFK clip"
    );
    assert_eq!(
        digest_hex(&Sha256::digest(&bytes)),
        "8cda406b2cf6e25a495646c906762a4033129b4acdf09f4da40d3502f71c2008"
    );
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// Wraps the native recognition model with tracing and an optional backlog.
fn native_model(trace: Arc<LiveTrace>, backlog: Option<Arc<ExactBacklog>>) -> RecordingTranscriber {
    skippy_runtime::test_support::require_native_runtime();
    let path = std::env::var_os("SKIPPY_ASR_TEST_MODEL").expect("set SKIPPY_ASR_TEST_MODEL");
    assert_eq!(
        digest_hex(&Sha256::digest(std::fs::read(&path).unwrap())),
        "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"
    );
    let model = SpeechRecognitionModel::open(path, Some("MTL0")).unwrap();
    assert!(model.info().on_accelerator, "the diagnostic requires Metal");
    trace.record("model", json!({ "device": "MTL0", "on_accelerator": true }));
    RecordingTranscriber {
        model,
        trace,
        backlog,
    }
}

/// Paces the fixture PCM through the live session in fixed frames.
async fn send_paced_audio(
    input: mpsc::Sender<RealtimeInput>,
    pcm: Vec<i16>,
    trace: Arc<LiveTrace>,
    backlog: Option<Arc<ExactBacklog>>,
) {
    for (frame, samples) in pcm.as_chunks::<1_600>().0.iter().enumerate() {
        input
            .send(RealtimeInput::Audio(samples.to_vec()))
            .await
            .unwrap();
        trace
            .sent_samples
            .fetch_add(samples.len(), Ordering::SeqCst);
        trace.record(
            "audio_sent",
            json!({ "frame": frame + 1, "samples": samples.len() }),
        );
        if let Some(backlog) = &backlog {
            backlog.frame_sent(frame + 1).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    input.send(RealtimeInput::Commit).await.unwrap();
    trace.record("commit_sent", json!({}));
}

/// Projects the realtime output fields the test records.
fn output_detail(output: &RealtimeOutput) -> Value {
    match output {
        RealtimeOutput::Delta { turn, text } => {
            json!({"type": "delta", "turn": turn, "text": text})
        }
        RealtimeOutput::Tentative { turn, text } => {
            json!({"type": "tentative", "turn": turn, "text": text})
        }
        RealtimeOutput::Completed { turn, transcript } => {
            json!({"type": "completed", "turn": turn, "text": transcript})
        }
        RealtimeOutput::Committed { turn } => json!({"type": "committed", "turn": turn}),
        other => json!({"type": "unexpected", "event": format!("{other:?}")}),
    }
}

/// Reads the session events until completion and collects the transcript.
async fn collect_transcript(
    output: &mut mpsc::Receiver<RealtimeOutput>,
    trace: &LiveTrace,
) -> (String, String) {
    let mut confirmed = String::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(90), output.recv())
            .await
            .expect("native session must answer within 90 seconds")
            .expect("native session must complete");
        trace.record("event", output_detail(&event));
        match event {
            RealtimeOutput::Delta { text, .. } => confirmed.push_str(&text),
            RealtimeOutput::Completed { transcript, .. } => return (confirmed, transcript),
            RealtimeOutput::Tentative { .. } | RealtimeOutput::Committed { .. } => {}
            other => panic!("unexpected event {other:?}"),
        }
    }
}

/// Counts non-overlapping word-window occurrences of a phrase.
fn phrase_occurrences(text: &str, phrase: &str) -> usize {
    let words = text
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    let phrase = phrase
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    words
        .windows(phrase.len())
        .filter(|window| *window == phrase.as_slice())
        .count()
}

/// A cold Metal live session preserves four paced utterances.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires pinned Tiny/JFK PCM fixtures, a packaged Metal runtime, and a SKIPPY_ASR_TRACE_PATH base directory; takes 44 seconds"]
async fn cold_metal_live_session_preserves_four_paced_utterances() {
    run_paced_session(
        "cold_metal_live_session_preserves_four_paced_utterances",
        None,
    )
    .await;
}

/// A cold Metal live session preserves utterances after one backlog.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires pinned Tiny/JFK PCM fixtures and packaged Metal; holds callback 28 for an exact two-second audio backlog"]
async fn cold_metal_live_session_preserves_utterances_after_one_backlog() {
    run_paced_session(
        "cold_metal_live_session_preserves_utterances_after_one_backlog",
        Some(Arc::new(ExactBacklog::default())),
    )
    .await;
}

/// Runs one paced live session and asserts the counted transcript phrases.
async fn run_paced_session(name: &str, backlog: Option<Arc<ExactBacklog>>) {
    // SKIPPY_ASR_TRACE_PATH is a base directory: each test writes its own
    // file under it, so parallel ignored runs cannot truncate one
    // another's records. An existing record is refused, never truncated.
    let base = std::env::var_os("SKIPPY_ASR_TRACE_PATH").expect("set SKIPPY_ASR_TRACE_PATH");
    let base = std::path::PathBuf::from(base);
    std::fs::create_dir_all(&base).unwrap();
    let trace = Arc::new(LiveTrace {
        file: Mutex::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(base.join(format!("{name}.jsonl")))
                .expect(
                    "SKIPPY_ASR_TRACE_PATH already holds this test's trace; use a fresh directory",
                ),
        ),
        started: Instant::now(),
        pass: AtomicUsize::new(0),
        sent_samples: AtomicUsize::new(0),
    });
    let pcm = pinned_pcm();
    let backend = SpeechRecognitionBackend::new(
        "pinned-whisper-tiny",
        Arc::new(native_model(trace.clone(), backlog.clone())),
        1,
    );
    let config = RealtimeTranscriptionConfig {
        language: Some("en".into()),
        input_sample_rate: 16_000,
        turn_detection: None,
        step_ms: None,
        tentative: true,
        ..RealtimeTranscriptionConfig::new("pinned-whisper-tiny")
    };
    trace.record(
        "config",
        json!({
            "input_sample_rate": config.input_sample_rate,
            "step_ms_requested": config.step_ms,
            "step_ms_resolved": stream_config(&config).step_ms,
            "manual_turn": config.turn_detection.is_none(),
            "tentative": config.tentative,
            "language": config.language,
            "prompt": config.prompt,
        }),
    );
    let mut session = backend
        .realtime_transcription(config, InferenceRequestContext::new())
        .await
        .unwrap();
    let sender = tokio::spawn(send_paced_audio(
        session.input.clone(),
        pcm,
        trace.clone(),
        backlog,
    ));
    let (confirmed, transcript) = collect_transcript(&mut session.output, &trace).await;
    sender.await.unwrap();
    drop(session);
    assert_eq!(trace.sent_samples.load(Ordering::SeqCst), 704_000);
    assert_eq!(confirmed, transcript);
    // The utterance opening must survive in both the confirmed (deltas) and
    // the completed transcript; the decoder may render it as "asked not",
    // but an omitted opening cannot pass.
    for (surface, text) in [
        ("confirmed", confirmed.as_str()),
        ("completed", transcript.as_str()),
    ] {
        let opening = phrase_occurrences(text, "ask not") + phrase_occurrences(text, "asked not");
        assert_eq!(
            opening, 4,
            "opening missing from the {surface} transcript: {text}"
        );
    }
    let counts = [
        "fellow americans",
        "can do for you",
        "what you can do",
        "do for your country",
        "for your country",
    ]
    .map(|phrase| (phrase, phrase_occurrences(&transcript, phrase)));
    trace.record(
        "result",
        json!({"transcript": transcript, "clause_counts": counts}),
    );
    for (phrase, count) in counts {
        assert_eq!(count, 4, "clause {phrase:?}: {transcript}");
    }
}
