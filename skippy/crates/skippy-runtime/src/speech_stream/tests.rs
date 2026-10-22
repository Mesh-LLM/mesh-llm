use super::*;
use crate::speech_recognition::TranscriptSegment;

/// Answers each pass with the next scripted hypothesis (segments as
/// `(end_ms, text)`) and records the buffer length and prompt it was given.
struct Script {
    passes: Vec<Vec<(i64, &'static str)>>,
    seen: Vec<(usize, Option<String>)>,
}

impl Script {
    fn new(passes: Vec<Vec<(i64, &'static str)>>) -> Self {
        Self {
            passes,
            seen: Vec::new(),
        }
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscriptionOptions,
    ) -> Result<Transcription> {
        let index = self.seen.len().min(self.passes.len() - 1);
        self.seen.push((samples.len(), options.prompt.clone()));
        Ok(Transcription {
            segments: self.passes[index]
                .iter()
                .map(|(end_ms, text)| TranscriptSegment {
                    start_ms: 0,
                    end_ms: *end_ms,
                    text: (*text).to_string(),
                    no_speech_prob: if text.starts_with('~') { 0.9 } else { 0.0 },
                })
                .collect(),
            language: "en".into(),
            duration_seconds: samples.len() as f64 / 16_000.0,
        })
    }
}

const MS: usize = 16;

fn loud(ms: usize) -> Vec<f32> {
    (0..ms * MS).map(|n| (n as f32 * 0.3).sin() * 0.3).collect()
}

fn quiet(ms: usize) -> Vec<f32> {
    vec![0.0; ms * MS]
}

fn manual() -> SpeechStreamConfig {
    SpeechStreamConfig {
        turn_detection: None,
        ..Default::default()
    }
}

fn deltas(events: &[SpeechStreamEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            SpeechStreamEvent::Delta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn words_are_confirmed_once_two_passes_agree() {
    let mut script = Script::new(vec![
        vec![(900, " Hello")],
        vec![(1_900, " Hello world")],
        vec![(2_900, " hello, world again")],
        vec![(3_000, " Hello world again today.")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());

    assert!(deltas(&stream.push(&loud(1_000), &mut transcribe).unwrap()).is_empty());
    assert_eq!(
        deltas(&stream.push(&loud(1_000), &mut transcribe).unwrap()),
        ["Hello"]
    );
    // Case and punctuation revisions still agree; the new pass's form is kept.
    assert_eq!(
        deltas(&stream.push(&loud(1_000), &mut transcribe).unwrap()),
        [" world"]
    );
    let done = stream.commit(&mut transcribe).unwrap();
    assert_eq!(deltas(&done), [" again today."]);
    assert_eq!(
        done.last(),
        Some(&SpeechStreamEvent::Completed {
            turn: 0,
            transcript: "Hello world again today.".into()
        })
    );
    assert!(!stream.has_pending_audio());
    assert_eq!(stream.current_turn(), 1);
}

#[test]
fn passes_wait_for_a_step_of_new_audio() {
    let mut script = Script::new(vec![vec![(500, " Hi")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    for _ in 0..4 {
        stream.push(&loud(240), &mut transcribe).unwrap();
    }
    assert!(script.seen.is_empty());
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    stream.push(&loud(240), &mut transcribe).unwrap();
    assert_eq!(script.seen.len(), 1);
}

#[test]
fn voice_activity_opens_and_closes_turns_with_prefix_padding() {
    let mut script = Script::new(vec![vec![(800, " Yes.")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig::default());

    // Silence alone never starts a turn or runs a pass.
    assert!(
        stream
            .push(&quiet(2_000), &mut transcribe)
            .unwrap()
            .is_empty()
    );
    let events = stream.push(&loud(400), &mut transcribe).unwrap();
    // Speech starts at 2000 ms; three frames confirm it, then 300 ms of
    // padding before that point is kept.
    assert_eq!(
        events,
        [SpeechStreamEvent::SpeechStarted {
            turn: 0,
            audio_start_ms: 1_700
        }]
    );
    let events = stream.push(&quiet(600), &mut transcribe).unwrap();
    assert_eq!(
        events,
        [
            SpeechStreamEvent::SpeechStopped {
                turn: 0,
                audio_end_ms: 2_400
            },
            SpeechStreamEvent::Delta {
                turn: 0,
                text: "Yes.".into()
            },
            SpeechStreamEvent::Completed {
                turn: 0,
                transcript: "Yes.".into()
            },
        ]
    );
    // The final pass saw the padding, the speech, and the trailing silence.
    assert_eq!(script.seen[0].0, (360 + 340 + 500) * MS);
}

#[test]
fn long_turns_trim_confirmed_segments_and_carry_them_as_prompt() {
    let config = SpeechStreamConfig {
        trim_after_ms: 3_000,
        prompt: Some("Mesh".into()),
        ..manual()
    };
    let mut script = Script::new(vec![
        vec![(1_000, " One two."), (3_500, " Three")],
        vec![(1_000, " One two."), (3_900, " Three four")],
        // After the cut at 1000 ms the buffer starts at "Three".
        vec![(3_400, " Three four five")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(config);
    stream.push(&loud(3_500), &mut transcribe).unwrap();
    let events = stream.push(&loud(1_000), &mut transcribe).unwrap();
    assert_eq!(deltas(&events), ["One two. Three"]);
    let events = stream.push(&loud(1_000), &mut transcribe).unwrap();
    assert_eq!(deltas(&events), [" four"]);
    assert_eq!(script.seen[1].0, 4_500 * MS);
    // 1000 ms of confirmed audio was cut before the third pass.
    assert_eq!(script.seen[2].0, (5_500 - 1_000) * MS);
    assert_eq!(script.seen[0].1.as_deref(), Some("Mesh"));
    assert_eq!(script.seen[2].1.as_deref(), Some("Mesh One two."));
}

#[test]
fn an_overlong_buffer_confirms_what_it_has_and_starts_over() {
    let config = SpeechStreamConfig {
        max_buffer_ms: 2_000,
        ..manual()
    };
    let mut script = Script::new(vec![vec![(2_400, " Never settles")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(config);
    let events = stream.push(&loud(2_500), &mut transcribe).unwrap();
    assert_eq!(deltas(&events), ["Never settles"]);
    assert!(!stream.has_pending_audio());
}

#[test]
fn segments_whisper_judges_silent_are_dropped() {
    let mut script = Script::new(vec![vec![(500, "~ Thank you."), (900, " Real words")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    stream.push(&loud(1_000), &mut transcribe).unwrap();
    let done = stream.commit(&mut transcribe).unwrap();
    assert_eq!(deltas(&done), ["Real words"]);
}

#[test]
fn commit_without_audio_reports_nothing_and_clear_drops_the_turn() {
    let mut script = Script::new(vec![vec![(900, " Dropped")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    assert!(stream.commit(&mut transcribe).unwrap().is_empty());
    stream.push(&loud(500), &mut transcribe).unwrap();
    assert!(stream.has_pending_audio());
    stream.clear();
    assert!(!stream.has_pending_audio());
    assert!(stream.commit(&mut transcribe).unwrap().is_empty());
    assert_eq!(stream.current_turn(), 0);
}

#[test]
fn word_matching_ignores_case_and_punctuation_only() {
    assert!(same_word("Hello,", "hello"));
    assert!(same_word("—", "—"));
    assert!(!same_word("—", "-"));
    assert!(!same_word("their", "there"));
}

/// Streams a real clip through a real model in 100 ms chunks when
/// `SKIPPY_ASR_TEST_MODEL` and `SKIPPY_ASR_TEST_AUDIO` are set.
#[test]
fn streams_a_real_clip_when_configured() {
    let (Some(model), Some(audio)) = (
        std::env::var_os("SKIPPY_ASR_TEST_MODEL"),
        std::env::var_os("SKIPPY_ASR_TEST_AUDIO"),
    ) else {
        return;
    };
    let device = std::env::var("SKIPPY_ASR_TEST_DEVICE").ok();
    let model = crate::SpeechRecognitionModel::open(model, device.as_deref()).unwrap();
    let samples = crate::decode_audio(&std::fs::read(audio).unwrap()).unwrap();
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| model.transcribe(s, o, || false);
    let mut stream = SpeechStream::new(SpeechStreamConfig::default());
    let mut events = Vec::new();
    for chunk in samples.chunks(1_600) {
        events.extend(stream.push(chunk, &mut transcribe).unwrap());
    }
    events.extend(stream.push(&quiet(1_000), &mut transcribe).unwrap());
    events.extend(stream.commit(&mut transcribe).unwrap());
    let completed = events
        .iter()
        .filter_map(|event| match event {
            SpeechStreamEvent::Completed { transcript, .. } => Some(transcript.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("deltas: {:?}", deltas(&events));
    eprintln!("completed: {completed}");
    let words = |text: &str| {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    assert!(
        deltas(&events).len() > 1,
        "words should arrive before the turn ends"
    );
    assert!(words(&completed).contains("ask not what your country can do for you"));
    // Deltas carry exactly the completed words, turn by turn.
    assert_eq!(words(&deltas(&events).join(" ")), words(&completed));
}
