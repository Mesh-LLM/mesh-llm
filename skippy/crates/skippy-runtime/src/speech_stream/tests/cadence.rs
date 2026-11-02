//! Audio-clock cadence must not depend on network packet or queue boundaries.

use super::*;

#[derive(PartialEq)]
struct ObservedPass {
    samples: Vec<f32>,
    options: TranscriptionOptions,
}

impl std::fmt::Debug for ObservedPass {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObservedPass")
            .field("samples", &self.samples.len())
            .field("first", &self.samples.first())
            .field("last", &self.samples.last())
            .field("options", &self.options)
            .finish()
    }
}

#[derive(Default)]
struct AudioWords {
    seen: Vec<ObservedPass>,
}

impl AudioWords {
    // Each second's amplitude identifies a word, so hypotheses depend on the
    // actual audio window rather than the number of times this model is called.
    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscriptionOptions,
    ) -> Result<Transcription> {
        self.seen.push(ObservedPass {
            samples: samples.to_vec(),
            options: options.clone(),
        });
        Ok(Transcription {
            segments: samples
                .chunks(1_000 * MS)
                .enumerate()
                .map(|(index, chunk)| TranscriptSegment {
                    start_ms: (index * 1_000) as i64,
                    end_ms: (index * 1_000 + chunk.len() / MS) as i64,
                    text: format!(" word{}", (chunk[0] * 100.0).round() as usize),
                    no_speech_prob: 0.0,
                })
                .collect(),
            language: "en".into(),
            duration_seconds: samples.len() as f64 / 16_000.0,
        })
    }
}

fn word_audio(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|index| (10 + index / (1_000 * MS)) as f32 / 100.0)
        .collect()
}

fn push_parts(
    stream: &mut SpeechStream,
    model: &mut AudioWords,
    audio: &[f32],
    parts: &[usize],
) -> Vec<SpeechStreamEvent> {
    let mut events = Vec::new();
    let mut offset = 0;
    for size in parts.iter().cycle() {
        let end = (offset + size).min(audio.len());
        events.extend(
            stream
                .push(&audio[offset..end], &mut |s, o| model.transcribe(s, o))
                .unwrap(),
        );
        offset = end;
        if offset == audio.len() {
            break;
        }
    }
    events
}

fn observed_turn(
    audio: &[f32],
    parts: &[usize],
    config: SpeechStreamConfig,
) -> (Vec<ObservedPass>, Vec<SpeechStreamEvent>) {
    let mut stream = SpeechStream::new(config);
    let mut model = AudioWords::default();
    let mut events = push_parts(&mut stream, &mut model, audio, parts);
    events.extend(stream.commit(&mut |s, o| model.transcribe(s, o)).unwrap());
    assert!(!stream.has_pending_audio());
    (model.seen, events)
}

#[test]
fn whole_small_and_partial_frame_packets_have_identical_windows_and_events() {
    let audio = word_audio(4_500 * MS + 17);
    let config = SpeechStreamConfig {
        tentative: true,
        ..manual()
    };
    let expected = observed_turn(&audio, &[100 * MS], config.clone());
    assert_eq!(
        expected
            .0
            .iter()
            .map(|pass| pass.samples.len())
            .collect::<Vec<_>>(),
        [16_000, 32_000, 48_000, 64_000, 72_017]
    );
    for parts in [vec![audio.len()], vec![17, 513, 31, 4_901]] {
        assert_eq!(observed_turn(&audio, &parts, config.clone()), expected);
    }
}

#[test]
fn non_frame_multiple_steps_keep_the_same_next_frame_cadence() {
    let audio = word_audio(1_000 * MS + 17);
    let config = SpeechStreamConfig {
        step_ms: 250,
        ..manual()
    };
    let expected = observed_turn(&audio, &[320], config.clone());
    assert_eq!(
        expected
            .0
            .iter()
            .map(|pass| pass.samples.len())
            .collect::<Vec<_>>(),
        [4_160, 8_320, 12_480, 16_017]
    );
    assert_eq!(observed_turn(&audio, &[audio.len()], config), expected);
}

#[test]
fn trimming_prompts_and_append_only_events_are_partition_invariant() {
    let audio = word_audio(8_500 * MS + 17);
    let config = SpeechStreamConfig {
        trim_after_ms: 2_500,
        prompt: Some("Mesh".into()),
        tentative: true,
        ..manual()
    };
    let expected = observed_turn(&audio, &[100 * MS], config.clone());
    assert!(
        expected.0.iter().any(|pass| pass.samples[0] > audio[0]),
        "audio really was trimmed"
    );
    assert!(expected.0.iter().any(|pass| {
        pass.options
            .prompt
            .as_deref()
            .is_some_and(|prompt| prompt.contains("word10"))
    }));
    assert_eq!(
        deltas(&expected.1).concat(),
        "word10 word11 word12 word13 word14 word15 word16 word17 word18"
    );
    for parts in [vec![audio.len()], vec![19, 4_973, 809]] {
        assert_eq!(observed_turn(&audio, &parts, config.clone()), expected);
    }
}

#[test]
fn vad_finalization_at_a_due_frame_does_not_run_a_second_pass() {
    let config = SpeechStreamConfig {
        turn_detection: Some(TurnDetection {
            prefix_padding_ms: 0,
            silence_duration_ms: 400,
            ..Default::default()
        }),
        ..SpeechStreamConfig::default()
    };
    let audio = [quiet(300), loud(600), quiet(400), loud(600), quiet(400)].concat();
    let expected = observed_turn(&audio, &[100 * MS], config.clone());
    assert_eq!(expected.0.len(), 2, "only each turn's final pass is due");
    assert_eq!(
        expected
            .0
            .iter()
            .map(|pass| pass.samples.len())
            .collect::<Vec<_>>(),
        [16_000, 16_000]
    );
    assert_eq!(
        expected
            .1
            .iter()
            .filter(|event| matches!(event, SpeechStreamEvent::Completed { .. }))
            .count(),
        2
    );
    for parts in [vec![audio.len()], vec![17, 4_973, 809]] {
        assert_eq!(observed_turn(&audio, &parts, config.clone()), expected);
    }
}

#[test]
fn config_changes_clear_and_commit_keep_their_audio_boundaries() {
    let mut stream = SpeechStream::new(manual());
    let mut model = AudioWords::default();
    push_parts(&mut stream, &mut model, &word_audio(700 * MS), &[700 * MS]);
    assert!(model.seen.is_empty());
    stream.set_config(SpeechStreamConfig {
        step_ms: 400,
        language: Some("fr".into()),
        prompt: Some("changed".into()),
        ..manual()
    });
    push_parts(&mut stream, &mut model, &word_audio(620 * MS), &[620 * MS]);
    assert_eq!(
        model
            .seen
            .iter()
            .map(|pass| pass.samples.len())
            .collect::<Vec<_>>(),
        [720 * MS, 1_120 * MS]
    );
    assert!(
        model
            .seen
            .iter()
            .all(|pass| pass.options.language.as_deref() == Some("fr")
                && pass.options.prompt.as_deref() == Some("changed"))
    );
    stream.set_config(SpeechStreamConfig {
        step_ms: 1_000,
        ..manual()
    });
    push_parts(&mut stream, &mut model, &word_audio(800 * MS), &[800 * MS]);
    assert_eq!(model.seen.last().unwrap().samples.len(), 2_120 * MS);
    let before = model.seen.len();
    stream.commit(&mut |s, o| model.transcribe(s, o)).unwrap();
    assert_eq!(
        model.seen.len(),
        before + 1,
        "commit retains its final pass even at a due boundary"
    );
    push_parts(
        &mut stream,
        &mut model,
        &word_audio(900 * MS + 17),
        &[900 * MS + 17],
    );
    stream.clear();
    assert!(
        stream
            .commit(&mut |s, o| model.transcribe(s, o))
            .unwrap()
            .is_empty()
    );
    let before = model.seen.len();
    push_parts(
        &mut stream,
        &mut model,
        &word_audio(1_000 * MS),
        &[1_000 * MS],
    );
    assert_eq!(model.seen.len(), before + 1);
    assert_eq!(
        model.seen.last().unwrap().samples.len(),
        1_000 * MS,
        "clear drops the partial frame too"
    );
}
