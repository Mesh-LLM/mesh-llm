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
fn tentative_text_shows_what_a_pass_heard_before_it_is_confirmed() {
    let mut script = Script::new(vec![
        vec![(900, " Hello")],
        vec![(1_900, " Hello world")],
        vec![(2_900, " Hello world")],
        vec![(3_000, " Hello world again.")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        tentative: true,
        ..manual()
    });
    let tentative = |events: Vec<SpeechStreamEvent>| {
        events
            .into_iter()
            .filter_map(|event| match event {
                SpeechStreamEvent::Tentative { text, .. } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
    };

    assert_eq!(
        tentative(stream.push(&loud(1_000), &mut transcribe).unwrap()),
        ["Hello"]
    );
    // It reads on from the confirmed text, as a delta does.
    assert_eq!(
        tentative(stream.push(&loud(1_000), &mut transcribe).unwrap()),
        [" world"]
    );
    // All of it is confirmed: an empty one clears what was shown.
    assert_eq!(
        tentative(stream.push(&loud(1_000), &mut transcribe).unwrap()),
        [""]
    );
    assert!(tentative(stream.commit(&mut transcribe).unwrap()).is_empty());
}

#[test]
fn fast_passes_confirm_nothing_until_a_turn_has_two_seconds_of_audio() {
    // Passes this short can agree on a misheard first word; it must not stick.
    let mut script = Script::new(vec![
        vec![(400, " pull")],
        vec![(800, " pull")],
        vec![(1_200, " What pull requests")],
        vec![(1_600, " What pull requests do")],
        vec![(2_000, " What pull requests do I")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        step_ms: 400,
        ..manual()
    });
    let mut confirmed = Vec::new();
    for _ in 0..4 {
        confirmed.extend(deltas(&stream.push(&loud(400), &mut transcribe).unwrap()));
    }
    assert!(confirmed.is_empty(), "{confirmed:?}");
    confirmed.extend(deltas(&stream.push(&loud(400), &mut transcribe).unwrap()));
    assert_eq!(confirmed, ["What pull requests do"]);
}

#[test]
fn tentative_text_is_off_by_default() {
    let mut script = Script::new(vec![vec![(900, " Hello")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    let events = stream.push(&loud(1_000), &mut transcribe).unwrap();
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SpeechStreamEvent::Tentative { .. }))
    );
}

#[test]
fn clear_replaces_reported_tentative_text_with_an_empty_event() {
    let mut script = Script::new(vec![vec![(900, " Dropped words")], vec![(900, " Later")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        tentative: true,
        ..manual()
    });
    stream.push(&loud(1_000), &mut transcribe).unwrap();
    assert!(matches!(
        stream.clear().as_slice(),
        [SpeechStreamEvent::Tentative { turn: 0, text }] if text.is_empty()
    ));
    // Without reported text, clearing reports nothing.
    assert!(stream.clear().is_empty());
    // With tentative off, discarded words report nothing either.
    let mut silent = SpeechStream::new(manual());
    silent.push(&loud(1_000), &mut transcribe).unwrap();
    assert!(silent.clear().is_empty());
}

#[test]
fn switching_tentative_back_on_reports_the_last_text_again() {
    let mut script = Script::new(vec![
        vec![(900, " Hello")],
        vec![(900, " Hello")],
        vec![(900, " Hello")],
        vec![(900, " Hello")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let tentative_cfg = |on: bool| SpeechStreamConfig {
        tentative: on,
        step_ms: 300,
        ..manual()
    };
    let mut stream = SpeechStream::new(tentative_cfg(true));
    let text = |events: Vec<SpeechStreamEvent>| {
        events
            .into_iter()
            .filter_map(|event| match event {
                SpeechStreamEvent::Tentative { text, .. } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        text(stream.push(&loud(300), &mut transcribe).unwrap()),
        ["Hello"]
    );
    // Unchanged text does not repeat.
    assert!(text(stream.push(&loud(300), &mut transcribe).unwrap()).is_empty());
    stream.set_config(tentative_cfg(false));
    assert!(text(stream.push(&loud(300), &mut transcribe).unwrap()).is_empty());
    // A consumer that stopped asking may have cleared its display, so the
    // same text reports again once it is back on.
    stream.set_config(tentative_cfg(true));
    assert_eq!(
        text(stream.push(&loud(300), &mut transcribe).unwrap()),
        ["Hello"]
    );
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
    // The audio is gone but the turn stays open until its words are
    // reported in a `Completed`.
    assert!(stream.buffer.is_empty());
    assert!(stream.has_pending_audio());
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
fn a_pass_hearing_fewer_words_than_confirmed_realigns_instead_of_stalling() {
    let mut script = Script::new(vec![
        vec![(900, " A B C")],
        vec![(1_900, " A B C D")],
        // Whisper merges words: fewer than the three already confirmed.
        vec![(2_900, " A B")],
        vec![(3_900, " A B E")],
        vec![(4_900, " A B E F")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    let mut all = Vec::new();
    for _ in 0..5 {
        all.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
    }
    assert_eq!(deltas(&all), ["A B C", " E"]);
}

#[test]
fn a_shortened_hypothesis_keeps_the_last_confirmed_word_out_of_the_tail() {
    let mut script = Script::new(vec![
        vec![(900, " A B C D")],
        vec![(1_900, " A B C D E")],
        // The beginning was revised away, but D is still the confirmed boundary.
        vec![(2_900, " C d, E")],
        vec![(3_900, " C D E F")],
        vec![(4_900, " C D E F G")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    for expected in [vec![], vec!["A B C D"], vec![], vec![" E"], vec![" F"]] {
        let events = stream.push(&loud(1_000), &mut transcribe).unwrap();
        assert_eq!(deltas(&events), expected);
    }
    let done = stream.commit(&mut transcribe).unwrap();
    assert_eq!(deltas(&done), [" G"]);
    assert_eq!(
        done.last(),
        Some(&SpeechStreamEvent::Completed {
            turn: 0,
            transcript: "A B C D E F G".into(),
        })
    );
}

#[test]
fn silence_without_turn_detection_is_bounded_and_never_transcribed() {
    let config = SpeechStreamConfig {
        max_buffer_ms: 2_000,
        ..manual()
    };
    let mut script = Script::new(vec![vec![(900, " Never")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(config);
    for _ in 0..10 {
        stream.push(&quiet(1_000), &mut transcribe).unwrap();
    }
    assert!(stream.buffer.len() <= 3_000 * MS, "{}", stream.buffer.len());
    assert!(script.seen.is_empty());
}

#[test]
fn turn_detection_can_be_switched_off_mid_turn() {
    let mut script = Script::new(vec![vec![(400, " Hold")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig::default());
    stream.push(&loud(400), &mut transcribe).unwrap();
    stream.set_config(manual());
    // Silence no longer ends the turn; the caller does.
    let events = stream.push(&quiet(1_000), &mut transcribe).unwrap();
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SpeechStreamEvent::Completed { .. }))
    );
    assert!(stream.has_pending_audio());
    let done = stream.commit(&mut transcribe).unwrap();
    assert!(matches!(
        done.last(),
        Some(SpeechStreamEvent::Completed { .. })
    ));
}

#[test]
fn two_turns_inside_one_push_are_reported_in_order() {
    let mut script = Script::new(vec![vec![(400, " One.")], vec![(400, " Two.")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig::default());
    let audio = [loud(400), quiet(600), loud(400), quiet(600)].concat();
    let events = stream.push(&audio, &mut transcribe).unwrap();
    let kinds = events
        .iter()
        .map(|event| match event {
            SpeechStreamEvent::SpeechStarted { turn, .. } => format!("start{turn}"),
            SpeechStreamEvent::SpeechStopped { turn, .. } => format!("stop{turn}"),
            SpeechStreamEvent::Delta { turn, .. } => format!("delta{turn}"),
            SpeechStreamEvent::Completed { turn, transcript } => format!("done{turn}:{transcript}"),
            SpeechStreamEvent::Tentative { turn, .. } => format!("tentative{turn}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            "start0",
            "stop0",
            "delta0",
            "done0:One.",
            "start1",
            "stop1",
            "delta1",
            "done1:Two."
        ]
    );
}

#[test]
fn a_commit_after_an_overlong_reset_still_completes_the_turn() {
    let config = SpeechStreamConfig {
        max_buffer_ms: 2_000,
        ..manual()
    };
    let mut script = Script::new(vec![vec![(2_400, " Never settles")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(config);
    stream.push(&loud(2_500), &mut transcribe).unwrap();
    let done = stream.commit(&mut transcribe).unwrap();
    assert_eq!(
        done.last(),
        Some(&SpeechStreamEvent::Completed {
            turn: 0,
            transcript: "Never settles".into()
        })
    );
}

/// Over many scripted runs, the deltas of each turn always join to exactly
/// its completed transcript, whatever Whisper revises between passes.
#[test]
fn deltas_always_join_to_the_completed_transcript() {
    const WORDS: [&str; 6] = ["alpha", "Beta,", "gamma", "delta.", "eps", "zeta"];
    let mut seed = 0x2545_f491_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..200 {
        let passes = (0..6)
            .map(|_| {
                let count = 1 + (next() % 7) as usize;
                let text = (0..count)
                    .map(|_| WORDS[(next() % WORDS.len() as u64) as usize])
                    .collect::<Vec<_>>()
                    .join(" ");
                vec![(
                    1_000,
                    Box::leak(format!(" {text}").into_boxed_str()) as &'static str,
                )]
            })
            .collect::<Vec<_>>();
        let mut script = Script::new(passes);
        let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
        let mut stream = SpeechStream::new(manual());
        let mut events = Vec::new();
        for _ in 0..5 {
            events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
        }
        events.extend(stream.commit(&mut transcribe).unwrap());
        let Some(SpeechStreamEvent::Completed { transcript, .. }) = events.last() else {
            panic!("the turn completes");
        };
        assert_eq!(&deltas(&events).concat(), transcript);
    }
}

#[test]
fn word_matching_ignores_case_and_punctuation_only() {
    assert!(same_word("Hello,", "hello"));
    assert!(same_word("—", "—"));
    assert!(!same_word("—", "-"));
    assert!(!same_word("their", "there"));
}

/// Streams the clip named by `SKIPPY_ASR_TEST_AUDIO` through the model named
/// by `SKIPPY_ASR_TEST_MODEL` in 100 ms chunks.
#[test]
#[ignore = "needs SKIPPY_ASR_TEST_MODEL and SKIPPY_ASR_TEST_AUDIO"]
fn streams_a_real_clip() {
    let model = std::env::var_os("SKIPPY_ASR_TEST_MODEL").expect("set SKIPPY_ASR_TEST_MODEL");
    let audio = std::env::var_os("SKIPPY_ASR_TEST_AUDIO").expect("set SKIPPY_ASR_TEST_AUDIO");
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

#[test]
fn turning_tentative_off_replaces_reported_text_with_an_empty_event() {
    let mut script = Script::new(vec![
        vec![(300, " Hello")],
        vec![(600, " Hello")],
        vec![(900, " Hello")],
        vec![(1_200, " Hello")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let tentative_cfg = |on: bool| SpeechStreamConfig {
        tentative: on,
        step_ms: 300,
        ..manual()
    };
    let mut stream = SpeechStream::new(tentative_cfg(true));
    let text = |events: Vec<SpeechStreamEvent>| {
        events
            .into_iter()
            .filter_map(|event| match event {
                SpeechStreamEvent::Tentative { text, .. } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        text(stream.push(&loud(300), &mut transcribe).unwrap()),
        ["Hello"]
    );
    // Switching tentative off reports the empty replacement itself, so a
    // consumer stops showing the words immediately.
    assert_eq!(text(stream.set_config(tentative_cfg(false))), [""]);
    assert!(text(stream.push(&loud(300), &mut transcribe).unwrap()).is_empty());
    // Re-enabling still frees the next pass to repeat the text.
    stream.set_config(tentative_cfg(true));
    assert_eq!(
        text(stream.push(&loud(300), &mut transcribe).unwrap()),
        ["Hello"]
    );
}

#[test]
fn leading_silence_does_not_count_toward_the_confirmation_delay() {
    // With turn detection off a turn opens on silence; two agreeing passes
    // after only 800 ms of speech must not confirm a misheard first word.
    let mut script = Script::new(vec![
        vec![(2_800, " Yes")],
        vec![(3_200, " Yes")],
        vec![(3_600, " Yes")],
        vec![(4_400, " Yes")],
        vec![(4_800, " Yes")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        step_ms: 400,
        ..manual()
    });
    stream.push(&quiet(2_500), &mut transcribe).unwrap();
    let mut confirmed = Vec::new();
    for _ in 0..2 {
        confirmed.extend(deltas(&stream.push(&loud(400), &mut transcribe).unwrap()));
    }
    assert!(confirmed.is_empty(), "{confirmed:?}");
    // Two seconds of heard speech confirm the agreed word.
    for _ in 0..3 {
        confirmed.extend(deltas(&stream.push(&loud(400), &mut transcribe).unwrap()));
    }
    assert_eq!(confirmed, ["Yes"]);
}

#[test]
fn a_final_pass_that_hears_nothing_keeps_the_words_already_heard() {
    // Whisper can drop the audio of a short turn's final pass; the words
    // earlier passes heard must still reach the transcript.
    let mut script = Script::new(vec![vec![(400, " Yes")], vec![(800, " Yes")], vec![]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        step_ms: 400,
        ..manual()
    });
    stream.push(&loud(400), &mut transcribe).unwrap();
    stream.push(&loud(400), &mut transcribe).unwrap();
    let done = stream.commit(&mut transcribe).unwrap();
    assert_eq!(deltas(&done), ["Yes"]);
    let transcript = done
        .iter()
        .find_map(|event| match event {
            SpeechStreamEvent::Completed { transcript, .. } => Some(transcript.clone()),
            _ => None,
        })
        .expect("completed event");
    assert_eq!(transcript, "Yes");
}
