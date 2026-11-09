use super::*;

/// The newest confirmed segment remains the transcription context.
#[test]
fn the_newest_confirmed_segment_remains_as_transcription_context() {
    let mut script = Script::new(vec![
        vec![(1_000, " A B")],
        vec![(2_000, " A B")],
        vec![(3_000, " A B")],
        vec![(3_000, " A B"), (4_000, " C")],
        vec![(1_900, " C D")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        trim_after_ms: 2_500,
        max_buffer_ms: 8_000,
        ..manual()
    });
    let mut events = Vec::new();
    for pass in 1..=5 {
        events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
        if pass == 3 {
            assert_eq!(stream.buffer.len(), 3_000 * MS);
            assert_eq!(stream.buffer_start, 0);
            assert_eq!(stream.confirmed_in_buffer, 2);
        } else if pass == 4 {
            assert_eq!(stream.buffer.len(), 1_000 * MS);
            assert_eq!(stream.buffer_start, 3_000 * MS as u64);
            assert_eq!(stream.confirmed_in_buffer, 0);
        }
    }
    events.extend(stream.commit(&mut transcribe).unwrap());
    assert_eq!(deltas(&events).concat(), "A B C D");
    assert_eq!(script.seen[3], (4_000 * MS, None));
    assert_eq!(script.seen[4], (2_000 * MS, Some("A B".into())));
}

/// A future-only later segment cannot leave an empty context window.
#[test]
fn a_future_only_later_segment_cannot_leave_an_empty_context_window() {
    let mut script = Script::new(vec![vec![(2_000, " A"), (5_000, " B")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        trim_after_ms: 1_500,
        ..manual()
    });
    stream.push(&loud(1_000), &mut transcribe).unwrap();
    let events = stream.push(&loud(1_000), &mut transcribe).unwrap();
    assert_eq!(deltas(&events).concat(), "A B");
    assert_eq!(stream.buffer.len(), 2_000 * MS);
    assert_eq!(stream.buffer_start, 0);
    assert_eq!(stream.confirmed_in_buffer, 2);
    // Once real audio exists after A, it can be removed while B stays owned.
    stream.push(&loud(1_000), &mut transcribe).unwrap();
    assert_eq!(stream.buffer.len(), 1_000 * MS);
    assert_eq!(stream.buffer_start, 2_000 * MS as u64);
    assert_eq!(stream.confirmed_in_buffer, 1);
}

/// A permanently single segment still obeys the hard window bound.
#[test]
fn a_permanently_single_segment_still_obeys_the_hard_window_bound() {
    // The settling pass and the final pass hear the retained window, which
    // holds no new words.
    let mut script = Script::new(vec![vec![(900, " A B")], vec![(900, " A B")], vec![]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        trim_after_ms: 1_500,
        max_buffer_ms: 3_500,
        ..manual()
    });
    let mut events = Vec::new();
    for pass in 1..=4 {
        events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
        if pass == 3 {
            assert_eq!(stream.buffer.len(), 3_000 * MS);
            assert_eq!(stream.confirmed_in_buffer, 2);
        }
    }
    // The hard bound still settles the window, confirms `A B`, and leaves
    // the turn open with the newest window retained for the next pass. The
    // segment ends after the cut, so its audio stays and its words stay owned.
    assert_eq!(stream.buffer.len(), 3_500 * MS);
    assert_eq!(stream.buffer_start, 500 * MS as u64);
    assert_eq!(stream.confirmed_in_buffer, 2);
    // The last pass heard nothing, so the owners are omitted, not observed.
    assert_eq!(stream.confirmed_prefix.observed_len(), 0);
    events.extend(stream.commit(&mut transcribe).unwrap());
    assert_eq!(deltas(&events).concat(), "A B");
}

/// Words whose audio survives an overflow are not confirmed a second time.
#[test]
fn an_overflow_does_not_reconfirm_words_in_the_retained_audio() {
    // The overflow at 4 s keeps the audio from 0.5 s, so the next window
    // hears `A B` again, now ending at 0.4 s. Its words are still owned.
    let mut script = Script::new(vec![
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(400, " A B")],
        vec![],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        trim_after_ms: 1_500,
        max_buffer_ms: 3_500,
        ..manual()
    });
    let mut events = Vec::new();
    for _ in 0..5 {
        events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
    }
    events.extend(stream.commit(&mut transcribe).unwrap());
    assert_eq!(deltas(&events).concat(), "A B");
}

/// A step beyond the window still runs a pass before audio leaves it.
#[test]
fn a_step_beyond_the_window_is_clamped_to_it() {
    let mut script = Script::new(vec![vec![(900, " A B")]]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        step_ms: 60_000,
        ..manual()
    });
    stream.push(&loud(31_000), &mut transcribe).unwrap();
    assert_eq!(script.seen.len(), 1);
    assert_eq!(script.seen[0].0, WHISPER_WINDOW_MS as usize * MS);
}

/// A slow step keeps every decoder input inside the whisper window.
#[test]
fn a_slow_step_keeps_every_decoder_input_inside_the_whisper_window() {
    // 36 s of audio at a 4 s step crosses the transcriber's 30 s window:
    // the settling passes must drain to the cap. Each second carries its
    // own constant level, so a decoder input's first sample names the
    // second where its window starts. The single segment answers from its
    // audio until the drain passes, after which the window holds no new
    // words.
    let mut script = Script::new(vec![
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![(900, " A B")],
        vec![],
    ]);
    let mut decoder_inputs: Vec<(usize, f32)> = Vec::new();
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| {
        decoder_inputs.push((s.len(), s.first().copied().unwrap_or(0.0)));
        script.transcribe(s, o)
    };
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        step_ms: 4_000,
        ..manual()
    });
    let mut events = Vec::new();
    for second in 0..36u32 {
        let audio = vec![0.01 * (second as f32 + 1.0); 1_000 * MS];
        events.extend(stream.push(&audio, &mut transcribe).unwrap());
    }
    events.extend(stream.commit(&mut transcribe).unwrap());
    // Every pass (and the commit) must hear at most the transcriber's 30 s
    // window, and a settling pass must actually reach the cap rather than
    // trim early.
    for (samples, _) in &script.seen {
        assert!(
            *samples <= WHISPER_WINDOW_MS as usize * MS,
            "decoder input {samples} exceeds the whisper window"
        );
    }
    assert!(
        script
            .seen
            .iter()
            .any(|(samples, _)| *samples == WHISPER_WINDOW_MS as usize * MS),
        "a pass must be capped exactly at the whisper window: {:?}",
        script.seen
    );
    // The capped inputs must be the newest 30 s, not merely 30 s long: the
    // 32 s pass drains to the window starting at second 3, and the 36 s
    // pass settles on the window starting at second 7.
    let capped: Vec<f32> = decoder_inputs
        .iter()
        .filter(|(samples, _)| *samples == WHISPER_WINDOW_MS as usize * MS)
        .map(|(_, first)| *first)
        .collect();
    assert_eq!(
        capped,
        vec![0.01 * 3.0, 0.01 * 7.0],
        "capped inputs must start at the newest-window second: {:?}",
        decoder_inputs
    );
    assert_eq!(deltas(&events).concat(), "A B");
}
