use super::*;

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

#[test]
fn a_permanently_single_segment_still_obeys_the_hard_window_bound() {
    let mut script = Script::new(vec![vec![(900, " A B")]]);
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
    assert!(stream.buffer.is_empty());
    assert_eq!(stream.buffer_start, 4_000 * MS as u64);
    assert_eq!(stream.confirmed_in_buffer, 0);
    assert_eq!(stream.confirmed_prefix.observed_len(), 0);
    events.extend(stream.commit(&mut transcribe).unwrap());
    assert_eq!(deltas(&events).concat(), "A B");
}
