use super::*;

#[test]
fn a_revised_prior_phrase_and_leading_comma_do_not_replay_the_last_word() {
    // Captured base.en/400 ms passes91–95. The preceding output omitted the
    // third utterance's "ask not", although the latest hypothesis restored it.
    let canonical = "and so my fellow Americans what you country can do for you ask what you can do for your country and so my fellow Americans";
    let observed = "and so my fellow Americans asked not what your country can do for you ask what you can do for your country and so my fellow Americans";
    let words = |text: &str| {
        text.split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let mut stream = SpeechStream::new(manual());
    stream.confirm(words(canonical), &mut Vec::new());
    assert_eq!(stream.confirmed_prefix.align(&words(observed)), Some(27));
    assert_eq!(stream.confirmed_in_buffer, 25);
    let mut script = Script::new(vec![
        vec![(
            30_000,
            " and so my fellow Americans... asked not! what your country can do for you. ask what you can do for your country. and so my fellow Americans... asked...",
        )],
        vec![(
            30_000,
            " , ask what you can do for your country. ask what you can do for your country. and so my fellow Americans... asked me what you can do for your country.",
        )],
        vec![(
            30_000,
            " ask not! what you country can do for you. ask what you can do for your country. and so my fellow Americans. ask not!",
        )],
        vec![(
            30_000,
            " , ask not! what you country can do for you. ask what you can do for your country. and so my fellow Americans... asked not!",
        )],
        vec![(
            30_000,
            " , ask not! what you country can do for you. ask what you can do for your country. and so my fellow Americans... asked not!",
        )],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut events = Vec::new();
    for _ in 0..5 {
        events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
    }
    events.extend(stream.commit(&mut transcribe).unwrap());
    assert_eq!(deltas(&events).concat(), " asked not!");
}

#[test]
fn a_wholly_substituted_prefix_keeps_agreed_new_speech_streaming() {
    let mut script = Script::new(vec![
        vec![(900, " Alpha")],
        vec![(1_900, " Alpha")],
        vec![(2_900, " Beta fresh words")],
        vec![(3_900, " Beta fresh words")],
        vec![(4_900, " Beta fresh words today")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    let mut events = Vec::new();
    for expected in [vec![], vec!["Alpha"], vec![], vec![" fresh words"], vec![]] {
        let batch = stream.push(&loud(1_000), &mut transcribe).unwrap();
        assert_eq!(deltas(&batch), expected);
        events.extend(batch);
    }
    let done = stream.commit(&mut transcribe).unwrap();
    assert_eq!(deltas(&done), [" today"]);
    assert_eq!(
        done.last(),
        Some(&SpeechStreamEvent::Completed {
            turn: 0,
            transcript: "Alpha fresh words today".into(),
        })
    );
    events.extend(done);
    assert_eq!(deltas(&events).concat(), "Alpha fresh words today");
    assert!(!stream.has_pending_audio());
    assert_eq!(stream.current_turn(), 1);
}

#[test]
fn a_final_wholly_substituted_prefix_keeps_new_speech_at_commit() {
    let mut script = Script::new(vec![
        vec![(900, " Alpha")],
        vec![(1_900, " Alpha")],
        vec![(2_000, " Beta final words")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    let mut events = stream.push(&loud(1_000), &mut transcribe).unwrap();
    events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
    assert_eq!(deltas(&events), ["Alpha"]);
    let done = stream.commit(&mut transcribe).unwrap();
    assert_eq!(deltas(&done), [" final words"]);
    assert_eq!(
        done.last(),
        Some(&SpeechStreamEvent::Completed {
            turn: 0,
            transcript: "Alpha final words".into(),
        })
    );
    events.extend(done);
    assert_eq!(deltas(&events).concat(), "Alpha final words");
    assert!(!stream.has_pending_audio());
    assert_eq!(stream.current_turn(), 1);
}

#[test]
fn an_unsupported_prefix_does_not_replay_a_restored_omitted_owner() {
    // Y revises the retained prefix before the already emitted B returns.
    // The cheapest alignment has no supported match, although B occurs later.
    // An observed-length fallback would expose Y B and emit B a second time.
    let mut script = Script::new(vec![
        vec![(900, " A B")],
        vec![(1_900, " A B")],
        vec![(2_900, " A")],
        vec![(3_900, " X Y B")],
        vec![(4_900, " X Y B")],
        // Keep the unsupported hypothesis through the final commit pass.
        vec![(5_900, " X Y B")],
    ]);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
    let mut stream = SpeechStream::new(manual());
    let mut events = Vec::new();
    for (pass, expected) in [vec![], vec!["A B"], vec![], vec![], vec![], vec![]]
        .into_iter()
        .enumerate()
    {
        let batch = stream.push(&loud(1_000), &mut transcribe).unwrap();
        assert_eq!(deltas(&batch), expected);
        if pass == 2 {
            assert_eq!(stream.confirmed_in_buffer, 2);
            assert_eq!(stream.confirmed_prefix.observed_len(), 1);
        }
        events.extend(batch);
    }
    let done = stream.commit(&mut transcribe).unwrap();
    assert!(deltas(&done).is_empty());
    assert_eq!(
        done.last(),
        Some(&SpeechStreamEvent::Completed {
            turn: 0,
            transcript: "A B".into(),
        })
    );
    events.extend(done);
    assert_eq!(deltas(&events).concat(), "A B");
}

#[test]
fn wholly_rewritten_prefixes_without_new_speech_do_not_append_words() {
    for rewrite in ["Gamma Delta", "Gamma"] {
        let mut script = Script::new(vec![
            vec![(900, " Alpha Beta")],
            vec![(1_900, " Alpha Beta")],
            vec![(2_900, rewrite)],
            vec![(3_900, rewrite)],
        ]);
        let mut transcribe = |s: &[f32], o: &TranscriptionOptions| script.transcribe(s, o);
        let mut stream = SpeechStream::new(manual());
        let mut events = Vec::new();
        for _ in 0..4 {
            events.extend(stream.push(&loud(1_000), &mut transcribe).unwrap());
        }
        let done = stream.commit(&mut transcribe).unwrap();
        assert!(deltas(&done).is_empty(), "rewrite: {rewrite}");
        assert_eq!(
            done.last(),
            Some(&SpeechStreamEvent::Completed {
                turn: 0,
                transcript: "Alpha Beta".into(),
            }),
            "rewrite: {rewrite}",
        );
        events.extend(done);
        assert_eq!(deltas(&events).concat(), "Alpha Beta", "rewrite: {rewrite}");
    }
}
