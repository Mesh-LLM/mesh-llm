//! A commit takes the log of the turn it names, and no other's.

use super::*;

const SEQ: i32 = 1;

/// A ubatch of `SEQ` at `positions`, whose tokens `request` tells apart
/// from another request's at the same positions.
fn ubatch_of(request: i32, positions: std::ops::Range<i32>) -> UbatchRecord {
    let mut record = ubatch(
        &positions.clone().map(|p| (SEQ, p)).collect::<Vec<_>>(),
        &positions.map(|p| p as u32).collect::<Vec<_>>(),
        true,
    );
    record.token.iter_mut().for_each(|t| *t += 10_000 * request);
    record
}

/// The root a stage commits for a turn it alone recorded.
fn root_alone(request: i32, positions: std::ops::Range<i32>) -> String {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let end = positions.end as u32;
    record(&recorder, ubatch_of(request, positions));
    recorder
        .commit("alone", SEQ, TurnId::default(), 0..end)
        .unwrap()
        .root
}

// Request A's lane is reset and handed to request B before A commits. A's
// commit names A's turn, so it gets A's leaves, although B's log is the
// sequence's live one and covers the same positions.
#[test]
fn a_commit_gets_its_own_turn_after_the_sequence_was_reused() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let (a, b) = (TurnId(1), TurnId(2));
    recorder.begin_turn(SEQ, a);
    record(&recorder, ubatch_of(1, 0..3));
    recorder.memory_op(MemoryOp::SeqRm, SEQ, -1, -1, -1);
    recorder.begin_turn(SEQ, b);
    record(&recorder, ubatch_of(2, 0..4));

    assert_eq!(recorder.recorded_range(SEQ, a), Some(0..3));
    assert_eq!(recorder.recorded_range(SEQ, b), Some(0..4));
    let committed = recorder.commit("a", SEQ, a, 0..3).unwrap();
    assert_eq!(committed.root, root_alone(1, 0..3));
    assert_ne!(committed.root, root_alone(2, 0..3));
    // B's turn is untouched, and commits its own leaves
    let committed = recorder.commit("b", SEQ, b, 0..4).unwrap();
    assert_eq!(committed.root, root_alone(2, 0..4));
    // A's log was used up by its commit
    assert_eq!(recorder.recorded_range(SEQ, a), None);
    assert!(recorder.commit("a2", SEQ, a, 0..3).is_err());
}

// The same when the lane was not reset whole in between: the session that
// takes the sequence starts a new log.
#[test]
fn a_session_taking_a_sequence_starts_a_new_log() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    recorder.begin_turn(SEQ, TurnId(1));
    record(&recorder, ubatch_of(1, 0..3));
    recorder.begin_turn(SEQ, TurnId(2));
    recorder.memory_op(MemoryOp::SeqRm, SEQ, -1, -1, -1);
    record(&recorder, ubatch_of(2, 0..3));
    let committed = recorder.commit("b", SEQ, TurnId(2), 0..3).unwrap();
    assert_eq!(committed.root, root_alone(2, 0..3));
    let committed = recorder.commit("a", SEQ, TurnId(1), 0..3).unwrap();
    assert_eq!(committed.root, root_alone(1, 0..3));
}

// A commit that names a turn the sequence holds no log of is refused, and
// takes nothing from the turn the sequence records.
#[test]
fn a_commit_of_another_turn_is_refused() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    recorder.begin_turn(SEQ, TurnId(2));
    record(&recorder, ubatch_of(2, 0..4));
    for stale in [TurnId::default(), TurnId(1), TurnId(3)] {
        let error = recorder.commit("a", SEQ, stale, 0..3).unwrap_err();
        assert!(format!("{error:#}").contains("no log of turn"), "{error:#}");
        assert_eq!(recorder.recorded_range(SEQ, stale), None);
    }
    let committed = recorder.commit("b", SEQ, TurnId(2), 0..4).unwrap();
    assert_eq!(committed.root, root_alone(2, 0..4));
}

// The log of a turn that ended uncommitted waits for its commit only for a
// while, and counts toward the recorder's window meanwhile.
#[test]
fn an_uncommitted_log_expires_and_is_counted() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    recorder.begin_turn(SEQ, TurnId(1));
    record(&recorder, ubatch_of(1, 0..3));
    recorder.begin_turn(SEQ, TurnId(2));
    let held = recorder.lock().retired_bytes;
    assert!(held > 0, "a retired log holds its events and leaves");
    let later = Instant::now() + RETIRED_LOG_TTL;
    recorder.lock().drop_stale_logs(later);
    assert_eq!(recorder.lock().retired_bytes, 0);
    assert!(recorder.commit("a", SEQ, TurnId(1), 0..3).is_err());

    // past the window, uncommitted logs go before committed turns
    record(&recorder, ubatch_of(2, 0..3));
    recorder.commit("measured", SEQ, TurnId(2), 0..3).unwrap();
    let committed = recorder.lock().committed_bytes;
    let window = committed + held + held / 2;
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu").with_window_bytes(window);
    recorder.begin_turn(SEQ, TurnId(1));
    record(&recorder, ubatch_of(1, 0..3));
    recorder.commit("kept", SEQ, TurnId(1), 0..3).unwrap();
    for turn in 2..6 {
        recorder.begin_turn(SEQ + turn as i32, TurnId(turn));
        record_on(&recorder, &Stage::new(1), {
            let mut record = ubatch_of(1, 0..3);
            record.seq.fill(SEQ + turn as i32);
            record.cell.iter_mut().for_each(|c| *c += 10 * turn as u32);
            record
        });
        recorder.begin_turn(SEQ + turn as i32, TurnId(100 + turn));
    }
    let state = recorder.lock();
    assert_eq!(state.retired_bytes, held, "one uncommitted log fits");
    assert_eq!(state.committed.len(), 1, "the committed turn stayed");
}
