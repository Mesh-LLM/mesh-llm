//! What the recorder keeps at its byte budgets: a turn a client can still
//! challenge keeps what opens it, and new work is refused first.

use super::*;

/// Commits a turn of three positions of `seq` on cells from `10 * seq`.
fn commit_turn(recorder: &Recorder, seq: i32, id: &str) -> anyhow::Result<StageCommitment> {
    let columns: Vec<(i32, i32)> = (0..3).map(|p| (seq, p)).collect();
    let cells: Vec<u32> = (0..3).map(|p| (10 * seq + p) as u32).collect();
    record(recorder, ubatch(&columns, &cells, true));
    recorder.commit(id, seq, TurnId::default(), 0..3)
}

// Past the window, a turn inside its challenge window keeps its record and
// the new turn gets no receipt; a turn past its window makes room.
#[test]
fn the_window_drops_only_turns_past_their_challenge_window() {
    let measure = Recorder::new(1, STAGES[1].clone(), "cpu");
    commit_turn(&measure, 1, "one").unwrap();
    let turn = measure.lock().committed_bytes;
    let recorder =
        Recorder::new(1, STAGES[1].clone(), "cpu").with_window_bytes(2 * turn + turn / 2);
    commit_turn(&recorder, 1, "victim").unwrap();
    commit_turn(&recorder, 2, "second").unwrap();
    let error = commit_turn(&recorder, 3, "third").unwrap_err();
    assert!(
        format!("{error:#}").contains("gets no receipt"),
        "{error:#}"
    );
    assert!(recorder.holds_turn("victim") && recorder.holds_turn("second"));
    assert!(!recorder.holds_turn("third"));
    // the victim's window passed (here: nobody can challenge it any more)
    recorder.release_pin("victim");
    commit_turn(&recorder, 4, "fourth").unwrap();
    assert!(!recorder.holds_turn("victim"));
    assert!(recorder.holds_turn("second") && recorder.holds_turn("fourth"));
}

// Past the lineage budget, the ubatch that wrote a prefix cell of a turn a
// client can still challenge is kept, and newer traffic loses its own: a
// later turn that reads it as its prefix gets no receipt.
#[test]
fn the_lineage_of_a_turn_inside_its_window_outlasts_the_budget() {
    // a stage after the first keeps 4 bytes of input per token
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu").with_lineage_input_bytes(12);
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.commit("r0", 1, TurnId::default(), 0..2).unwrap();
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    recorder.commit("r1", 1, TurnId::default(), 2..3).unwrap();
    // another request's prompt takes the budget past its end
    record(&recorder, ubatch(&[(2, 0), (2, 1)], &[10, 11], true));
    let replayer = FakeReplayer::for_stage(1);
    recorder
        .open(&challenge(&[(2, &[0, 2])]), &replayer)
        .expect("the prefix position opens");

    // the newer request's turn reads cells whose ubatch the stage dropped
    recorder.memory_op(MemoryOp::SeqCp, 2, 3, -1, -1);
    record(&recorder, ubatch(&[(3, 2)], &[12], true));
    let error = recorder
        .commit("r3", 3, TurnId::default(), 2..3)
        .unwrap_err();
    assert!(format!("{error:#}").contains("no longer held"), "{error:#}");

    // once no client can challenge the turn, its lineage goes first
    recorder.release_pin("r1");
    record(&recorder, ubatch(&[(4, 0), (4, 1)], &[20, 21], true));
    let error = recorder
        .open(&challenge(&[(2, &[1, 2])]), &FakeReplayer::for_stage(1))
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("no longer holds"),
        "{error:#}"
    );
}

/// Whether the turn `id` is still pinned.
fn pinned(recorder: &Recorder, id: &str, now: Instant) -> bool {
    let state = recorder.lock();
    turn(&state.committed, id).is_some_and(|turn| turn.pin.holds(now))
}

// A client that sends bound long turns cannot keep the cache pinned, and no
// turn loses its receipt for it: past its key's share the oldest pin of the
// key gives way, past the share of all keys the oldest pin of any key.
#[test]
fn past_a_share_the_oldest_pins_give_way_and_every_turn_keeps_its_receipt() {
    const A: [u8; 32] = [1; 32];
    const B: [u8; 32] = [2; 32];
    const C: [u8; 32] = [3; 32];
    let now = Instant::now();
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    // a stream of 12 cells: a key pins 6 at most, every key together 9
    recorder.set_pin_budget(12);
    let bound = |seq: i32, id: &str, key: [u8; 32]| {
        commit_turn(&recorder, seq, id).unwrap();
        recorder.admit_pin(id, key, now).unwrap();
    };
    let pins =
        |ids: &[&str]| -> Vec<bool> { ids.iter().map(|id| pinned(&recorder, id, now)).collect() };
    bound(1, "a1", A);
    bound(2, "a2", A);
    assert_eq!(pins(&["a1", "a2"]), [true, true]);
    // a third turn of the key: its oldest pin gives way
    bound(3, "a3", A);
    assert_eq!(pins(&["a1", "a2", "a3"]), [false, true, true]);
    bound(4, "b1", B);
    assert_eq!(recorder.pinned_cells(now).len(), 9);
    // a turn of a third key: the oldest pin of any key gives way
    bound(5, "c1", C);
    assert_eq!(pins(&["a2", "a3", "b1", "c1"]), [false, true, true, true]);
    assert_eq!(recorder.pinned_cells(now).len(), 9);
    // every turn is still committed, so each has its receipt and opens
    // until a ubatch writes its cells
    assert!(
        ["a1", "a2", "a3", "b1", "c1"]
            .iter()
            .all(|id| recorder.holds_turn(id))
    );
}

// A turn that a client has opened waits for its rows request, which will
// come; an unchallenged turn most likely has no challenge coming. So the
// unchallenged pins give way first, whatever their age.
#[test]
fn a_turn_that_waits_for_its_rows_request_keeps_its_pin_longest() {
    const A: [u8; 32] = [1; 32];
    let now = Instant::now();
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    recorder.set_pin_budget(12);
    for (seq, id) in [(1, "opened"), (2, "idle")] {
        commit_turn(&recorder, seq, id).unwrap();
        recorder.admit_pin(id, A, now).unwrap();
    }
    {
        let mut state = recorder.lock();
        let turn = turn_mut(&mut state.committed, "opened").unwrap();
        turn.pin.opened(now, true);
    }
    commit_turn(&recorder, 3, "new").unwrap();
    recorder.admit_pin("new", A, now).unwrap();
    assert!(pinned(&recorder, "opened", now) && pinned(&recorder, "new", now));
    assert!(!pinned(&recorder, "idle", now));
}

// A turn that is past a share by itself still gets its pin and its
// receipt: nothing else it could give way to.
#[test]
fn a_turn_past_a_share_alone_is_pinned_alone() {
    const A: [u8; 32] = [1; 32];
    const B: [u8; 32] = [2; 32];
    let now = Instant::now();
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    // a turn of 3 cells is past both shares of a stream of 3
    recorder.set_pin_budget(3);
    commit_turn(&recorder, 1, "first").unwrap();
    recorder.admit_pin("first", A, now).unwrap();
    assert!(pinned(&recorder, "first", now));
    commit_turn(&recorder, 2, "second").unwrap();
    recorder.admit_pin("second", B, now).unwrap();
    assert!(pinned(&recorder, "second", now) && !pinned(&recorder, "first", now));
    assert!(recorder.holds_turn("first") && recorder.holds_turn("second"));
}

// A turn's log counts against the window as its ubatches arrive: a long
// prefill that outgrows it is shed before it commits, and gets no receipt,
// rather than holding past the window until the commit's check.
#[test]
fn a_live_log_past_the_window_is_shed_before_its_commit() {
    // a later stage's ubatch keeps its input rows: 4 KiB a token here
    let wide = |columns: &[(i32, i32)], cells: &[u32], last: bool| {
        let mut record = ubatch(columns, cells, last);
        record.input = Some(StageInput {
            rows: vec![7; columns.len() * 4096],
            row_bytes: 4096,
            ggml_type: 0,
        });
        record
    };
    let measure = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&measure, wide(&[(1, 0)], &[0], true));
    measure.commit("one", 1, TurnId::default(), 0..1).unwrap();
    let one = measure.lock().committed_bytes;
    let window = one + 2 * 3 * 4096;

    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu").with_window_bytes(window);
    for chunk in 0..4 {
        let columns: Vec<(i32, i32)> = (0..3).map(|p| (1, 3 * chunk + p)).collect();
        let cells: Vec<u32> = (0..3).map(|p| (3 * chunk + p) as u32).collect();
        record(&recorder, wide(&columns, &cells, chunk == 3));
        let held = recorder.lock().held_bytes();
        assert!(held <= window, "ubatch {chunk}: {held} of {window}");
    }
    {
        // a shed log keeps nothing per position, also for the ubatches
        // after it was shed, and its prefix only as a stand-in
        let state = recorder.lock();
        let log = &state.seqs[&1].live;
        assert!(log.shed);
        assert!(log.leaves.is_empty() && log.events.is_empty());
        assert!(matches!(log.prefix, Some(Err(_))));
        assert_eq!(log.live_bytes(), 0);
    }
    let error = recorder
        .commit("long", 1, TurnId::default(), 0..12)
        .unwrap_err();
    assert!(format!("{error:#}").contains("outgrew"), "{error:#}");

    // the sequence's next turn records and commits again
    recorder.begin_turn(1, TurnId(1));
    record(&recorder, wide(&[(1, 12)], &[12], true));
    recorder.commit("next", 1, TurnId(1), 12..13).unwrap();
    assert!(recorder.holds_turn("next"));
}

// The window counts the hashes a live log keeps per position and its
// prefix, not only its ubatches: a turn of decode tokens, whose ubatches
// carry no input rows, still counts what it holds.
#[test]
fn a_live_log_counts_its_positions_and_prefix() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    let state = recorder.lock();
    let log = &state.seqs[&1].live;
    let positions: usize = log.leaves.values().map(leaves::PositionLeaves::bytes).sum();
    assert_eq!(log.leaves.len(), 2);
    assert!(positions > 0);
    assert_eq!(
        log.live_bytes(),
        log.event_bytes + positions + log.prefix_bytes()
    );
    assert_eq!(log.live_bytes(), log.bytes());
}
