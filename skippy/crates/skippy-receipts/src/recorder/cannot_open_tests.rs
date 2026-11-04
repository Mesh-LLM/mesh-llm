//! A stage says so, with a reason, when it can no longer open a turn it
//! committed, and loses a turn's cells only to what can have changed them.

use super::*;

/// The [`CannotOpen`] an opening failed with.
fn cannot_open(error: &anyhow::Error) -> &CannotOpen {
    error
        .downcast_ref::<CannotOpen>()
        .unwrap_or_else(|| panic!("not a cannot-open answer: {error:#}"))
}

/// Sequences 1 and 2 each commit a turn of two positions after a prefix of
/// three. Sequence 1 goes on holding its cells; sequence 2 ends, so its
/// cells are free and keep their rows. Returns the seed of each turn's
/// replay.
fn two_turns(recorder: &Recorder) -> [ReplaySeed; 2] {
    [1, 2].map(|seq| {
        let cells = |positions: std::ops::Range<i32>| -> Vec<u32> {
            positions.map(|p| (seq * 10 + p) as u32).collect()
        };
        let columns = |positions: std::ops::Range<i32>| -> Vec<(i32, i32)> {
            positions.map(|p| (seq, p)).collect()
        };
        record(recorder, ubatch(&columns(0..3), &cells(0..3), true));
        let earlier = format!("earlier-{seq}");
        recorder
            .commit(&earlier, seq, TurnId::default(), 0..3)
            .unwrap();
        record(recorder, ubatch(&columns(3..5), &cells(3..5), true));
        let id = format!("r{seq}");
        recorder.commit(&id, seq, TurnId::default(), 3..5).unwrap();
        if seq == 2 {
            recorder.memory_op(MemoryOp::SeqRm, seq, -1, -1, -1);
        }
        let replayer = FakeReplayer::for_stage(1);
        let mut challenge = challenge(&[(2, &[4])]);
        challenge.receipt_id = id;
        recorder.open(&challenge, &replayer).unwrap();
        let seed = replayer.seed.borrow().clone().expect("a seeded replay");
        recorder.check_seed(&seed).unwrap();
        seed
    })
}

// A ubatch the recorder could not follow may have written any cell no
// sequence held, and no other: a turn whose cells are still held stays
// openable, and one whose cells were free is refused with the reason.
#[test]
fn an_unreceiptable_ubatch_costs_only_the_cells_no_sequence_holds() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let [held, freed] = two_turns(&recorder);
    recorder.unreceiptable("two lanes in one recurrent ubatch");
    recorder.check_seed(&held).unwrap();
    let error = recorder.check_seed(&freed).unwrap_err();
    let cannot = cannot_open(&error);
    assert_eq!(cannot.reason, CannotOpenReason::CellsUnknown);
    assert_eq!(cannot.stage_index, 1);
}

// An image request beside a receipted turn: the native recorder reports a
// ubatch of embeddings it cannot receipt, and names neither its sequences
// nor its cells. Taken as the recorder losing track of the stage, it costs
// every live turn its receipt; taken as one ubatch, it costs only the turns
// whose logs show it computed something they lack.
#[test]
fn an_unrecorded_ubatch_costs_only_the_turns_it_computed() {
    let lost = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(
        &lost,
        ubatch(&[(1, 0), (1, 1), (1, 2)], &[10, 11, 12], true),
    );
    lost.unreceiptable("a ubatch of embeddings cannot be receipted");
    record(&lost, ubatch(&[(1, 3)], &[13], true));
    let error = lost.commit("r1", 1, TurnId::default(), 0..4).unwrap_err();
    assert!(
        format!("{error:#}").contains("cannot be replayed"),
        "{error:#}"
    );

    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(
        &recorder,
        ubatch(&[(1, 0), (1, 1), (1, 2)], &[10, 11, 12], true),
    );
    record(
        &recorder,
        ubatch(&[(3, 0), (3, 1), (3, 2)], &[30, 31, 32], true),
    );
    // sequence 3's image at positions 3 and 4
    recorder.unrecorded_ubatch("a ubatch of embeddings cannot be receipted");
    record(&recorder, ubatch(&[(1, 3), (3, 5)], &[13, 35], true));
    // sequence 1 recorded every position of its turn: it keeps its receipt
    recorder.commit("r1", 1, TurnId::default(), 0..4).unwrap();
    // sequence 3 lacks what the ubatch computed, whichever positions it names
    for positions in [0..6, 5..6] {
        let error = recorder
            .commit("r3", 3, TurnId::default(), positions.clone())
            .unwrap_err();
        assert!(!recorder.holds_turn("r3"), "{positions:?}: {error:#}");
    }
}

// The cache writes a free pinned cell only when nothing else fits, so a
// ubatch the recorder could not record leaves a pinned turn openable once
// its rows are read and match; a turn that is not pinned, or whose rows no
// longer match, is answered with the reason.
#[test]
fn an_unrecorded_ubatch_leaves_a_pinned_turn_openable() {
    let freed_turn = |recorder: &Recorder| {
        record(
            recorder,
            ubatch(&[(2, 0), (2, 1), (2, 2)], &[20, 21, 22], true),
        );
        recorder
            .commit("earlier", 2, TurnId::default(), 0..3)
            .unwrap();
        record(recorder, ubatch(&[(2, 3), (2, 4)], &[23, 24], true));
        recorder.commit("r2", 2, TurnId::default(), 3..5).unwrap();
        recorder.memory_op(MemoryOp::SeqRm, 2, -1, -1, -1);
        let mut challenge = challenge(&[(2, &[4])]);
        challenge.receipt_id = "r2".into();
        challenge
    };
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let challenge = freed_turn(&recorder);
    assert_eq!(recorder.pinned_cells(Instant::now()).len(), 5);
    recorder.unrecorded_ubatch("a ubatch of embeddings cannot be receipted");
    let replayer = FakeReplayer::for_stage(1);
    recorder.open(&challenge, &replayer).unwrap();
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    recorder.check_seed(&seed).unwrap();

    // rows another write left in a pinned cell
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let challenge = freed_turn(&recorder);
    recorder.pinned_cells(Instant::now());
    recorder.unrecorded_ubatch("a ubatch of embeddings cannot be receipted");
    let replayer = FakeReplayer {
        stale_rows: true,
        ..FakeReplayer::for_stage(1)
    };
    let error = recorder.open(&challenge, &replayer).unwrap_err();
    let cannot = cannot_open(&error);
    assert_eq!(cannot.reason, CannotOpenReason::CellsOverwritten);
    assert_eq!(cannot.receipt_id, "r2");

    // nothing pinned: the free cells are forgotten
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let challenge = freed_turn(&recorder);
    recorder.unrecorded_ubatch("a ubatch of embeddings cannot be receipted");
    let replayer = FakeReplayer::for_stage(1);
    recorder.open(&challenge, &replayer).unwrap();
    // the stage checks the seed with the stage idle, before it copies it
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    let error = recorder.check_seed(&seed).unwrap_err();
    assert_eq!(cannot_open(&error).reason, CannotOpenReason::CellsUnknown);
}

// What happens to one sequence's cells leaves another's turn openable.
#[test]
fn a_change_of_one_sequence_leaves_the_others_openable() {
    for op in [MemoryOp::SeqShift, MemoryOp::StateRead] {
        let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
        let [held, _] = two_turns(&recorder);
        recorder.memory_op(op, 5, -1, -1, -1);
        recorder.check_seed(&held).unwrap();
        recorder.memory_op(op, 1, -1, 0, -1);
        let error = recorder.check_seed(&held).unwrap_err();
        assert_eq!(cannot_open(&error).reason, CannotOpenReason::CellsUnknown);
    }
    // a shift of sequence 1 does not reach the free cells of sequence 2's turn
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let [_, freed] = two_turns(&recorder);
    recorder.memory_op(MemoryOp::SeqShift, 1, -1, 0, -1);
    recorder.check_seed(&freed).unwrap();
}

// A cell a later request wrote, a turn that left the window and a dropped
// recurrent state are each answered as such.
#[test]
fn an_opening_names_why_the_stage_cannot_make_it() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let [_, freed] = two_turns(&recorder);
    record(&recorder, ubatch(&[(9, 0)], &[21], true));
    let error = recorder.check_seed(&freed).unwrap_err();
    assert_eq!(
        cannot_open(&error).reason,
        CannotOpenReason::CellsOverwritten
    );

    let replayer = FakeReplayer::for_stage(1);
    let mut gone = challenge(&[(2, &[4])]);
    gone.receipt_id = "never-committed".into();
    let error = recorder.open(&gone, &replayer).unwrap_err();
    let cannot = cannot_open(&error);
    assert_eq!(cannot.reason, CannotOpenReason::Expired);
    assert_eq!(cannot.receipt_id, "never-committed");
    assert!(recorder.check_challenge(&gone).is_err());
    assert!(!recorder.holds_turn("never-committed") && recorder.holds_turn("r1"));
}

// A stage checks a challenge's size against what a draw makes before it
// replays anything, also when it holds no receipt to check it against.
#[test]
fn a_stage_refuses_a_challenge_no_draw_makes() {
    let node = Node::two_lane_turn();
    node.commit(7, 0..4, 3);
    let recorder = &node.stages[1];
    recorder
        .check_challenge(&challenge(&[(2, &[0, 3]), (3, &[1])]))
        .unwrap();
    let every_layer: Vec<(u32, &[u32])> = (0..8).map(|l| (l, &[0u32][..])).collect();
    let mut twice = challenge(&[(2, &[1])]);
    twice.attention = vec![
        crate::attention::opening::AttentionPick {
            position: 1,
            layer: 2,
            head_draw: 0,
        };
        2
    ];
    let mut heads = challenge(&[(2, &[1])]);
    heads.head = vec![0, 1, 2, 3, 4];
    for refused in [
        challenge(&every_layer),
        challenge(&[(2, &[0, 1, 2, 3, 4])]),
        challenge(&[(2, &[])]),
        challenge(&[(2, &[2, 1])]),
        challenge(&[(2, &[4])]),
        challenge(&[(3, &[1]), (2, &[1])]),
        challenge(&[(9, &[1])]),
        twice,
        heads,
    ] {
        assert!(
            recorder.check_challenge(&refused).is_err(),
            "admitted {refused:?}"
        );
    }
}
