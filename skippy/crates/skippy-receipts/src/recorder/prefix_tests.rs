//! A stage opens a picked position of a turn's cached prefix by replaying
//! the one ubatch that wrote it, on the test model of `tests.rs`.

use super::*;

/// The decode step of a replay, as the replayer was given it.
fn decode(step: &ReplayStep) -> (&UbatchRecord, &[u32]) {
    match step {
        ReplayStep::Decode {
            record, capture, ..
        } => (record, capture),
        ReplayStep::SeqRm { .. } => panic!("a decode step"),
    }
}

/// Sequence 7 computes three positions, then a turn of two more.
fn continued_turn() -> (Node, Receipt) {
    let node = Node::new();
    node.record(ubatch(&[(7, 0), (7, 1), (7, 2)], &[0, 1, 2], true));
    node.commit(7, 0..3, 3);
    node.record(ubatch(&[(7, 3)], &[3], true));
    node.record(ubatch(&[(7, 4)], &[4], true));
    let receipt = node.commit(7, 3..5, 4);
    (node, receipt)
}

#[test]
fn a_prefix_position_is_opened_by_replaying_its_ubatch() {
    let (node, receipt) = continued_turn();
    verify_shape(&receipt, MODEL_LAYERS).unwrap();
    // the stage that holds layer 0 commits the prefix's tokens
    let prefix = |stage: usize| receipt.stages[stage].prefix.clone().unwrap();
    assert_eq!((prefix(0).positions, prefix(1).positions), (3, 3));
    assert!(prefix(0).tokens.is_some() && prefix(1).tokens.is_none());

    let challenge = challenge(&[(1, &[1, 4]), (3, &[1, 4])]);
    let opening = node.open(&challenge).unwrap();
    verify_paths(&receipt, &challenge, &opening).unwrap();
    assert_eq!(opening.stages[0].prefix_tokens, vec![1000, 1001, 1002]);
    assert!(opening.stages[1].prefix_tokens.is_empty());
    for stage in &opening.stages {
        let opened: Vec<u32> = stage.prefix.iter().map(|p| p.position).collect();
        assert_eq!(opened, [1]);
        assert_eq!(stage.prefix[0].cell, 1);
        // a leaf of the prefix has no path
        assert!(stage.layers[0].leaves[0].mixing_path.is_empty());
        assert!(!stage.layers[0].leaves[1].mixing_path.is_empty());
        assert!(stage.exit[0].path.is_empty());
    }
    assert_eq!(opening.stages[0].anchors[0].token, 1001);
    assert!(opening.stages[0].anchors[0].path.is_empty());
    // the turn is replayed from its prefix and its own cells; the ubatch of
    // position 1 began at position 0, so it is replayed from nothing
    let runs = node.replayers[1].runs.borrow();
    assert_eq!(
        *runs,
        vec![
            (vec![0, 1, 2, 3], vec![vec![4]]),
            (Vec::new(), vec![vec![1]])
        ]
    );
    // a stage after the first replays the ubatch from its recorded input
    let steps = node.replayers[1].steps.borrow();
    assert_eq!(decode(&steps[0]).0.input.as_ref().unwrap().rows.len(), 12);
}

#[test]
fn a_prefix_position_of_another_ubatch_is_replayed_from_the_cells_below_it() {
    let node = Node::new();
    node.record(ubatch(&[(7, 0), (7, 1)], &[0, 1], true));
    node.record(ubatch(&[(7, 2), (7, 3)], &[2, 3], true));
    node.commit(7, 0..4, 4);
    node.record(ubatch(&[(7, 4)], &[4], true));
    let receipt = node.commit(7, 4..5, 4);
    // both positions of the second ubatch, and one of the first
    for (picked, expected) in [
        (vec![2, 3, 4], vec![(vec![0, 1], vec![vec![2, 3]])]),
        (
            vec![1, 3, 4],
            vec![(Vec::new(), vec![vec![1]]), (vec![0, 1], vec![vec![3]])],
        ),
    ] {
        let node_challenge = challenge(&[(2, &picked)]);
        let opening = node.open(&node_challenge).unwrap();
        verify_paths(&receipt, &node_challenge, &opening).unwrap();
        let runs = node.replayers[1].runs.borrow();
        // the turn's own replay comes first
        let lineage = &runs[runs.len() - expected.len()..];
        assert_eq!(lineage, expected, "{picked:?}");
    }
}

#[test]
fn a_prefix_another_sequence_computed_is_replayed_as_the_turns_own() {
    let node = Node::new();
    // a cached prompt in sequence 3, shared with sequence 1
    node.record(ubatch(&[(3, 0), (3, 1), (3, 2)], &[0, 1, 2], true));
    for recorder in &node.stages {
        recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    }
    node.record(ubatch(&[(1, 3), (1, 4)], &[3, 4], true));
    let receipt = node.commit(1, 3..5, 4);
    let challenge = challenge(&[(0, &[2, 4]), (2, &[2, 4])]);
    let opening = node.open(&challenge).unwrap();
    verify_shape(&receipt, MODEL_LAYERS).unwrap();
    verify_paths(&receipt, &challenge, &opening).unwrap();
    // the ubatch of sequence 3 runs as sequence 1, at its recorded cells
    let steps = node.replayers[1].steps.borrow();
    let (record, capture) = decode(&steps[0]);
    assert_eq!(record.seq, vec![1, 1, 1]);
    assert_eq!(record.token, vec![1000, 1001, 1002]);
    assert_eq!(
        (&record.pos, &record.cell),
        (&vec![0, 1, 2], &vec![0, 1, 2])
    );
    assert_eq!(capture, [2]);
    assert!(node.replayers[1].seed.borrow().is_none());
}

#[test]
fn a_prefix_ubatch_replays_whole_with_the_columns_the_prefix_dropped() {
    let node = Node::new();
    // the cache sequence computed five positions and kept three
    let columns: Vec<(i32, i32)> = (0..5).map(|p| (3, p)).collect();
    node.record(ubatch(&columns, &[0, 1, 2, 3, 4], true));
    for recorder in &node.stages {
        recorder.memory_op(MemoryOp::SeqRm, 3, -1, 3, -1);
        recorder.memory_op(MemoryOp::SeqCp, 3, 1, 0, 3);
    }
    node.record(ubatch(&[(1, 3)], &[9], true));
    let receipt = node.commit(1, 3..4, 3);
    assert_eq!(receipt.stages[0].prefix.as_ref().unwrap().positions, 3);
    let challenge = challenge(&[(3, &[1, 3])]);
    let opening = node.open(&challenge).unwrap();
    verify_paths(&receipt, &challenge, &opening).unwrap();
    let steps = node.replayers[1].steps.borrow();
    let (record, capture) = decode(&steps[0]);
    assert_eq!(record.pos, vec![0, 1, 2, 3, 4]);
    assert_eq!(record.seq, vec![1; 5]);
    assert_eq!(capture, [1]);
}

#[test]
fn a_column_of_the_turns_own_sequence_in_a_prefix_ubatch_is_a_dummy() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    // sequence 1 served another request when sequence 3 computed the cache
    record(
        &recorder,
        ubatch(&[(1, 40), (3, 0), (3, 1)], &[9, 0, 1], true),
    );
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, -1, -1);
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    recorder.commit("r1", 1, TurnId::default(), 2..3).unwrap();
    let replayer = FakeReplayer::for_stage(1);
    recorder
        .open(&challenge(&[(2, &[1, 2])]), &replayer)
        .unwrap();
    let runs = replayer.runs.borrow();
    assert_eq!(runs.last().unwrap(), &(Vec::new(), vec![vec![1]]));
    drop(runs);
    let steps = replayer.steps.borrow();
    let (record, _) = decode(&steps[0]);
    // the other request's column takes the writer's number, as a dummy
    assert_eq!(record.seq, vec![3, 1, 1]);
    assert_eq!(record.token, vec![DUMMY_TOKEN, 1000, 1001]);
    assert_eq!(record.pos, vec![0, 0, 1]);
    assert_eq!(record.cell, vec![9, 0, 1]);
    assert_eq!(
        record.input.as_ref().unwrap().rows,
        vec![0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7]
    );
}

#[test]
fn a_prefix_replay_that_diverges_is_anchored_to_the_committed_values() {
    let (node, receipt) = continued_turn();
    let mut replayer = FakeReplayer::for_stage(1);
    replayer.fused = true;
    let challenge = challenge(&[(2, &[1, 4])]);
    let opened = node.stages[1].open(&challenge, &replayer).unwrap();
    // each scope replays the turn and the prefix's ubatch
    assert_eq!(
        *replayer.scopes.borrow(),
        [
            "full",
            "full",
            "committed",
            "committed",
            "anchored",
            "anchored"
        ]
    );
    let anchors = replayer.anchors.borrow();
    assert!(anchors.iter().any(|a| a.pos == 1 && a.seq == 7));
    let (fresh, _) = continued_turn();
    let honest = FakeReplayer::for_stage(1);
    assert_eq!(opened, fresh.stages[1].open(&challenge, &honest).unwrap());
    drop(receipt);
}

#[test]
fn a_replay_that_does_not_reproduce_the_prefix_is_refused() {
    let (node, _) = continued_turn();
    let mut replayer = FakeReplayer::for_stage(1);
    replayer.corrupt = true;
    let error = node.stages[1]
        .open(&challenge(&[(2, &[1, 4])]), &replayer)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("replay diverged"),
        "{error:#}"
    );
}

#[test]
fn a_prefix_cell_a_later_request_wrote_cannot_be_opened() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(3, 0), (3, 1)], &[0, 1], true));
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    recorder.commit("r1", 1, TurnId::default(), 2..3).unwrap();
    // both sequences end, and another request takes the cell of position 1
    recorder.memory_op(MemoryOp::SeqRm, 3, -1, -1, -1);
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, -1, -1);
    record(&recorder, ubatch(&[(5, 0)], &[1], true));
    let replayer = FakeReplayer::for_stage(1);
    let error = recorder
        .open(&challenge(&[(2, &[1, 2])]), &replayer)
        .unwrap_err();
    let cannot = error.downcast_ref::<CannotOpen>().expect("a cannot-open");
    assert_eq!(cannot.reason, CannotOpenReason::CellsOverwritten);
    assert_eq!(cannot.receipt_id, "r1");
    assert_eq!(replayer.replays.get(), 0);
}

/// Why a turn of sequence 1 at position `at` cannot be committed.
fn refused(recorder: &Recorder, at: i32) -> String {
    record(recorder, ubatch(&[(1, at)], &[20], true));
    let at = u32::try_from(at).unwrap();
    let error = recorder
        .commit("r", 1, TurnId::default(), at..at + 1)
        .unwrap_err();
    format!("{error:#}")
}

#[test]
fn a_turn_whose_prefix_cannot_be_replayed_gets_no_receipt() {
    let cached = || {
        let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
        record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
        recorder.commit("r0", 1, TurnId::default(), 0..2).unwrap();
        recorder
    };
    // nothing in the way: the turn commits its prefix
    let recorder = cached();
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    recorder.commit("r", 1, TurnId::default(), 2..3).unwrap();

    // a state load: the rows the turn reads were computed elsewhere
    let recorder = cached();
    recorder.memory_op(MemoryOp::StateRead, 1, -1, -1, -1);
    assert!(refused(&recorder, 2).contains("cached prefix is unknown"));

    // a compaction moved every cell
    let recorder = cached();
    recorder.memory_op(MemoryOp::Compact, -1, -1, -1, -1);
    assert!(refused(&recorder, 2).contains("cached prefix is unknown"));

    // a ubatch the recorder could not record computed position 2
    let recorder = cached();
    recorder.unrecorded_ubatch("a ubatch of embeddings");
    let message = refused(&recorder, 3);
    assert!(message.contains("cached prefix is unknown"), "{message}");
    assert!(message.contains("position 2"), "{message}");

    // cells of two requests
    let recorder = cached();
    record(
        &recorder,
        ubatch(&[(2, 0), (2, 1), (2, 2)], &[10, 11, 12], true),
    );
    recorder.memory_op(MemoryOp::SeqCp, 2, 1, 2, 3);
    let message = refused(&recorder, 3);
    assert!(message.contains("cannot follow"), "{message}");
}

#[test]
fn a_prefix_past_the_input_budget_gets_no_receipt() {
    // a stage after the first keeps 4 bytes of input per token; the second
    // ubatch is past the budget, so the first goes
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu").with_lineage_input_bytes(12);
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    record(&recorder, ubatch(&[(1, 2), (1, 3)], &[2, 3], true));
    recorder.commit("r0", 1, TurnId::default(), 0..4).unwrap();
    let message = refused(&recorder, 4);
    assert!(message.contains("no longer held"), "{message}");

    // a ubatch dropped while the turn ran: the turn read the prefix, but
    // the stage could not open it, so it commits nothing
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu").with_lineage_input_bytes(12);
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.commit("r0", 1, TurnId::default(), 0..2).unwrap();
    record(&recorder, ubatch(&[(1, 2), (1, 3)], &[2, 3], true));
    let error = recorder
        .commit("r", 1, TurnId::default(), 2..4)
        .unwrap_err();
    assert!(format!("{error:#}").contains("no longer held"), "{error:#}");
    // a first stage keeps no input rows, so nothing is dropped
    let first = Recorder::new(0, STAGES[0].clone(), "cpu").with_lineage_input_bytes(0);
    let stage = Stage::new(0);
    let mut prompt = ubatch(&[(1, 0), (1, 1)], &[0, 1], true);
    prompt.input = None;
    record_on(&first, &stage, prompt);
    first.commit("r0", 1, TurnId::default(), 0..2).unwrap();
    let mut next = ubatch(&[(1, 2)], &[2], true);
    next.input = None;
    record_on(&first, &stage, next);
    let commitment = first.commit("r", 1, TurnId::default(), 2..3).unwrap();
    assert!(commitment.prefix.unwrap().tokens.is_some());
}

#[test]
fn a_hybrid_stage_gives_no_receipt_for_a_cached_prefix() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    recorder.set_recurrent(true);
    recorder.record_state(1, Some(b"state at 0"));
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.commit("r0", 1, TurnId::default(), 0..2).unwrap();
    recorder.record_state(1, Some(b"state at 2"));
    let message = refused(&recorder, 2);
    assert!(message.contains("recurrent state"), "{message}");
}

#[test]
fn a_stage_admits_only_the_prefix_positions_a_draw_picks() {
    let (node, _) = continued_turn();
    let recorder = &node.stages[1];
    recorder
        .check_challenge(&challenge(&[(2, &[0, 2, 3, 4]), (3, &[0, 2, 4])]))
        .unwrap();
    let mut head = challenge(&[(2, &[1, 4])]);
    head.head = vec![1];
    for refused in [
        // three positions of the prefix
        challenge(&[(2, &[0, 1, 2, 4])]),
        // other positions in one layer than in the other
        challenge(&[(2, &[0, 4]), (3, &[1, 4])]),
        // no position of the turn
        challenge(&[(2, &[0, 1])]),
        // a head position in the prefix
        head,
    ] {
        assert!(
            recorder.check_challenge(&refused).is_err(),
            "admitted {refused:?}"
        );
    }
}
