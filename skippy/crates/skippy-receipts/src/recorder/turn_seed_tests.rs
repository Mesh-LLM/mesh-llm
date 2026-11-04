//! Which ubatches a turn replays, and the cells its replay is seeded with,
//! on the test model of `tests.rs`.

use super::*;

/// Every replay the replayer ran: the cells it copied and what each of its
/// steps captured.
fn runs(replayer: &FakeReplayer) -> Vec<Run> {
    replayer.runs.borrow().clone()
}

#[test]
fn a_long_turn_replays_only_the_ubatches_it_opens() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], false));
    record(&recorder, ubatch(&[(1, 2), (1, 3)], &[2, 3], false));
    record(&recorder, ubatch(&[(1, 4), (1, 5)], &[4, 5], true));
    record(&recorder, ubatch(&[(1, 6)], &[6], true));
    recorder.commit("r1", 1, 0..7).unwrap();
    let replayer = FakeReplayer::for_stage(1);
    recorder
        .open(&challenge(&[(2, &[3, 6])]), &replayer)
        .unwrap();

    // each opened ubatch replays alone, from the cells written before it
    assert_eq!(
        runs(&replayer),
        vec![
            (vec![0, 1], vec![vec![3]]),
            (vec![0, 1, 2, 3, 4, 5], vec![vec![6]]),
        ]
    );
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    assert_eq!(seed.positions(), (0..6).collect::<Vec<i32>>());
    recorder.check_seed(&seed).unwrap();
}

#[test]
fn a_cell_another_sequence_held_is_not_copied_before_the_turn_wrote_it() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    // sequence 3 holds cell 2 when the turn starts, then gives it up, and
    // the turn writes there later
    record(
        &recorder,
        ubatch(&[(1, 0), (1, 1), (3, 0)], &[0, 1, 2], true),
    );
    recorder.memory_op(MemoryOp::SeqRm, 3, -1, -1, -1);
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    record(&recorder, ubatch(&[(1, 3)], &[3], true));
    recorder.commit("r1", 1, 0..4).unwrap();
    let replayer = FakeReplayer::for_stage(1);
    recorder
        .open(&challenge(&[(2, &[1, 3])]), &replayer)
        .unwrap();

    // the first replay copies nothing, so its dummy column in cell 2 is
    // harmless; the second copies cell 2 after the turn wrote it
    assert_eq!(
        runs(&replayer),
        vec![(vec![], vec![vec![1]]), (vec![0, 1, 2], vec![vec![3]])]
    );
}

/// A turn of seven positions whose serving kept the prefix up to position
/// 4 and dropped the rest.
fn dropped_tail() -> Recorder {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], false));
    record(&recorder, ubatch(&[(1, 2), (1, 3)], &[2, 3], false));
    record(&recorder, ubatch(&[(1, 4), (1, 5)], &[4, 5], true));
    record(&recorder, ubatch(&[(1, 6)], &[6], true));
    recorder.commit("r1", 1, 0..7).unwrap();
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, 4, -1);
    recorder
}

#[test]
fn a_turn_whose_dropped_cells_keep_their_rows_copies_them() {
    let recorder = dropped_tail();
    let replayer = FakeReplayer::for_stage(1);
    recorder
        .open(&challenge(&[(2, &[1, 6])]), &replayer)
        .unwrap();
    assert_eq!(
        runs(&replayer),
        vec![
            (vec![], vec![vec![1]]),
            (vec![0, 1, 2, 3, 4, 5], vec![vec![6]])
        ]
    );
}

#[test]
fn a_turn_whose_last_cells_were_reused_replays_from_the_first_reused() {
    let recorder = dropped_tail();
    // another request writes cell 4
    record(&recorder, ubatch(&[(2, 0)], &[4], true));
    let replayer = FakeReplayer::for_stage(1);
    recorder
        .open(&challenge(&[(2, &[1, 6])]), &replayer)
        .unwrap();
    // the ubatch of position 1, then everything from the first reused cell
    assert_eq!(
        runs(&replayer),
        vec![
            (vec![], vec![vec![1]]),
            (vec![0, 1, 2, 3], vec![vec![], vec![6]])
        ]
    );
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    recorder.check_seed(&seed).unwrap();
}

#[test]
fn a_turn_with_rejected_drafts_replays_only_the_ubatches_it_opens() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    // a verify window: the last token and two drafts, the second rejected
    record(
        &recorder,
        ubatch(&[(1, 2), (1, 3), (1, 4)], &[2, 3, 4], true),
    );
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, 4, -1);
    // the next window writes position 4 again, in another cell
    record(&recorder, ubatch(&[(1, 4), (1, 5)], &[5, 4], true));
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, 5, -1);
    // the last window: the turn stops at position 5, so its drafts at 6 and
    // 7 stay in the cache, past the turn
    record(
        &recorder,
        ubatch(&[(1, 5), (1, 6), (1, 7)], &[4, 6, 7], true),
    );
    recorder.commit("r1", 1, 0..6).unwrap();
    let replayer = FakeReplayer::for_stage(1);
    recorder
        .open(&challenge(&[(2, &[3, 4, 5])]), &replayer)
        .unwrap();

    // each window replays alone, from the cells the turn left before it
    assert_eq!(
        runs(&replayer),
        vec![
            (vec![0, 1], vec![vec![3]]),
            (vec![0, 1, 2, 3], vec![vec![4]]),
            (vec![0, 1, 2, 3, 5], vec![vec![5]]),
        ]
    );
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    recorder.check_seed(&seed).unwrap();
}
