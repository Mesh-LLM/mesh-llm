//! A stage whose KV cache keeps a stream per sequence, on the test model of
//! `tests.rs`.

use super::*;

/// Sequence 3 computes a prompt, sequence 2 writes the same cells of its
/// own stream, and sequence 1 continues the prompt from a copy of
/// sequence 3's stream.
fn continued(streams: KvStreams) -> Recorder {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    recorder.set_kv_streams(streams);
    record(
        &recorder,
        ubatch(&[(3, 0), (3, 1), (3, 2)], &[0, 1, 2], true),
    );
    record(&recorder, ubatch(&[(2, 0)], &[0], true));
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    record(&recorder, ubatch(&[(1, 3), (1, 4)], &[3, 4], true));
    recorder
}

#[test]
fn each_sequence_keeps_its_cells_in_its_own_stream() {
    let recorder = continued(KvStreams::PerSequence);
    recorder.commit("r1", 1, TurnId::default(), 3..5).unwrap();
    let pinned = recorder.pinned_cells(std::time::Instant::now());
    let expected: Vec<KvCell> = (0..5).map(|cell| KvCell { stream: 1, cell }).collect();
    assert_eq!(pinned, expected, "the prefix was copied into stream 1");
    // another lane writing its own stream leaves the turn's cells alone
    record(&recorder, ubatch(&[(2, 1), (2, 2)], &[1, 2], true));
    let replayer = FakeReplayer::for_stage(1);
    recorder.open(&challenge(&[(2, &[4])]), &replayer).unwrap();
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    assert!(seed.cells.iter().all(|c| c.stream == 1));
    assert_eq!(&seed.indices()[..3], &[0, 1, 2]);
    recorder.check_seed(&seed).unwrap();
}

#[test]
fn a_shared_stream_loses_the_cells_another_lane_writes() {
    let recorder = continued(KvStreams::Shared);
    // sequence 2 wrote cell 0 before the copy, so the prompt's first
    // position is gone from the cache
    assert!(recorder.commit("r1", 1, TurnId::default(), 3..5).is_err());
}
