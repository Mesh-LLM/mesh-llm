//! The recurrent state a turn of a hybrid model starts from, on the test
//! model of `tests.rs`.

use super::*;
use crate::attention::opening::AttentionPick;
use crate::commitment::recurrent_state_digest;
use crate::recurrent::verify_region;
use crate::wire::{MixingKind, MixingLeafOpening, TensorOpening};

fn hybrid() -> Recorder {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    recorder.set_recurrent(true);
    recorder
}

/// A turn of three positions in two ubatches, from `state`.
fn turn(recorder: &Recorder, state: &[u8]) -> StageCommitment {
    recorder.record_state(1, Some(state));
    record(recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    record(recorder, ubatch(&[(1, 2)], &[2], true));
    recorder.commit("r1", 1, 0..3).unwrap()
}

#[test]
fn a_hybrid_turn_commits_its_state_and_replays_from_it() {
    let recorder = hybrid();
    let commitment = turn(&recorder, b"state at 0");
    assert_eq!(
        commitment.recurrent_state,
        Some(hex::encode(recurrent_state_digest(b"state at 0")))
    );
    let replayer = FakeReplayer::for_stage(1);
    recorder.open(&challenge(&[(2, &[2])]), &replayer).unwrap();
    // the turn's own cells hold position 0 and 1, but the recurrent state
    // is known only at the turn's start, so every ubatch is replayed
    assert_eq!(replayer.steps.borrow().len(), 2);
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    assert!(seed.cells.is_empty());
    assert_eq!(seed.state.as_deref(), Some(&b"state at 0"[..]));
}

#[test]
fn a_hybrid_turn_without_its_state_is_not_committed() {
    let recorder = hybrid();
    record(&recorder, ubatch(&[(1, 0)], &[0], true));
    let error = recorder.commit("r1", 1, 0..1).unwrap_err();
    assert!(
        format!("{error:#}").contains("no recurrent state"),
        "{error:#}"
    );

    recorder.record_state(1, None);
    record(&recorder, ubatch(&[(1, 1)], &[1], true));
    let error = recorder.commit("r2", 1, 1..2).unwrap_err();
    assert!(format!("{error:#}").contains("could not read"), "{error:#}");
}

#[test]
fn a_state_reported_mid_turn_is_ignored() {
    let recorder = hybrid();
    recorder.record_state(1, Some(b"start"));
    record(&recorder, ubatch(&[(1, 0)], &[0], true));
    recorder.record_state(1, Some(b"later"));
    record(&recorder, ubatch(&[(1, 1)], &[1], true));
    let commitment = recorder.commit("r1", 1, 0..2).unwrap();
    assert_eq!(
        commitment.recurrent_state,
        Some(hex::encode(recurrent_state_digest(b"start")))
    );
}

#[test]
fn removing_a_sequence_drops_the_state_of_its_next_turn() {
    let recorder = hybrid();
    recorder.record_state(1, Some(b"stale"));
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, -1, -1);
    record(&recorder, ubatch(&[(1, 0)], &[0], true));
    assert!(recorder.commit("r1", 1, 0..1).is_err());
}

#[test]
fn a_turn_no_client_can_challenge_drops_its_state() {
    let recorder = hybrid();
    turn(&recorder, &[7; 1000]);
    recorder.release_pin("r1");
    let replayer = FakeReplayer::for_stage(1);
    let error = recorder
        .open(&challenge(&[(2, &[2])]), &replayer)
        .unwrap_err();
    assert!(format!("{error:#}").contains("no longer held"), "{error:#}");
    let state = recorder.lock();
    assert_eq!(state.committed_bytes, state.committed[0].bytes);
    assert!(
        state.committed[0]
            .state
            .as_ref()
            .is_some_and(|s| s.bytes.is_none())
    );
}

#[test]
fn a_stage_without_hybrid_memory_commits_no_state() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let commitment = turn(&recorder, b"ignored");
    assert!(commitment.recurrent_state.is_none());
}

/// A stage whose layer 2 reads a recurrent state, over a prompt of three
/// positions and two decode steps.
fn region_turn() -> (Recorder, FakeReplayer, Receipt) {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let mut stage = Stage::new(1);
    stage.state_layers = vec![2];
    for record in [
        ubatch(&[(1, 0), (1, 1), (1, 2)], &[0, 1, 2], true),
        ubatch(&[(1, 3)], &[3], true),
        ubatch(&[(1, 4)], &[4], true),
    ] {
        record_on(&recorder, &stage, record);
    }
    let receipt = Receipt {
        version: RECEIPT_VERSION,
        receipt_id: "r1".into(),
        model: "m".into(),
        position_start: 0,
        position_end: 5,
        tokens: recorder.turn_tokens("r1").unwrap_or_default(),
        generation_start: 2,
        final_token: None,
        stages: vec![recorder.commit("r1", 1, 0..5).unwrap()],
        binding: None,
    };
    (recorder, FakeReplayer::new(stage), receipt)
}

fn attention_at(position: u32, layer: u32) -> Challenge {
    let mut challenge = challenge(&[(layer, &[position])]);
    challenge.attention = vec![AttentionPick {
        position,
        layer,
        head_draw: 0,
    }];
    challenge
}

fn state_opened(leaf: &MixingLeafOpening) -> bool {
    leaf.mixing
        .iter()
        .any(|t| t.kind == MixingKind::State && t.values.is_some())
}

#[test]
fn a_pick_on_a_state_layer_opens_its_ubatch_and_the_state_it_left() {
    let (recorder, replayer, receipt) = region_turn();
    let challenge = attention_at(1, 2);
    let opening = recorder.open(&challenge, &replayer).unwrap();
    let region = &opening.recurrent[0];
    assert_eq!((region.start, region.end), (0, 3));
    let positions: Vec<u32> = region.leaves.iter().map(|l| l.position).collect();
    assert_eq!(positions, vec![0, 1, 2, 3]);
    let opened: Vec<bool> = region.leaves.iter().map(state_opened).collect();
    assert_eq!(opened, vec![true, false, false, true]);
    // every ubatch is replayed, so the state is the one the turn computed
    assert_eq!(*replayer.scopes.borrow(), vec!["full"]);
    verify_region(&receipt, &challenge.attention[0], region).unwrap();

    // a state the stage did not commit no longer hashes to its leaf
    let mut forged = region.clone();
    let state = forged.leaves[3]
        .mixing
        .iter_mut()
        .find(|t| t.kind == MixingKind::State)
        .unwrap();
    let mut bytes = state.bytes().unwrap().unwrap();
    bytes[0] ^= 1;
    *state = TensorOpening::new(MixingKind::State, 0, Some((GGML_F32, &bytes)));
    assert!(verify_region(&receipt, &challenge.attention[0], &forged).is_err());

    // nor does a region that leaves out the state it left
    let mut short = region.clone();
    short.leaves.pop();
    assert!(verify_region(&receipt, &challenge.attention[0], &short).is_err());
}

#[test]
fn the_last_ubatch_of_a_turn_opens_no_state_after_it() {
    let (recorder, replayer, receipt) = region_turn();
    let challenge = attention_at(4, 2);
    let opening = recorder.open(&challenge, &replayer).unwrap();
    let region = &opening.recurrent[0];
    assert_eq!((region.start, region.end), (4, 5));
    assert_eq!(region.leaves.len(), 1);
    assert!(state_opened(&region.leaves[0]));
    verify_region(&receipt, &challenge.attention[0], region).unwrap();
}

#[test]
fn a_pick_on_an_attention_layer_opens_no_region() {
    let (recorder, replayer, _) = region_turn();
    let opening = recorder.open(&attention_at(1, 3), &replayer).unwrap();
    assert!(opening.recurrent.is_empty());
}
