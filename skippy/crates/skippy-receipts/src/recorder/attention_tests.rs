//! The stage's side of the attention check, on the test model of
//! `tests.rs`: one head over its three-value stored row, which serves as
//! both K and V.

use std::collections::BTreeMap;

use super::*;
use crate::attention::opening::{AttentionPick, BlockChallenge, BlockPick, RowProof, RowsOpening};
use crate::attention::{AttentionLayout, AttentionOp, BLOCK_POSITIONS, Window};
use crate::attention_check::{
    block_challenge, block_passed, check_blocks, check_partials, verify_rows,
};
use crate::recorder::answered_rows::MAX_ROWS_REQUESTS_PER_RECEIPT;
use crate::wire::{LeafOpening, Opening};

fn layout(layer: u32) -> AttentionLayout {
    AttentionLayout {
        layer,
        op: AttentionOp {
            n_head: 1,
            n_head_kv: 1,
            head_dim_k: 3,
            head_dim_v: 3,
            scale: 0.01,
            softcap: 0.0,
            sinks: None,
            window: Window::Causal,
            output_rotation: None,
            q_rounding: None,
        },
        q_input: 0,
        k_store: 0,
        v_store: 0,
        output: 0,
        output_gate: None,
    }
}

fn leaf(opening: &Opening, layer: u32, position: u32) -> &LeafOpening {
    opening
        .stages
        .iter()
        .flat_map(|s| &s.layers)
        .find(|l| l.layer == layer)
        .and_then(|l| l.leaves.iter().find(|leaf| leaf.position == position))
        .expect("an opened leaf")
}

/// Opens `pick` with the challenge, then the rows of its blocks, and
/// checks them as a client would. Returns the rows opening.
fn open_and_check(
    recorder: &Recorder,
    replayer: &FakeReplayer,
    receipt: &Receipt,
    mut challenge: Challenge,
    pick: AttentionPick,
) -> RowsOpening {
    let layer = pick.layer;
    challenge.attention = vec![pick.clone()];
    let stage = recorder.open(&challenge, replayer).unwrap();
    assert_eq!(stage.attention.len(), 1);
    // past the challenge window, an opened turn stays pinned for its rows
    let later = std::time::Instant::now() + CHALLENGE_PIN;
    assert!(!recorder.pinned_cells(later).is_empty());
    let opening = Opening {
        receipt_id: challenge.receipt_id.clone(),
        stages: vec![stage.clone()],
    };
    let layout = layout(layer);
    let leaf = leaf(&opening, layer, pick.position);
    let head = check_partials(&layout, &pick, leaf, &stage.attention[0]).unwrap();
    assert_eq!(head.context.len(), pick.position as usize + 1);
    let blocks = block_challenge(&challenge.receipt_id, std::slice::from_ref(&head), b"seed");
    let rows = RowsOpening {
        receipt_id: challenge.receipt_id.clone(),
        stages: vec![recorder.open_rows(&blocks, replayer).unwrap()],
    };
    assert!(recorder.pinned_cells(later).is_empty());
    let layouts = BTreeMap::from([(layer, layout.clone())]);
    let context = verify_rows(receipt, &layouts, &rows).unwrap();
    let q = layout.q_values(&leaf.mixing).unwrap();
    let error = check_blocks(&layout, &head, &blocks.picks[0], &q, &context).unwrap();
    assert!(block_passed(error), "block error {error}");
    rows
}

fn with_attention(mut replayer: FakeReplayer, layer: u32) -> FakeReplayer {
    replayer.attention = Some(layout(layer));
    replayer
}

#[test]
fn a_turn_opens_partials_and_the_rows_it_computed() {
    let mut node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    node.replayers[1] = with_attention(FakeReplayer::for_stage(1), 3);
    let pick = AttentionPick {
        position: 3,
        layer: 3,
        head_draw: 5,
    };
    let rows = open_and_check(
        &node.stages[1],
        &node.replayers[1],
        &receipt,
        challenge(&[(3, &[1, 3])]),
        pick,
    );
    let opened = &rows.stages[0].rows;
    assert_eq!(opened.len(), 4, "the one block holds every position");
    assert!(
        opened
            .iter()
            .all(|r| matches!(r.proof, RowProof::Turn { .. }))
    );

    // a changed row no longer hashes to its leaf
    let mut forged = rows.clone();
    let row = &mut forged.stages[0].rows[2];
    let mut bytes = row.stored[0].bytes().unwrap().unwrap();
    bytes[0] ^= 1;
    row.stored[0] = TensorOpening::new(MixingKind::Stored, 0, Some((CACHE_TYPE, &bytes)));
    let layouts = BTreeMap::from([(3, layout(3))]);
    assert!(verify_rows(&receipt, &layouts, &forged).is_err());
}

// A client that repeats a rows request gets the kept answer, and the stage
// reads no rows for it: not while it holds the runtime's lock, and not after
// the cells were written. Past the cap, a request that needs a read is
// refused.
#[test]
fn a_repeated_rows_request_reads_no_rows_again() {
    let mut node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    node.replayers[1] = with_attention(FakeReplayer::for_stage(1), 3);
    let pick = AttentionPick {
        position: 3,
        layer: 3,
        head_draw: 5,
    };
    let (recorder, replayer) = (&node.stages[1], &node.replayers[1]);
    let first = open_and_check(
        recorder,
        replayer,
        &receipt,
        challenge(&[(3, &[1, 3])]),
        pick,
    );
    let blocks = BlockChallenge {
        receipt_id: receipt.receipt_id.clone(),
        picks: vec![BlockPick {
            position: 3,
            layer: 3,
            blocks: vec![0],
        }],
        proof: None,
    };
    let reads = replayer.row_reads.get();
    for _ in 0..100 {
        assert_eq!(
            recorder.open_rows(&blocks, replayer).unwrap(),
            first.stages[0]
        );
    }
    assert_eq!(replayer.row_reads.get(), reads);
    // the turn's cells are written; the kept answer still opens the rows
    recorder.memory_op(MemoryOp::SeqRm, 7, -1, -1, -1);
    record(
        recorder,
        ubatch(&[(5, 0), (5, 1), (5, 2), (5, 3)], &[0, 1, 2, 3], true),
    );
    assert_eq!(
        recorder.open_rows(&blocks, replayer).unwrap(),
        first.stages[0]
    );

    for _ in 1..MAX_ROWS_REQUESTS_PER_RECEIPT {
        recorder.count_rows_read(&receipt.receipt_id).unwrap();
    }
    let error = recorder.count_rows_read(&receipt.receipt_id).unwrap_err();
    assert!(
        format!("{error:#}").contains("already read rows"),
        "{error:#}"
    );
}

#[test]
fn a_cached_prefix_opens_its_rows_through_the_prefix_root() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    // a cached prompt of seventy positions, which another sequence continues
    let prompt: Vec<(i32, i32)> = (0..70).map(|p| (3, p)).collect();
    let cells: Vec<u32> = (0..70).collect();
    record(&recorder, ubatch(&prompt, &cells, true));
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    record(&recorder, ubatch(&[(1, 70), (1, 71)], &[70, 71], true));
    let commitment = recorder.commit("r1", 1, TurnId::default(), 70..72).unwrap();
    let receipt = Receipt {
        version: RECEIPT_VERSION,
        receipt_id: "r1".into(),
        model: "m".into(),
        position_start: 70,
        position_end: 72,
        tokens: vec![1070, 1071],
        generation_start: 71,
        final_token: None,
        stages: vec![commitment],
        binding: None,
    };
    let replayer = with_attention(FakeReplayer::for_stage(1), 2);
    let pick = AttentionPick {
        position: 71,
        layer: 2,
        head_draw: 0,
    };
    let rows = open_and_check(
        &recorder,
        &replayer,
        &receipt,
        challenge(&[(2, &[71])]),
        pick,
    );
    // two blocks of 72 positions: both are drawn, so every row is opened
    let opened = &rows.stages[0].rows;
    assert_eq!(opened.len(), 72);
    const { assert!(BLOCK_POSITIONS < 72) };
    let prefix = opened
        .iter()
        .filter(|r| matches!(r.proof, RowProof::Prefix { .. }))
        .count();
    assert_eq!(prefix, 70);

    // a block the stage did not open partials for is refused
    let unasked = BlockChallenge {
        receipt_id: "r1".into(),
        picks: vec![BlockPick {
            position: 70,
            layer: 2,
            blocks: vec![0],
        }],
        proof: None,
    };
    assert!(recorder.open_rows(&unasked, &replayer).is_err());
    let past = BlockChallenge {
        receipt_id: "r1".into(),
        picks: vec![BlockPick {
            position: 71,
            layer: 2,
            blocks: vec![2],
        }],
        proof: None,
    };
    assert!(recorder.open_rows(&past, &replayer).is_err());

    // once a prefix cell is reused, its rows cannot be opened
    recorder.memory_op(MemoryOp::SeqRm, 3, -1, -1, -1);
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, -1, -1);
    record(&recorder, ubatch(&[(5, 0)], &[1], true));
    let again = BlockChallenge {
        receipt_id: "r1".into(),
        picks: vec![BlockPick {
            position: 71,
            layer: 2,
            blocks: vec![0],
        }],
        proof: None,
    };
    assert!(recorder.open_rows(&again, &replayer).is_err());
}

/// The pinned cells, all of the one stream.
fn cells(recorder: &Recorder, now: std::time::Instant) -> Vec<u32> {
    let pinned = recorder.pinned_cells(now);
    assert!(pinned.iter().all(|c| c.stream == 0));
    pinned.iter().map(|c| c.cell).collect()
}

#[test]
fn a_recent_turn_pins_the_cells_its_attention_reads() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    // a cached prompt of three positions, which another sequence continues
    let prompt: Vec<(i32, i32)> = (0..3).map(|p| (3, p)).collect();
    record(&recorder, ubatch(&prompt, &[10, 11, 12], true));
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    record(&recorder, ubatch(&[(1, 3), (1, 4)], &[13, 14], true));
    recorder.commit("r1", 1, TurnId::default(), 3..5).unwrap();
    let now = std::time::Instant::now();
    assert_eq!(cells(&recorder, now), vec![10, 11, 12, 13, 14]);

    // freed cells keep their rows, so they stay pinned
    recorder.memory_op(MemoryOp::SeqRm, -1, -1, -1, -1);
    assert_eq!(cells(&recorder, now), vec![10, 11, 12, 13, 14]);
    // a cell another request wrote holds nothing of the turn
    record(&recorder, ubatch(&[(2, 0)], &[11], true));
    assert_eq!(cells(&recorder, now), vec![10, 12, 13, 14]);
    // and once the challenge window passes, nothing is pinned
    assert!(recorder.pinned_cells(now + CHALLENGE_PIN).is_empty());
}

#[test]
fn a_turn_no_client_can_challenge_pins_nothing() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[4, 5], true));
    recorder.commit("r1", 1, TurnId::default(), 0..2).unwrap();
    let now = std::time::Instant::now();
    assert_eq!(cells(&recorder, now), vec![4, 5]);
    recorder.release_pin("r1");
    assert!(recorder.pinned_cells(now).is_empty());
}

// A challenge that arrives while its turn is pinned keeps the turn's cells
// pinned until the opening ends, also past the challenge pin: the opening
// reads them after the replay, which takes seconds for a long prompt.
#[test]
fn a_turn_being_opened_stays_pinned_past_its_challenge_pin() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[4, 5], true));
    recorder.commit("r1", 1, TurnId::default(), 0..2).unwrap();
    let now = std::time::Instant::now();
    let past_window = now + CHALLENGE_PIN;
    assert!(recorder.pinned_cells(past_window).is_empty());

    let arrived = now + std::time::Duration::from_secs(10);
    recorder.hold_for_opening("r1", arrived);
    assert_eq!(cells(&recorder, past_window), vec![4, 5]);
    assert_eq!(
        recorder.next_pin_expiry(past_window),
        Some(arrived + OPENING_PIN)
    );
    // an opening that never ends gives the cells back
    assert!(recorder.pinned_cells(arrived + OPENING_PIN).is_empty());
    // one that ends without rows to read leaves the challenge pin
    recorder.end_opening("r1");
    assert!(recorder.pinned_cells(past_window).is_empty());
    assert_eq!(cells(&recorder, now), vec![4, 5]);

    // a challenge that comes after the pin ran out pins what still holds
    // the turn's rows
    recorder.memory_op(MemoryOp::SeqRm, -1, -1, -1, -1);
    record(&recorder, ubatch(&[(2, 0)], &[5], true));
    recorder.hold_for_opening("r1", past_window);
    assert_eq!(cells(&recorder, past_window), vec![4]);
}

// A challenge may pick a position of the cached prefix. Its attention is
// checked like a position of the turn: the query is the one its mixing leaf
// holds, and every row it attends to is a cell of the prefix.
#[test]
fn an_attention_pick_in_the_prefix_reads_the_cells_below_it() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    let prompt: Vec<(i32, i32)> = (0..70).map(|p| (3, p)).collect();
    let cells: Vec<u32> = (0..70).collect();
    record(&recorder, ubatch(&prompt, &cells, true));
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    record(&recorder, ubatch(&[(1, 70), (1, 71)], &[70, 71], true));
    let commitment = recorder.commit("r1", 1, TurnId::default(), 70..72).unwrap();
    let receipt = Receipt {
        version: RECEIPT_VERSION,
        receipt_id: "r1".into(),
        model: "m".into(),
        position_start: 70,
        position_end: 72,
        tokens: vec![1070, 1071],
        generation_start: 71,
        final_token: None,
        stages: vec![commitment],
        binding: None,
    };
    let replayer = with_attention(FakeReplayer::for_stage(1), 2);
    let pick = AttentionPick {
        position: 40,
        layer: 2,
        head_draw: 0,
    };
    let rows = open_and_check(
        &recorder,
        &replayer,
        &receipt,
        challenge(&[(2, &[40, 71])]),
        pick,
    );
    // the query attends to positions 0..=40, all cells of the prefix
    let opened = &rows.stages[0].rows;
    assert_eq!(opened.len(), 41);
    assert!(
        opened
            .iter()
            .all(|r| matches!(r.proof, RowProof::Prefix { .. }))
    );
    // a row bound to other leaves than the prefix commits is refused
    let mut forged = rows.clone();
    if let RowProof::Prefix { leaves, .. } = &mut forged.stages[0].rows[3].proof {
        *leaves = hex::encode([9u8; 32]);
    }
    let layouts = BTreeMap::from([(2, layout(2))]);
    assert!(verify_rows(&receipt, &layouts, &forged).is_err());
}
