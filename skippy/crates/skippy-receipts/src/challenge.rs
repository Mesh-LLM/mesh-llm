//! The client's random challenge: [`LAYERS_PER_CHALLENGE`] layers of the
//! claimed model, each chosen with probability proportional to its weight
//! bytes, and [`POSITIONS_PER_LAYER`] positions in each.
//!
//! It also picks [`HEAD_POSITIONS`] of the positions the turn sampled a
//! token from, whose token the client checks against the output head.
//! And it picks [`ATTENTION_POSITIONS_PER_LAYER`] of each layer's positions,
//! with a head each, whose attention the client checks
//! ([`crate::attention`]).
//!
//! Every draw comes from the VRF output of the receipt's turn root under
//! the client's key ([`crate::vrf`]), through a [`DrawStream`], so anyone
//! holding the proof can recompute the challenge and see that the client
//! did not pick it.
//!
//! Spreading the positions over several layers matters: a node that breaks
//! the chain from one layer to the next at a single layer is caught only
//! when that layer is picked, so four layers catch it about four times as
//! often as one. Half of each layer's positions are decode positions when
//! the turn generated any, because a node that cheats only on the tokens it
//! shows would otherwise hide behind a long prompt.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use anyhow::{Result, bail};

use crate::attention::opening::AttentionPick;
use crate::draw::DrawStream;
use crate::wire::{Challenge, Pick, Receipt};

pub const LAYERS_PER_CHALLENGE: usize = 4;
pub const POSITIONS_PER_LAYER: usize = 4;
pub const HEAD_POSITIONS: usize = 4;
/// Positions of each picked layer whose attention the client checks.
pub const ATTENTION_POSITIONS_PER_LAYER: usize = 2;

/// Label of the challenge draws.
const CHALLENGE_LABEL: &str = "challenge";
/// Label of the head positions.
const HEAD_LABEL: &str = "head";
/// Label of the attention picks.
const ATTENTION_LABEL: &str = "attention";
/// Label of the audit-rate decision.
const AUDIT_LABEL: &str = "audit";

/// Whether the client audits the turn whose VRF output is `seed`, at `rate`.
pub fn audits(seed: &[u8], rate: f64) -> bool {
    DrawStream::new(seed, AUDIT_LABEL).unit() < rate
}

/// Draws a challenge for `receipt` from `seed`, the VRF output of its turn
/// root. `layer_bytes` comes from the claimed GGUF; layers it does not
/// list, or lists with no bytes, are never picked.
pub fn draw(receipt: &Receipt, layer_bytes: &BTreeMap<u32, u64>, seed: &[u8]) -> Result<Challenge> {
    let mut stream = DrawStream::new(seed, CHALLENGE_LABEL);
    let mut candidates: Vec<(u32, u64)> = layer_bytes
        .iter()
        .filter(|(_, bytes)| **bytes > 0)
        .map(|(layer, bytes)| (*layer, *bytes))
        .collect();
    if candidates.is_empty() {
        bail!("the claimed model has no weight bytes");
    }
    if receipt.position_end <= receipt.position_start {
        bail!("the receipt covers no positions");
    }
    let mut layers = BTreeSet::new();
    while layers.len() < LAYERS_PER_CHALLENGE && !candidates.is_empty() {
        let total: u64 = candidates.iter().map(|(_, b)| b).sum();
        let mut ticket = stream.below(total);
        let index = candidates
            .iter()
            .position(|(_, bytes)| {
                let hit = ticket < *bytes;
                ticket = ticket.saturating_sub(*bytes);
                hit
            })
            .expect("a ticket below the total");
        layers.insert(candidates.swap_remove(index).0);
    }
    let prompt = receipt.position_start..receipt.generation_start;
    let decode = receipt.generation_start..receipt.position_end;
    let picks: Vec<Pick> = layers
        .into_iter()
        .map(|layer| Pick {
            layer,
            positions: draw_positions(prompt.clone(), decode.clone(), &mut stream),
        })
        .collect();
    Ok(Challenge {
        receipt_id: receipt.receipt_id.clone(),
        attention: draw_attention(&picks, seed),
        picks,
        proof: None,
        head: draw_head(receipt, seed),
    })
}

/// [`ATTENTION_POSITIONS_PER_LAYER`] of each pick's positions, with a
/// head draw each. A challenge whose picks are changed draws them again.
pub fn draw_attention(picks: &[Pick], seed: &[u8]) -> Vec<AttentionPick> {
    let mut stream = DrawStream::new(seed, ATTENTION_LABEL);
    let mut attention = Vec::new();
    for pick in picks {
        let mut positions = pick.positions.clone();
        let count = ATTENTION_POSITIONS_PER_LAYER.min(positions.len());
        for _ in 0..count {
            let index = stream.below(positions.len() as u64) as usize;
            let position = positions.swap_remove(index);
            attention.push(AttentionPick {
                position,
                layer: pick.layer,
                head_draw: stream.below(1 << 32) as u32,
            });
        }
    }
    attention
}

/// [`HEAD_POSITIONS`] distinct output positions, sorted.
fn draw_head(receipt: &Receipt, seed: &[u8]) -> Vec<u32> {
    let outputs = receipt.output_positions();
    let mut stream = DrawStream::new(seed, HEAD_LABEL);
    let mut positions = BTreeSet::new();
    sample(
        &outputs,
        HEAD_POSITIONS.min(outputs.len()),
        &mut stream,
        &mut positions,
    );
    positions.into_iter().collect()
}

/// [`POSITIONS_PER_LAYER`] distinct positions, half from each span when both
/// have enough, sorted.
fn draw_positions(prompt: Range<u32>, decode: Range<u32>, stream: &mut DrawStream) -> Vec<u32> {
    let wanted = POSITIONS_PER_LAYER.min(prompt.len() + decode.len());
    let from_decode = (wanted / 2)
        .max(wanted.saturating_sub(prompt.len()))
        .min(decode.len());
    let mut positions = BTreeSet::new();
    sample(&decode, from_decode, stream, &mut positions);
    sample(&prompt, wanted - from_decode, stream, &mut positions);
    positions.into_iter().collect()
}

fn sample(span: &Range<u32>, count: usize, stream: &mut DrawStream, into: &mut BTreeSet<u32>) {
    let target = into.len() + count;
    while into.len() < target {
        into.insert(stream.in_range(span.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{RECEIPT_VERSION, StageCommitment};

    fn receipt(start: u32, generation: u32, end: u32) -> Receipt {
        Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: start,
            position_end: end,
            tokens: vec![0; (end - start) as usize],
            generation_start: generation,
            final_token: Some(1),
            stages: vec![StageCommitment {
                stage_index: 0,
                layer_start: 0,
                layer_end: 8,
                root: String::new(),
                leaf_count: 0,
                backend: String::new(),
                cache_types: Vec::new(),
                prefix: None,
                ubatches: Vec::new(),
                recurrent_state: None,
            }],
            binding: None,
        }
    }

    fn bytes(layers: u32) -> BTreeMap<u32, u64> {
        (0..layers).map(|l| (l, 1)).collect()
    }

    #[test]
    fn picks_distinct_layers_with_sorted_positions_in_range() {
        for seed in 0..50u8 {
            let c = draw(&receipt(10, 30, 40), &bytes(8), &[seed]).unwrap();
            assert_eq!(c.picks.len(), LAYERS_PER_CHALLENGE);
            assert!(c.picks.windows(2).all(|w| w[0].layer < w[1].layer));
            for pick in &c.picks {
                assert_eq!(pick.positions.len(), POSITIONS_PER_LAYER);
                assert!(pick.positions.windows(2).all(|w| w[0] < w[1]));
                assert!(pick.positions.iter().all(|p| (10..40).contains(p)));
            }
        }
    }

    #[test]
    fn half_of_each_layers_positions_are_decode_positions() {
        // 3000 prompt positions and 4 decode positions
        for seed in 0..50u8 {
            let c = draw(&receipt(0, 3000, 3004), &bytes(8), &[seed]).unwrap();
            for pick in &c.picks {
                let decode = pick.positions.iter().filter(|&&p| p >= 3000).count();
                assert_eq!(decode, POSITIONS_PER_LAYER / 2);
            }
        }
    }

    #[test]
    fn a_short_span_is_filled_from_the_other() {
        let c = draw(&receipt(0, 1, 9), &bytes(8), b"seed").unwrap();
        for pick in &c.picks {
            assert_eq!(pick.positions.len(), POSITIONS_PER_LAYER);
        }
        let c = draw(&receipt(3, 5, 6), &bytes(8), b"seed").unwrap();
        assert_eq!(c.picks[0].positions, vec![3, 4, 5]);
    }

    #[test]
    fn head_positions_are_output_positions() {
        for seed in 0..50u8 {
            let r = receipt(10, 30, 40);
            let c = draw(&r, &bytes(8), &[seed]).unwrap();
            assert_eq!(c.head.len(), HEAD_POSITIONS);
            assert!(c.head.windows(2).all(|w| w[0] < w[1]));
            assert!(c.head.iter().all(|p| r.output_positions().contains(p)));
        }
        // two outputs: the last prompt position and the one that sampled the final token
        let c = draw(&receipt(0, 5, 6), &bytes(8), b"seed").unwrap();
        assert_eq!(c.head, vec![4, 5]);
    }

    #[test]
    fn the_seed_decides_the_challenge() {
        let r = receipt(0, 100, 200);
        let a = draw(&r, &bytes(36), b"one").unwrap();
        assert_eq!(a, draw(&r, &bytes(36), b"one").unwrap());
        assert_ne!(a, draw(&r, &bytes(36), b"two").unwrap());
    }

    #[test]
    fn the_audit_rate_holds_over_many_seeds() {
        let audited = (0..10_000u32)
            .filter(|seed| audits(&seed.to_le_bytes(), 0.1))
            .count();
        assert!((800..1200).contains(&audited), "{audited}");
        assert!(!audits(b"x", 0.0));
        assert!(audits(b"x", 1.0));
    }

    #[test]
    fn layers_without_bytes_are_never_picked() {
        let only = BTreeMap::from([(2, 5), (3, 0)]);
        let c = draw(&receipt(0, 2, 4), &only, b"seed").unwrap();
        assert_eq!(c.picks.len(), 1);
        assert_eq!(c.picks[0].layer, 2);
    }
}
