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
//! When the turn continued a cached prefix, it picks [`PREFIX_POSITIONS`]
//! positions of the prefix as well, the same in every layer. The node opens
//! them like the turn's own, so a prefix the node did not compute with the
//! claimed model is caught like a turn it did not.
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
/// Positions of the turn's cached prefix a challenge picks, in every picked
/// layer. A node replays one recorded ubatch for each, so this bounds the
/// work a cached turn's opening adds.
pub const PREFIX_POSITIONS: usize = 2;

/// Label of the challenge draws.
const CHALLENGE_LABEL: &str = "challenge";
/// Label of the head positions.
const HEAD_LABEL: &str = "head";
/// Label of the attention picks.
const ATTENTION_LABEL: &str = "attention";
/// Label of the prefix positions.
const PREFIX_LABEL: &str = "prefix";
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
    let prefix = draw_prefix(receipt.position_start, seed);
    let picks: Vec<Pick> = layers
        .into_iter()
        .map(|layer| Pick {
            layer,
            // the prefix lies below the turn, so the positions stay sorted
            positions: prefix
                .iter()
                .copied()
                .chain(draw_positions(prompt.clone(), decode.clone(), &mut stream))
                .collect(),
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

/// Refuses a challenge that no draw makes: more layers, positions, head
/// positions or attention picks than [`draw`] picks, in another order, or
/// outside the receipt. A node checks this before it replays anything, so
/// the work of an opening is bounded whoever sends the challenge.
pub fn check_shape(challenge: &Challenge, receipt: &Receipt) -> Result<()> {
    if challenge.receipt_id != receipt.receipt_id {
        bail!("the challenge names another receipt");
    }
    if challenge.picks.len() > LAYERS_PER_CHALLENGE {
        bail!(
            "the challenge picks {} layers; a draw picks at most {LAYERS_PER_CHALLENGE}",
            challenge.picks.len()
        );
    }
    let increasing = |values: &[u32]| values.windows(2).all(|pair| pair[0] < pair[1]);
    let layers: Vec<u32> = challenge.picks.iter().map(|p| p.layer).collect();
    if !increasing(&layers) {
        bail!("the challenge's layers are not in increasing order");
    }
    if let Some(pick) = challenge
        .picks
        .iter()
        .find(|pick| pick.layer >= receipt.layer_count())
    {
        bail!("layer {} is not in the receipt", pick.layer);
    }
    check_picked_positions(
        &challenge.picks,
        &(receipt.position_start..receipt.position_end),
    )?;
    // a head position is one the turn sampled a token from, so it is never
    // a position of the prefix: a prefix holds what an earlier request fed
    let outputs = receipt.output_positions();
    if challenge.head.len() > HEAD_POSITIONS
        || !increasing(&challenge.head)
        || !challenge.head.iter().all(|p| outputs.contains(p))
    {
        bail!(
            "the head positions are not at most {HEAD_POSITIONS} increasing positions the turn sampled from"
        );
    }
    let mut seen = BTreeSet::new();
    let mut per_layer: BTreeMap<u32, usize> = BTreeMap::new();
    for pick in &challenge.attention {
        let picked = challenge
            .picks
            .iter()
            .any(|p| p.layer == pick.layer && p.positions.contains(&pick.position));
        if !picked || !seen.insert((pick.layer, pick.position)) {
            bail!(
                "attention pick at position {} of layer {} is not one picked position, once",
                pick.position,
                pick.layer
            );
        }
        let count = per_layer.entry(pick.layer).or_default();
        *count += 1;
        if *count > ATTENTION_POSITIONS_PER_LAYER {
            bail!(
                "layer {} has more than {ATTENTION_POSITIONS_PER_LAYER} attention picks",
                pick.layer
            );
        }
    }
    Ok(())
}

/// Refuses picks whose positions no draw makes for a turn at positions
/// `turn`: in every pick, 1 to [`POSITIONS_PER_LAYER`] positions of the
/// turn, after at most [`PREFIX_POSITIONS`] positions of its prefix, the
/// same in every pick, all in increasing order. Every stage checks this, so
/// a stage opens at most [`PREFIX_POSITIONS`] positions of a prefix for one
/// challenge.
pub(crate) fn check_picked_positions(picks: &[Pick], turn: &Range<u32>) -> Result<()> {
    let mut prefix: Option<&[u32]> = None;
    for pick in picks {
        let increasing = pick.positions.windows(2).all(|pair| pair[0] < pair[1]);
        if !increasing || pick.positions.last().is_some_and(|last| *last >= turn.end) {
            bail!(
                "the positions of layer {} are not increasing positions of the turn or its prefix",
                pick.layer
            );
        }
        let (below, own) = pick
            .positions
            .split_at(pick.positions.partition_point(|p| *p < turn.start));
        if own.is_empty() || own.len() > POSITIONS_PER_LAYER {
            bail!(
                "layer {} picks {} positions of the turn; a draw picks 1 to {POSITIONS_PER_LAYER}",
                pick.layer,
                own.len()
            );
        }
        if below.len() > PREFIX_POSITIONS {
            bail!(
                "layer {} picks {} positions of the prefix; a draw picks at most {PREFIX_POSITIONS}",
                pick.layer,
                below.len()
            );
        }
        if prefix.is_some_and(|picked| picked != below) {
            bail!("the challenge's layers pick different positions of the prefix");
        }
        prefix = Some(below);
    }
    Ok(())
}

/// [`PREFIX_POSITIONS`] distinct positions of a prefix of `length`
/// positions, sorted, or every position of a shorter one.
fn draw_prefix(length: u32, seed: &[u8]) -> Vec<u32> {
    let mut stream = DrawStream::new(seed, PREFIX_LABEL);
    let mut positions = BTreeSet::new();
    sample(
        &(0..length),
        PREFIX_POSITIONS.min(length as usize),
        &mut stream,
        &mut positions,
    );
    positions.into_iter().collect()
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
                // two positions of the prefix, then four of the turn
                assert_eq!(pick.positions.len(), PREFIX_POSITIONS + POSITIONS_PER_LAYER);
                assert!(pick.positions.windows(2).all(|w| w[0] < w[1]));
                let (prefix, turn) = pick.positions.split_at(PREFIX_POSITIONS);
                assert!(prefix.iter().all(|p| (0..10).contains(p)));
                assert!(turn.iter().all(|p| (10..40).contains(p)));
                assert_eq!(prefix, &c.picks[0].positions[..PREFIX_POSITIONS]);
            }
        }
    }

    #[test]
    fn a_prefix_is_picked_exactly_when_the_turn_continues_one() {
        let prefix_of = |start: u32| {
            let r = receipt(start, start + 2, start + 4);
            let c = draw(&r, &bytes(8), b"seed").unwrap();
            check_shape(&c, &r).unwrap();
            c.prefix_positions(start)
        };
        assert!(prefix_of(0).is_empty());
        // a prefix of one position is picked whole
        assert_eq!(prefix_of(1), vec![0]);
        assert_eq!(prefix_of(2), vec![0, 1]);
        let large = prefix_of(30_000);
        assert_eq!(large.len(), PREFIX_POSITIONS);
        assert!(large.iter().all(|p| *p < 30_000));
        // the positions come from a stream of their own: the layers and the
        // turn's positions are the ones a turn without a prefix would get
        let (with, without) = (receipt(5, 7, 9), receipt(0, 2, 4));
        let with = draw(&with, &bytes(8), b"seed").unwrap();
        let without = draw(&without, &bytes(8), b"seed").unwrap();
        for (a, b) in with.picks.iter().zip(&without.picks) {
            assert_eq!(a.layer, b.layer);
            let turn: Vec<u32> = a.positions[PREFIX_POSITIONS..]
                .iter()
                .map(|p| p - 5)
                .collect();
            assert_eq!(turn, b.positions);
        }
    }

    #[test]
    fn every_draw_fits_the_shape_a_node_admits() {
        for seed in 0..50u8 {
            for (start, generation, end) in [(10, 30, 40), (0, 3000, 3004), (3, 5, 6), (0, 1, 2)] {
                let r = receipt(start, generation, end);
                let c = draw(&r, &bytes(8), &[seed]).unwrap();
                check_shape(&c, &r).unwrap();
            }
        }
    }

    #[test]
    fn a_challenge_no_draw_makes_is_refused() {
        let r = receipt(10, 30, 40);
        let drawn = draw(&r, &bytes(8), b"seed").unwrap();
        type Change = Box<dyn Fn(&mut Challenge)>;
        let changes: Vec<Change> = vec![
            Box::new(|c| c.receipt_id = "other".into()),
            // every layer at every position
            Box::new(|c| {
                c.picks = (0..8)
                    .map(|layer| Pick {
                        layer,
                        positions: (10..40).collect(),
                    })
                    .collect();
            }),
            Box::new(|c| c.picks[0].positions = (10..15).collect()),
            Box::new(|c| c.picks[0].positions = Vec::new()),
            Box::new(|c| c.picks[0].positions = vec![12, 11]),
            Box::new(|c| c.picks[0].positions = vec![11, 11]),
            Box::new(|c| c.picks[0].positions = vec![9]),
            Box::new(|c| c.picks[0].positions = vec![40]),
            // more positions of the prefix than a draw picks
            Box::new(|c| {
                for pick in &mut c.picks {
                    let turn = pick.positions.split_off(PREFIX_POSITIONS);
                    pick.positions = (0..=PREFIX_POSITIONS as u32).chain(turn).collect();
                }
            }),
            // other positions of the prefix in one layer than in the rest
            Box::new(|c| {
                let other = (0..10).find(|p| !c.picks[0].positions.contains(p)).unwrap();
                c.picks[0].positions[0] = other;
                c.picks[0].positions.sort_unstable();
            }),
            // a position of the prefix sampled no token
            Box::new(|c| c.head = vec![c.picks[0].positions[0]]),
            Box::new(|c| c.picks[0].layer = 8),
            Box::new(|c| c.picks.swap(0, 1)),
            Box::new(|c| c.picks[1].layer = c.picks[0].layer),
            Box::new(|c| c.head = (29..40).collect()),
            Box::new(|c| c.head = vec![12]),
            Box::new(|c| c.head = vec![31, 30]),
            Box::new(|c| c.attention[0].layer = 99),
            Box::new(|c| c.attention[0].position = 40),
            Box::new(|c| c.attention[1] = c.attention[0].clone()),
            Box::new(|c| {
                let layer = c.picks[0].layer;
                c.attention = c.picks[0]
                    .positions
                    .iter()
                    .map(|&position| AttentionPick {
                        position,
                        layer,
                        head_draw: 0,
                    })
                    .collect();
            }),
        ];
        for (i, change) in changes.iter().enumerate() {
            let mut c = drawn.clone();
            change(&mut c);
            assert!(check_shape(&c, &r).is_err(), "change {i} was admitted");
        }
        check_shape(&drawn, &r).unwrap();
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
        assert_eq!(c.picks[0].positions[PREFIX_POSITIONS..], [3, 4, 5]);
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
