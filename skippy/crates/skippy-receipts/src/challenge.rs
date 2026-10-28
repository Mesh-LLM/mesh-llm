//! The client's random challenge: one layer, chosen with probability
//! proportional to its weight bytes, and up to 16 positions of the turn.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use rand::RngExt;

use crate::wire::{Challenge, Receipt};

pub const POSITIONS_PER_CHALLENGE: usize = 16;

/// Draws a challenge for `receipt`. `layer_bytes` comes from the claimed
/// GGUF; layers it does not list are never picked.
pub fn draw(
    receipt: &Receipt,
    layer_bytes: &BTreeMap<u32, u64>,
    rng: &mut impl rand::Rng,
) -> Result<Challenge> {
    let layers = receipt.layer_count();
    let weighted: Vec<(u32, u64)> = layer_bytes
        .iter()
        .filter(|(layer, bytes)| **layer < layers && **bytes > 0)
        .map(|(layer, bytes)| (*layer, *bytes))
        .collect();
    let total: u64 = weighted.iter().map(|(_, b)| b).sum();
    if total == 0 {
        bail!("the claimed model has no weight bytes in the receipt's layers");
    }
    let mut ticket = rng.random_range(0..total);
    let mut layer = weighted[0].0;
    for (candidate, bytes) in &weighted {
        if ticket < *bytes {
            layer = *candidate;
            break;
        }
        ticket -= bytes;
    }
    let span = receipt.position_end.saturating_sub(receipt.position_start);
    if span == 0 {
        bail!("the receipt covers no positions");
    }
    let wanted = POSITIONS_PER_CHALLENGE.min(span as usize);
    let mut positions = BTreeSet::new();
    while positions.len() < wanted {
        positions.insert(receipt.position_start + rng.random_range(0..span));
    }
    Ok(Challenge {
        receipt_id: receipt.receipt_id.clone(),
        layer,
        positions: positions.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{RECEIPT_VERSION, StageCommitment};

    fn receipt(start: u32, end: u32) -> Receipt {
        Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: start,
            position_end: end,
            stages: vec![StageCommitment {
                stage_index: 0,
                layer_start: 0,
                layer_end: 4,
                root: String::new(),
                leaf_count: 0,
                backend: String::new(),
            }],
        }
    }

    #[test]
    fn positions_are_distinct_sorted_and_in_range() {
        let bytes = BTreeMap::from([(0, 1), (1, 1), (2, 1), (3, 1)]);
        let mut rng = rand::rng();
        for _ in 0..50 {
            let c = draw(&receipt(10, 40), &bytes, &mut rng).unwrap();
            assert_eq!(c.positions.len(), POSITIONS_PER_CHALLENGE);
            assert!(c.positions.windows(2).all(|w| w[0] < w[1]));
            assert!(c.positions.iter().all(|p| (10..40).contains(p)));
            assert!(c.layer < 4);
        }
    }

    #[test]
    fn short_turns_open_every_position() {
        let bytes = BTreeMap::from([(2, 5)]);
        let c = draw(&receipt(3, 8), &bytes, &mut rand::rng()).unwrap();
        assert_eq!(c.positions, vec![3, 4, 5, 6, 7]);
        assert_eq!(c.layer, 2);
    }
}
