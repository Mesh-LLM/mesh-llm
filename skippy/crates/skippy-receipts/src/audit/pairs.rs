//! The (token, expert) pairs whose expert matmuls an audit checks.
//!
//! A MoE layer routes each opened token to several experts, and checking
//! rows of every expert used costs a fetch per expert: 21 to 26 seconds on
//! large MoE models from Hugging Face (RECEIPTS.md, "Row fetch"). So an
//! audit checks [`PAIRS_PER_LAYER`] pairs drawn from those the opened
//! tokens were routed to, from the rows seed. The router is still checked
//! whole, so the routing itself is checked at every opened token, and the
//! glue check still combines every opened expert output.

use std::collections::BTreeSet;

use crate::draw::DrawStream;
use crate::wire::LayerOpening;

/// Pairs checked per MoE layer.
pub const PAIRS_PER_LAYER: usize = 16;

/// The pairs of `opening` to check, as (position, expert), or `None` when
/// the layer has no expert matmul.
pub fn picked_pairs(opening: &LayerOpening, seed: &[u8]) -> Option<BTreeSet<(u32, i32)>> {
    let routed: BTreeSet<(u32, i32)> = opening
        .leaves
        .iter()
        .flat_map(|leaf| {
            leaf.matrices
                .iter()
                .filter_map(|m| m.experts.as_ref())
                .flatten()
                .map(move |&expert| (leaf.position, expert))
        })
        .collect();
    if routed.is_empty() {
        return None;
    }
    if routed.len() <= PAIRS_PER_LAYER {
        return Some(routed);
    }
    let routed: Vec<(u32, i32)> = routed.into_iter().collect();
    let mut stream = DrawStream::new(seed, &format!("pairs/{}", opening.layer));
    let mut picked = BTreeSet::new();
    while picked.len() < PAIRS_PER_LAYER {
        picked.insert(routed[stream.below(routed.len() as u64) as usize]);
    }
    Some(picked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opening(tokens: u32, experts: i32) -> LayerOpening {
        let leaf = |position: u32| {
            let ids: Vec<i32> = (0..experts).map(|e| (e + position as i32) % 64).collect();
            serde_json::json!({
                "position": position, "residual_in": "", "residual_out": null,
                "residual_in_path": [], "mixing_path": [], "residual_out_path": [],
                "mixing": [], "selections": [],
                "matrices": [
                    {"index": 0, "weight": "blk.3.ffn_gate_inp.weight", "router": true, "input": "", "output": ""},
                    {"index": 1, "weight": "blk.3.ffn_up_exps.weight", "experts": ids, "input": "", "output": ""},
                ],
            })
        };
        serde_json::from_value(serde_json::json!({
            "layer": 3,
            "leaves": (0..tokens).map(leaf).collect::<Vec<_>>(),
        }))
        .unwrap()
    }

    #[test]
    fn pairs_are_drawn_from_the_routed_ones() {
        let opening = opening(4, 8);
        let picked = picked_pairs(&opening, b"seed").unwrap();
        assert_eq!(picked.len(), PAIRS_PER_LAYER);
        for &(position, expert) in &picked {
            assert!(position < 4);
            assert!((0..8).contains(&(expert - position as i32)));
        }
        assert_eq!(picked_pairs(&opening, b"seed"), Some(picked.clone()));
        assert_ne!(picked_pairs(&opening, b"other"), Some(picked));
    }

    #[test]
    fn few_routed_pairs_are_all_checked_and_a_dense_layer_has_none() {
        assert_eq!(picked_pairs(&opening(2, 4), b"seed").unwrap().len(), 8);
        let dense: LayerOpening =
            serde_json::from_value(serde_json::json!({"layer": 0, "leaves": []})).unwrap();
        assert_eq!(picked_pairs(&dense, b"seed"), None);
    }
}
