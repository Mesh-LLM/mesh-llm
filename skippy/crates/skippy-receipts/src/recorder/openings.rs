//! Openings already answered, so a repeated challenge is not replayed again.
//!
//! An opening is kept until it is the least recently used past the byte
//! budget, or its turn leaves the opening window.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::wire::{Challenge, LeafOpening, StageOpening};

/// Bytes of openings kept, across turns.
const DEFAULT_OPENING_CACHE_BYTES: usize = 128 << 20;

struct Entry {
    challenge: Challenge,
    opening: Arc<StageOpening>,
    bytes: usize,
}

/// Least recently used last.
pub(super) struct OpeningCache {
    entries: VecDeque<Entry>,
    bytes: usize,
    budget: usize,
}

impl Default for OpeningCache {
    fn default() -> Self {
        Self::new(DEFAULT_OPENING_CACHE_BYTES)
    }
}

impl OpeningCache {
    pub(super) fn new(budget: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            budget,
        }
    }

    pub(super) fn get(&mut self, challenge: &Challenge) -> Option<Arc<StageOpening>> {
        let index = self
            .entries
            .iter()
            .position(|entry| &entry.challenge == challenge)?;
        let entry = self.entries.remove(index)?;
        let opening = entry.opening.clone();
        self.entries.push_back(entry);
        Some(opening)
    }

    pub(super) fn insert(&mut self, challenge: Challenge, opening: Arc<StageOpening>) {
        let bytes = opening_bytes(&opening);
        if bytes > self.budget {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.challenge == challenge)
        {
            self.remove(index);
        }
        self.bytes += bytes;
        self.entries.push_back(Entry {
            challenge,
            opening,
            bytes,
        });
        while self.bytes > self.budget {
            self.remove(0);
        }
    }

    /// Drops every opening of a receipt.
    pub(super) fn forget(&mut self, receipt_id: &str) {
        let mut bytes = 0;
        self.entries.retain(|entry| {
            let keep = entry.challenge.receipt_id != receipt_id;
            if !keep {
                bytes += entry.bytes;
            }
            keep
        });
        self.bytes -= bytes;
    }

    fn remove(&mut self, index: usize) {
        if let Some(entry) = self.entries.remove(index) {
            self.bytes -= entry.bytes;
        }
    }
}

fn opening_bytes(opening: &StageOpening) -> usize {
    let paths = |path: &[String]| path.iter().map(String::len).sum::<usize>();
    let layers: usize = opening
        .layers
        .iter()
        .flat_map(|layer| &layer.leaves)
        .map(leaf_bytes)
        .sum();
    let anchors: usize = opening
        .anchors
        .iter()
        .map(|a| a.residual.len() + paths(&a.path))
        .sum();
    let boundaries: usize = opening
        .entry
        .iter()
        .chain(&opening.exit)
        .map(|b| b.leaf.len() + paths(&b.path))
        .sum();
    let head: usize = opening
        .head
        .iter()
        .map(|h| h.residual.len() + h.logits.as_ref().map_or(0, String::len) + paths(&h.path))
        .sum();
    layers + anchors + boundaries + head
}

fn leaf_bytes(leaf: &LeafOpening) -> usize {
    let paths = |path: &[String]| path.iter().map(String::len).sum::<usize>();
    leaf.residual_in.len()
        + leaf.residual_out.as_ref().map_or(0, String::len)
        + paths(&leaf.residual_in_path)
        + paths(&leaf.mixing_path)
        + paths(&leaf.residual_out_path)
        + leaf
            .mixing
            .iter()
            .map(|t| t.values.as_ref().map_or(0, String::len))
            .sum::<usize>()
        + leaf
            .matrices
            .iter()
            .map(|m| {
                m.weight.len()
                    + m.input.len()
                    + m.output.len()
                    + m.experts.as_ref().map_or(0, |e| e.len() * 4)
            })
            .sum::<usize>()
        + leaf
            .selections
            .iter()
            .map(|s| s.ids.len() * 4)
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{LayerOpening, Pick};

    fn challenge(receipt: &str, layer: u32) -> Challenge {
        Challenge {
            receipt_id: receipt.into(),
            picks: vec![Pick {
                layer,
                positions: vec![1, 2],
            }],
            proof: None,
            head: Vec::new(),
            attention: Vec::new(),
        }
    }

    fn opening(bytes: usize) -> Arc<StageOpening> {
        Arc::new(StageOpening {
            stage_index: 0,
            layers: vec![LayerOpening {
                layer: 0,
                leaves: vec![LeafOpening {
                    position: 1,
                    residual_in: "0".repeat(bytes),
                    residual_out: None,
                    residual_in_path: Vec::new(),
                    mixing_path: Vec::new(),
                    residual_out_path: Vec::new(),
                    mixing: Vec::new(),
                    matrices: Vec::new(),
                    selections: Vec::new(),
                }],
            }],
            anchors: Vec::new(),
            entry: Vec::new(),
            exit: Vec::new(),
            head: Vec::new(),
            attention: Vec::new(),
            recurrent: Vec::new(),
        })
    }

    #[test]
    fn the_least_recently_used_opening_goes_first() {
        let mut cache = OpeningCache::new(25);
        cache.insert(challenge("a", 1), opening(10));
        cache.insert(challenge("b", 1), opening(10));
        assert!(cache.get(&challenge("a", 1)).is_some());
        cache.insert(challenge("c", 1), opening(10));
        assert!(cache.get(&challenge("b", 1)).is_none());
        assert!(cache.get(&challenge("a", 1)).is_some());
        assert!(cache.get(&challenge("c", 1)).is_some());
        assert_eq!(cache.bytes, 20);
    }

    #[test]
    fn a_receipt_can_be_forgotten() {
        let mut cache = OpeningCache::new(100);
        cache.insert(challenge("a", 1), opening(10));
        cache.insert(challenge("a", 2), opening(10));
        cache.insert(challenge("b", 1), opening(10));
        cache.forget("a");
        assert!(cache.get(&challenge("a", 1)).is_none());
        assert!(cache.get(&challenge("b", 1)).is_some());
        assert_eq!(cache.bytes, 10);
    }

    #[test]
    fn an_opening_over_the_budget_is_not_kept() {
        let mut cache = OpeningCache::new(5);
        cache.insert(challenge("a", 1), opening(10));
        assert!(cache.get(&challenge("a", 1)).is_none());
        assert_eq!(cache.bytes, 0);
    }
}
