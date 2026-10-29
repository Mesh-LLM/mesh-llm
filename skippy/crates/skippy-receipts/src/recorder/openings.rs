//! Openings already answered, so a repeated challenge is not replayed again.
//!
//! An opening is kept until it is the least recently used past the byte
//! budget, or its turn leaves the opening window.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::wire::{Challenge, Opening};

/// Bytes of openings kept, across turns.
const DEFAULT_OPENING_CACHE_BYTES: usize = 128 << 20;

struct Entry {
    challenge: Challenge,
    opening: Arc<Opening>,
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

    pub(super) fn get(&mut self, challenge: &Challenge) -> Option<Arc<Opening>> {
        let index = self
            .entries
            .iter()
            .position(|entry| &entry.challenge == challenge)?;
        let entry = self.entries.remove(index)?;
        let opening = entry.opening.clone();
        self.entries.push_back(entry);
        Some(opening)
    }

    pub(super) fn insert(&mut self, challenge: Challenge, opening: Arc<Opening>) {
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

fn opening_bytes(opening: &Opening) -> usize {
    let paths = |path: &[String]| path.iter().map(String::len).sum::<usize>();
    opening
        .leaves
        .iter()
        .map(|leaf| {
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
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::LeafOpening;

    fn challenge(receipt: &str, layer: u32) -> Challenge {
        Challenge {
            receipt_id: receipt.into(),
            layer,
            positions: vec![1, 2],
        }
    }

    fn opening(receipt: &str, bytes: usize) -> Arc<Opening> {
        Arc::new(Opening {
            receipt_id: receipt.into(),
            stage_index: 0,
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
        })
    }

    #[test]
    fn the_least_recently_used_opening_goes_first() {
        let mut cache = OpeningCache::new(25);
        cache.insert(challenge("a", 1), opening("a", 10));
        cache.insert(challenge("b", 1), opening("b", 10));
        assert!(cache.get(&challenge("a", 1)).is_some());
        cache.insert(challenge("c", 1), opening("c", 10));
        assert!(cache.get(&challenge("b", 1)).is_none());
        assert!(cache.get(&challenge("a", 1)).is_some());
        assert!(cache.get(&challenge("c", 1)).is_some());
        assert_eq!(cache.bytes, 20);
    }

    #[test]
    fn a_receipt_can_be_forgotten() {
        let mut cache = OpeningCache::new(100);
        cache.insert(challenge("a", 1), opening("a", 10));
        cache.insert(challenge("a", 2), opening("a", 10));
        cache.insert(challenge("b", 1), opening("b", 10));
        cache.forget("a");
        assert!(cache.get(&challenge("a", 1)).is_none());
        assert!(cache.get(&challenge("b", 1)).is_some());
        assert_eq!(cache.bytes, 10);
    }

    #[test]
    fn an_opening_over_the_budget_is_not_kept() {
        let mut cache = OpeningCache::new(5);
        cache.insert(challenge("a", 1), opening("a", 10));
        assert!(cache.get(&challenge("a", 1)).is_none());
        assert_eq!(cache.bytes, 0);
    }
}
