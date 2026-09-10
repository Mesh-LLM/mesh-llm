//! Shared-segment accounting: fractional credit so physical bytes are never
//! double-counted across entries (#1650 first slice).

use std::collections::BTreeMap;

use crate::policy::EntryKey;

/// Opaque shared-segment identity.
pub type SegmentId = u64;

/// Ledger of shared physical segments and their referencing entries.
///
/// A segment referenced by N entries contributes `size / N` bytes to each
/// entry's effective footprint; releasing an entry drops its reference, and a
/// segment with no references disappears entirely.
#[derive(Debug, Default, Clone)]
pub struct SharedSegmentLedger {
    segments: BTreeMap<SegmentId, SegmentRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentRecord {
    pub size: u64,
    pub references: Vec<EntryKey>,
}

impl SharedSegmentLedger {
    /// Register a segment of `size` bytes referenced by `entry`. Adding the
    /// same reference twice is a no-op.
    pub fn add(&mut self, segment: SegmentId, size: u64, entry: EntryKey) {
        let record = self.segments.entry(segment).or_insert(SegmentRecord {
            size,
            references: Vec::new(),
        });
        if !record.references.contains(&entry) {
            record.references.push(entry);
        }
    }

    /// Release `entry`'s reference to `segment`; drop the segment when the
    /// last reference goes.
    pub fn release(&mut self, segments: &[SegmentId], entry: EntryKey) {
        for segment in segments {
            if let Some(record) = self.segments.get_mut(segment) {
                record.references.retain(|&r| r != entry);
                if record.references.is_empty() {
                    self.segments.remove(segment);
                }
            }
        }
    }

    /// Fractional bytes charged to `entry` for its shared segments.
    pub fn fractional_bytes(&self, entry: EntryKey, segments: &[SegmentId]) -> f64 {
        segments
            .iter()
            .filter_map(|s| self.segments.get(s))
            .filter(|r| r.references.contains(&entry))
            .map(|r| r.size as f64 / r.references.len() as f64)
            .sum()
    }

    /// Read access for marginal-release computation.
    pub fn segment_record(&self, segment: SegmentId) -> Option<&SegmentRecord> {
        self.segments.get(&segment)
    }

    /// Physical bytes that would actually be released if `entry` were
    /// removed: its exclusive bytes are caller-side, so this covers only
    /// segments where this entry holds the last reference — the marginal
    /// physical release, not the fractional credit.
    pub fn marginal_physical_bytes(&self, entry: EntryKey, segments: &[SegmentId]) -> u64 {
        segments
            .iter()
            .filter_map(|s| self.segments.get(s))
            .filter(|r| r.references.len() == 1 && r.references[0] == entry)
            .map(|r| r.size)
            .sum()
    }

    /// Total physical bytes held in the ledger.
    pub fn total_bytes(&self) -> u64 {
        self.segments.values().map(|r| r.size).sum()
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_credit_splits_bytes_without_double_counting() {
        let mut ledger = SharedSegmentLedger::default();
        ledger.add(1, 300, 10);
        ledger.add(1, 300, 11);
        ledger.add(1, 300, 12);
        assert_eq!(ledger.total_bytes(), 300);
        for entry in [10u64, 11, 12] {
            assert_eq!(ledger.fractional_bytes(entry, &[1]), 100.0);
        }
    }

    #[test]
    fn releasing_last_reference_drops_segment() {
        let mut ledger = SharedSegmentLedger::default();
        ledger.add(7, 128, 1);
        ledger.release(&[7], 1);
        assert_eq!(ledger.total_bytes(), 0);
        assert_eq!(ledger.segment_count(), 0);
    }

    #[test]
    fn duplicate_reference_is_idempotent() {
        let mut ledger = SharedSegmentLedger::default();
        ledger.add(7, 128, 1);
        ledger.add(7, 128, 1);
        assert_eq!(ledger.fractional_bytes(1, &[7]), 128.0);
    }
}
