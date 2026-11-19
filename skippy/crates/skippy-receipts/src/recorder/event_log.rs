//! A turn's replay log: the ubatches that computed its sequence and the
//! removals between them, and what they leave at each position.
//!
//! The native cache names removed positions as `[p0, p1)`, where `p0 < 0`
//! starts at 0 and `p1 < 0` has no end ([`removed`]). Everything that reads
//! a log applies removals through this module, so a position a turn computed
//! and then removed is nowhere read as computed.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use super::UbatchRecord;

#[derive(Clone)]
pub(super) enum Event {
    Ubatch(Arc<UbatchRecord>),
    SeqRm { p0: i32, p1: i32 },
}

impl Event {
    pub(super) fn ubatch(&self) -> Option<&UbatchRecord> {
        match self {
            Self::Ubatch(record) => Some(record),
            Self::SeqRm { .. } => None,
        }
    }

    /// Bytes the event keeps in a log.
    pub(super) fn bytes(&self) -> usize {
        match self {
            Self::Ubatch(record) => record.bytes(),
            Self::SeqRm { .. } => 8,
        }
    }
}

/// The positions a removal of `[p0, p1)` drops.
pub(super) fn removed(p0: i32, p1: i32) -> Range<i32> {
    p0.max(0)..if p1 < 0 { i32::MAX } else { p1 }
}

/// Whether a removal of `[p0, p1)` drops `position`.
pub(super) fn removes(p0: i32, p1: i32, position: u32) -> bool {
    i32::try_from(position).is_ok_and(|position| removed(p0, p1).contains(&position))
}

/// Whether a removal of `[p0, p1)` drops every position.
pub(super) fn removes_all(p0: i32, p1: i32) -> bool {
    p0 <= 0 && p1 < 0
}

/// The column of a log that holds a position: its event and its column in
/// that ubatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Written {
    pub event: usize,
    pub col: usize,
}

/// By position of `seq`, the column that computed it last, for every
/// position no later removal dropped.
pub(super) fn last_writes(events: &[Event], seq: i32) -> BTreeMap<u32, Written> {
    let mut written = BTreeMap::new();
    for (event, entry) in events.iter().enumerate() {
        match entry {
            Event::Ubatch(record) => {
                for (col, (&s, &pos)) in record.seq.iter().zip(&record.pos).enumerate() {
                    if s == seq
                        && let Ok(pos) = u32::try_from(pos)
                    {
                        written.insert(pos, Written { event, col });
                    }
                }
            }
            Event::SeqRm { p0, p1 } => {
                written.retain(|&position, _| !removes(*p0, *p1, position));
            }
        }
    }
    written
}

/// The first position `seq` has in each ubatch of `events` that computed it.
pub(super) fn ubatch_starts(events: &[Event], seq: i32) -> Vec<u32> {
    events
        .iter()
        .filter_map(Event::ubatch)
        .filter_map(|record| {
            record
                .seq
                .iter()
                .zip(&record.pos)
                .filter(|&(&s, _)| s == seq)
                .filter_map(|(_, &p)| u32::try_from(p).ok())
                .min()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ubatch(seq: i32, positions: &[i32]) -> Event {
        Event::Ubatch(Arc::new(UbatchRecord {
            token: positions.iter().map(|p| 100 + p).collect(),
            seq: vec![seq; positions.len()],
            pos: positions.to_vec(),
            output: vec![0; positions.len()],
            cell: positions.iter().map(|&p| 10 + p as u32).collect(),
            n_kv: 64,
            input: None,
        }))
    }

    #[test]
    fn a_removal_reads_its_bounds_as_the_native_cache_does() {
        assert_eq!(removed(-1, -1), 0..i32::MAX);
        assert_eq!(removed(3, -1), 3..i32::MAX);
        assert_eq!(removed(-1, 4), 0..4);
        assert!(removes(2, 4, 2) && removes(2, 4, 3));
        assert!(!removes(2, 4, 1) && !removes(2, 4, 4));
        assert!(removes(2, -1, u32::try_from(i32::MAX - 1).unwrap()));
        assert!(removes_all(-1, -1) && removes_all(0, -1));
        assert!(!removes_all(1, -1) && !removes_all(0, 5));
    }

    // A position a turn computed and removed is not one it computed: it
    // names no ubatch, token or cell. One it computed again names the later
    // ubatch.
    #[test]
    fn a_removed_position_was_not_written() {
        let events = vec![
            ubatch(7, &[0, 1, 2, 3]),
            Event::SeqRm { p0: 2, p1: -1 },
            ubatch(7, &[2]),
            ubatch(9, &[3]),
        ];
        let written = last_writes(&events, 7);
        assert_eq!(written.keys().copied().collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(written[&1], Written { event: 0, col: 1 });
        assert_eq!(written[&2], Written { event: 2, col: 0 });
        assert!(!written.contains_key(&3), "position 3 was removed");
        assert_eq!(last_writes(&events, 9)[&3], Written { event: 3, col: 0 });
        assert_eq!(ubatch_starts(&events, 7), [0, 2]);
    }
}
