//! Bounded wake list: FIFO ingress-sequence order, sized to the reservation
//! table so a terminal write (one per slot) can never overflow it.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::reservation::SlotHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WakeEntry {
    pub handle: SlotHandle,
    pub ingress_sequence: u64,
}

struct Inner {
    next_sequence: u64,
    entries: VecDeque<WakeEntry>,
}

#[derive(Debug)]
pub struct WakeList {
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("next_sequence", &self.next_sequence)
            .field("entries_len", &self.entries.len())
            .finish()
    }
}

impl WakeList {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                next_sequence: 0,
                entries: VecDeque::new(),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Mint the next process-local ingress sequence without recording it.
    /// Only useful when the caller does not intend to push a wake entry for
    /// it (e.g. `rebuild`'s monotonicity test); a real terminal submission
    /// should use [`Self::push_next`] so sequence assignment and queue
    /// position stay atomic with each other.
    pub fn next_ingress_sequence(&self) -> u64 {
        let mut inner = self.lock();
        let sequence = inner.next_sequence;
        inner.next_sequence += 1;
        sequence
    }

    /// Assign the next ingress sequence and push `handle` for it as one
    /// atomic step, so concurrent callers can never observe push order
    /// diverge from sequence-assignment order.
    pub fn push_next(&self, handle: SlotHandle) -> u64 {
        let mut inner = self.lock();
        let sequence = inner.next_sequence;
        inner.next_sequence += 1;
        inner.entries.push_back(WakeEntry {
            handle,
            ingress_sequence: sequence,
        });
        sequence
    }

    /// Read the next sequence that would be minted, without consuming it.
    /// Pure observation for the API layer's cursor classification (task
    /// 13): unlike [`Self::next_ingress_sequence`], this never advances the
    /// counter, so calling it repeatedly is side-effect-free.
    #[must_use]
    pub fn peek_next_sequence(&self) -> u64 {
        self.lock().next_sequence
    }

    /// Drain every entry currently queued, oldest first.
    pub fn drain_all(&self) -> Vec<WakeEntry> {
        self.lock().entries.drain(..).collect()
    }

    /// Drain at most `max` entries, oldest first, leaving the rest queued.
    pub fn drain_up_to(&self, max: usize) -> Vec<WakeEntry> {
        let mut inner = self.lock();
        let take = max.min(inner.entries.len());
        inner.entries.drain(..take).collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for WakeList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{OperationId, OperationScope};

    use super::*;
    use crate::runtime_events::reservation::ReservationTable;

    #[test]
    fn drain_returns_entries_in_ingress_sequence_order() {
        let wake = WakeList::new();
        let table = ReservationTable::new(3);
        let handles: Vec<_> = (0..3)
            .map(|_| {
                table
                    .reserve(OperationScope::root_only(OperationId::new()))
                    .unwrap()
            })
            .collect();

        for handle in &handles {
            wake.push_next(*handle);
        }

        let drained = wake.drain_all();
        let sequences: Vec<u64> = drained.iter().map(|entry| entry.ingress_sequence).collect();
        assert_eq!(sequences, vec![0, 1, 2]);
        assert!(wake.is_empty());
    }

    #[test]
    fn drain_up_to_leaves_the_remainder_queued() {
        let wake = WakeList::new();
        let table = ReservationTable::new(2);
        for _ in 0..2 {
            let handle = table
                .reserve(OperationScope::root_only(OperationId::new()))
                .unwrap();
            wake.push_next(handle);
        }

        let first = wake.drain_up_to(1);
        assert_eq!(first.len(), 1);
        assert_eq!(wake.len(), 1);
    }
}
