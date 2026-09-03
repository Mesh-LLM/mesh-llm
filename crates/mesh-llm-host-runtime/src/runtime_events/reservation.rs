//! The reservation table: the only terminal channel.
//!
//! Each slot owns one write-once terminal record. Admission hands out an
//! index plus a generation counter; a later access whose generation does not
//! match the slot's current generation is treated as late/unreserved rather
//! than corrupting a reused slot.

use std::sync::Mutex;

use mesh_llm_runtime_event_contracts::{OperationId, OperationScope, RuntimeFact};

/// One terminal write for a slot: the fact plus whether it was synthesized
/// by a dropped guard or shutdown rather than submitted by a producer.
#[derive(Debug, Clone)]
pub struct TerminalRecord {
    pub fact: RuntimeFact,
    pub synthesized: bool,
}

#[derive(Debug, Default)]
struct Slot {
    occupant: Option<OperationScope>,
    terminal: Option<TerminalRecord>,
    progress: Option<RuntimeFact>,
    generation: u64,
}

/// Outcome of an admission attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveError {
    Exhausted,
}

/// A `(slot index, generation)` handle. Cheap to copy; used by the engine to
/// address a specific occupancy of a specific slot without holding a guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotHandle {
    pub index: usize,
    pub generation: u64,
}

/// Bounded slab of reservation slots. One write-once terminal per occupied
/// slot; no second terminal lane exists anywhere in this table.
#[derive(Debug)]
pub struct ReservationTable {
    slots: Vec<Mutex<Slot>>,
    free: Mutex<Vec<usize>>,
}

impl ReservationTable {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| Mutex::new(Slot::default())).collect(),
            free: Mutex::new((0..capacity).rev().collect()),
        }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Admit `scope`, returning a fresh `(index, generation)` handle or
    /// [`ReserveError::Exhausted`] when the table is full.
    pub fn reserve(&self, scope: OperationScope) -> Result<SlotHandle, ReserveError> {
        let index = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .ok_or(ReserveError::Exhausted)?;
        let mut slot = self.slot_lock(index);
        slot.occupant = Some(scope);
        slot.terminal = None;
        slot.progress = None;
        slot.generation += 1;
        SlotHandle {
            index,
            generation: slot.generation,
        }
        .pipe_ok()
    }

    /// Write the write-once terminal slot for `handle`. Returns `true` when
    /// this call performed the write, `false` when the handle is stale
    /// (late/unreserved/ID-mismatched) or a terminal was already present
    /// (duplicate).
    pub fn write_terminal(&self, handle: SlotHandle, record: TerminalRecord) -> bool {
        let mut slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation || slot.terminal.is_some() {
            return false;
        }
        slot.terminal = Some(record);
        true
    }

    /// Overwrite the single progress-coalescing slot bound to `handle`.
    /// Returns `false` for a stale handle.
    pub fn coalesce_progress(&self, handle: SlotHandle, fact: RuntimeFact) -> bool {
        let mut slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation {
            return false;
        }
        slot.progress = Some(fact);
        true
    }

    #[must_use]
    pub fn occupant(&self, handle: SlotHandle) -> Option<OperationScope> {
        let slot = self.slot_lock(handle.index);
        (slot.generation == handle.generation).then_some(slot.occupant)?
    }

    #[must_use]
    pub fn has_terminal(&self, handle: SlotHandle) -> bool {
        let slot = self.slot_lock(handle.index);
        slot.generation == handle.generation && slot.terminal.is_some()
    }

    /// Clone out the written terminal record for `handle`, or `None` for a
    /// stale handle or a slot with no terminal written yet.
    #[must_use]
    pub fn terminal_record(&self, handle: SlotHandle) -> Option<TerminalRecord> {
        let slot = self.slot_lock(handle.index);
        if slot.generation != handle.generation {
            return None;
        }
        slot.terminal.clone()
    }

    #[must_use]
    pub fn is_current(&self, handle: SlotHandle) -> bool {
        self.slot_lock(handle.index).generation == handle.generation
    }

    /// Reclaim `handle`'s slot: clear its contents and return the index to
    /// the free list. Advances the slot generation so any outstanding stale
    /// handle (a dropped guard fired after reuse) is provably invalidated.
    pub fn release(&self, handle: SlotHandle) {
        if !self.is_current(handle) {
            return;
        }
        {
            let mut slot = self.slot_lock(handle.index);
            slot.occupant = None;
            slot.terminal = None;
            slot.progress = None;
            // Advance the generation on release itself, not only on reuse,
            // so a guard that drops after a forced release (e.g. a child
            // whose root already released it) sees an immediate mismatch
            // instead of a window where the freed-but-not-yet-reused slot
            // still matches its stale handle.
            slot.generation += 1;
        }
        self.free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(handle.index);
    }

    /// The occupant of `index` at its *current* generation, regardless of
    /// whether the caller's own handle is stale. Used when force-completing
    /// outstanding children on root release.
    #[must_use]
    pub fn is_occupied(&self, index: usize) -> Option<OperationScope> {
        self.slot_lock(index).occupant
    }

    #[must_use]
    pub fn current_generation(&self, index: usize) -> u64 {
        self.slot_lock(index).generation
    }

    fn slot_lock(&self, index: usize) -> std::sync::MutexGuard<'_, Slot> {
        self.slots[index]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

trait PipeOk: Sized {
    fn pipe_ok<E>(self) -> Result<Self, E> {
        Ok(self)
    }
}
impl<T> PipeOk for T {}

#[must_use]
pub fn root_of(scope: OperationScope) -> OperationId {
    scope.root()
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, OperationId, RuntimeFact,
    };

    use super::*;

    fn terminal_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
    }

    #[test]
    fn reserve_then_release_recycles_the_slot() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");
        assert!(matches!(table.reserve(scope), Err(ReserveError::Exhausted)));

        table.release(handle);
        let reused = table.reserve(scope).expect("reserve after release");
        assert_ne!(reused.generation, handle.generation);
    }

    #[test]
    fn second_terminal_write_is_rejected_as_duplicate() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");

        assert!(table.write_terminal(
            handle,
            TerminalRecord {
                fact: terminal_fact(),
                synthesized: false,
            }
        ));
        assert!(!table.write_terminal(
            handle,
            TerminalRecord {
                fact: terminal_fact(),
                synthesized: false,
            }
        ));
    }

    #[test]
    fn stale_handle_after_release_cannot_write_a_terminal() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");
        table.release(handle);
        let _reused = table.reserve(scope).expect("reuse");

        assert!(!table.write_terminal(
            handle,
            TerminalRecord {
                fact: terminal_fact(),
                synthesized: false,
            }
        ));
    }
}
