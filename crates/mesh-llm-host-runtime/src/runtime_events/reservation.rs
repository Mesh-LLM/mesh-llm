//! The reservation table: the only terminal channel.
//!
//! Each slot owns one write-once terminal claim. Admission hands out an
//! index plus a generation counter; a later access whose generation does
//! not match the slot's current generation is treated as late/unreserved
//! rather than corrupting a reused slot.
//!
//! ## Why a slot is split in two
//!
//! A producer submitting a fact has to know three things about its slot:
//! is this handle still current, is the slot occupied, and has it been
//! cancelled. It must learn them without taking a lock -- a producer that
//! blocks to emit an event is the defect this subsystem exists to avoid.
//!
//! So a slot is a lock-free [`SlotHeader`] (generation plus a flags word)
//! and a mutex-protected [`SlotPayload`] (the occupant scope, the family
//! terminal synthesizer, the remembered identities). Every producer-path
//! question is answered from the header with atomic loads. The payload is
//! touched only when reserving, releasing, or synthesizing -- once per
//! operation, never once per event.
//!
//! ## The write-once terminal
//!
//! The terminal fact itself is no longer stored in the slot. Claiming
//! [`TERMINAL_CLAIMED`] with a compare-and-swap is what makes a terminal
//! admissible; the winner then places the fact in the ingress ring and
//! every later claimant is told `TerminalDeliveryFailed`. A producer
//! submitting a terminal, a dropped guard synthesizing one, and shutdown
//! settling an unsettled reservation all go through that same CAS, so two
//! of them cannot both win.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FamilyFact, OperationScope, RuntimeFact, ScopeIdentities,
};

use crate::runtime_events::lock_audit::{AuditedMutex, LockClass};

fn default_synthetic_terminal() -> RuntimeFact {
    RuntimeFact::Diagnostic(FamilyFact::new(
        DiagnosticEventKind::InvariantProtocolViolation,
    ))
}

/// The slot holds a live reservation.
const OCCUPIED: u32 = 1 << 0;
/// The reservation was explicitly cancelled. The slot can stay occupied
/// after this when a root's release is deferred behind a live child, so
/// producer paths must keep consulting this bit until `release` advances
/// the generation.
const CANCELLED: u32 = 1 << 1;
/// Some caller has won the write-once terminal claim for this generation.
const TERMINAL_CLAIMED: u32 = 1 << 2;

/// Lock-free slot header. Every question a producer asks about its
/// reservation is answered from here.
///
/// `generation` is read before and after the flags so a reader can detect
/// a concurrent release-and-reuse rather than acting on flags belonging to
/// a different occupancy.
#[derive(Debug, Default)]
struct SlotHeader {
    generation: AtomicU64,
    flags: AtomicU32,
}

impl SlotHeader {
    /// The flags for `generation`, or `None` when the slot has moved on.
    fn flags_for(&self, generation: u64) -> Option<u32> {
        if self.generation.load(Ordering::Acquire) != generation {
            return None;
        }
        let flags = self.flags.load(Ordering::Acquire);
        // Re-read: a release between the two loads would have advanced the
        // generation, so a matching second read proves the flags belong to
        // the generation the caller asked about.
        (self.generation.load(Ordering::Acquire) == generation).then_some(flags)
    }
}

/// Mutex-protected slot contents. Touched once per operation by reserve,
/// release, cancel, and synthesis -- never on the per-event submit path.
#[derive(Debug, Default)]
struct SlotPayload {
    occupant: Option<OperationScope>,
    /// Family-correct terminal constructor retained with the occupancy so
    /// shutdown can settle a live guard even though the guard itself may
    /// be held by another producer thread.
    synthetic_terminal: Option<fn() -> RuntimeFact>,
    /// Last non-empty scope supplied by an accepted submission for this
    /// generation. Shutdown and guard-drop synthesis use this to retain
    /// typed identities while the producer guard is still held.
    scope_identities: Option<ScopeIdentities>,
}

#[derive(Debug)]
struct Slot {
    header: SlotHeader,
    payload: AuditedMutex<SlotPayload>,
}

impl Slot {
    fn new() -> Self {
        Self {
            header: SlotHeader::default(),
            payload: AuditedMutex::new(LockClass::ReservationSlot, SlotPayload::default()),
        }
    }
}

/// Outcome of an admission attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveError {
    Exhausted,
}

/// Outcome of a write-once terminal claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalClaim {
    /// This caller won the claim and owns placing the terminal.
    Claimed,
    /// Another caller already claimed it, or the handle is stale,
    /// unoccupied, or cancelled.
    Refused,
}

/// A `(slot index, generation)` handle. Cheap to copy; used by the engine
/// to address a specific occupancy of a specific slot without holding a
/// guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotHandle {
    pub index: usize,
    pub generation: u64,
}

/// An occupied slot without a terminal, captured for shutdown synthesis.
#[derive(Clone)]
pub(crate) struct UnsettledReservation {
    pub(crate) handle: SlotHandle,
    pub(crate) scope: OperationScope,
    pub(crate) synthetic_terminal: fn() -> RuntimeFact,
    pub(crate) scope_identities: Option<ScopeIdentities>,
}

/// Bounded slab of reservation slots. One write-once terminal claim per
/// occupied slot; no second terminal lane exists anywhere in this table.
#[derive(Debug)]
pub struct ReservationTable {
    slots: Vec<Slot>,
    free: AuditedMutex<Vec<usize>>,
}

impl ReservationTable {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| Slot::new()).collect(),
            free: AuditedMutex::new(
                LockClass::ReservationFreeList,
                (0..capacity).rev().collect(),
            ),
        }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Admit `scope`, returning a fresh `(index, generation)` handle or
    /// [`ReserveError::Exhausted`] when the table is full.
    pub fn reserve(&self, scope: OperationScope) -> Result<SlotHandle, ReserveError> {
        self.reserve_with_synthesizer(scope, default_synthetic_terminal)
    }

    /// Reserve a slot while retaining the family-provided synthesizer for
    /// shutdown settlement. The plain `reserve` helper remains available
    /// for table-only tests and callers that do not need the engine's
    /// synthesis contract.
    pub fn reserve_with_synthesizer(
        &self,
        scope: OperationScope,
        synthetic_terminal: fn() -> RuntimeFact,
    ) -> Result<SlotHandle, ReserveError> {
        let index = self.free.lock().pop().ok_or(ReserveError::Exhausted)?;
        let slot = &self.slots[index];
        let mut payload = slot.payload.lock();
        payload.occupant = Some(scope);
        payload.synthetic_terminal = Some(synthetic_terminal);
        payload.scope_identities = None;
        // Publish the new occupancy through the header last: a producer
        // reading the header sees either the old generation (and rejects
        // its stale handle) or the new one with OCCUPIED already set.
        let generation = slot.header.generation.fetch_add(1, Ordering::AcqRel) + 1;
        slot.header.flags.store(OCCUPIED, Ordering::Release);
        drop(payload);
        Ok(SlotHandle { index, generation })
    }

    /// Win or lose the write-once terminal claim for `handle`.
    ///
    /// Lock-free: this is on the producer path. A stale, unoccupied, or
    /// cancelled slot refuses, as does a second claim on the same
    /// generation.
    pub fn claim_terminal(&self, handle: SlotHandle) -> TerminalClaim {
        let Some(slot) = self.slots.get(handle.index) else {
            return TerminalClaim::Refused;
        };
        loop {
            let Some(flags) = slot.header.flags_for(handle.generation) else {
                return TerminalClaim::Refused;
            };
            if flags & OCCUPIED == 0 || flags & (CANCELLED | TERMINAL_CLAIMED) != 0 {
                return TerminalClaim::Refused;
            }
            match slot.header.flags.compare_exchange_weak(
                flags,
                flags | TERMINAL_CLAIMED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                // Re-check the generation: a release between the flag read
                // and the swap would have reset the flags to zero, which
                // this CAS could otherwise have matched.
                Ok(_) => {
                    return if slot.header.generation.load(Ordering::Acquire) == handle.generation {
                        TerminalClaim::Claimed
                    } else {
                        TerminalClaim::Refused
                    };
                }
                Err(_) => continue,
            }
        }
    }

    /// Whether `handle` is live, occupied, and not cancelled -- the one
    /// question a non-terminal submit has to answer. Lock-free.
    #[must_use]
    pub fn is_live(&self, handle: SlotHandle) -> bool {
        self.slots.get(handle.index).is_some_and(|slot| {
            slot.header
                .flags_for(handle.generation)
                .is_some_and(|flags| flags & OCCUPIED != 0 && flags & CANCELLED == 0)
        })
    }

    /// Remember a non-empty scope from an accepted submission for this
    /// generation. A stale, cancelled, or unoccupied handle is rejected.
    ///
    /// Takes the payload lock, so this is NOT a producer-path call: the
    /// drain records identities when it applies a fact.
    pub fn remember_scope(&self, handle: SlotHandle, scope: ScopeIdentities) -> bool {
        if scope == ScopeIdentities::default() {
            return false;
        }
        let Some(slot) = self.slots.get(handle.index) else {
            return false;
        };
        if !self.is_live(handle) {
            return false;
        }
        let mut payload = slot.payload.lock();
        if slot.header.generation.load(Ordering::Acquire) != handle.generation {
            return false;
        }
        payload.scope_identities = Some(scope);
        true
    }

    #[must_use]
    pub fn occupant(&self, handle: SlotHandle) -> Option<OperationScope> {
        let slot = self.slots.get(handle.index)?;
        let flags = slot.header.flags_for(handle.generation)?;
        if flags & OCCUPIED == 0 {
            return None;
        }
        let payload = slot.payload.lock();
        (slot.header.generation.load(Ordering::Acquire) == handle.generation)
            .then_some(payload.occupant)
            .flatten()
    }

    /// Return the last non-empty scope remembered for this live generation.
    #[must_use]
    pub fn scope_identities(&self, handle: SlotHandle) -> Option<ScopeIdentities> {
        let slot = self.slots.get(handle.index)?;
        let flags = slot.header.flags_for(handle.generation)?;
        if flags & OCCUPIED == 0 {
            return None;
        }
        let payload = slot.payload.lock();
        (slot.header.generation.load(Ordering::Acquire) == handle.generation)
            .then(|| payload.scope_identities.clone())
            .flatten()
    }

    /// Whether the write-once terminal claim for `handle` has been taken.
    #[must_use]
    pub fn has_terminal(&self, handle: SlotHandle) -> bool {
        self.slots.get(handle.index).is_some_and(|slot| {
            slot.header
                .flags_for(handle.generation)
                .is_some_and(|flags| flags & TERMINAL_CLAIMED != 0)
        })
    }

    #[must_use]
    pub fn is_current(&self, handle: SlotHandle) -> bool {
        self.slots
            .get(handle.index)
            .is_some_and(|slot| slot.header.generation.load(Ordering::Acquire) == handle.generation)
    }

    /// Mark a live reservation generation as cancelled. The slot remains
    /// occupied when root release is deferred behind children, so ingress
    /// paths must consult this bit until `release` advances the generation.
    ///
    /// Lock-free: cancelling is called from a producer thread.
    pub fn mark_cancelled(&self, handle: SlotHandle) -> bool {
        let Some(slot) = self.slots.get(handle.index) else {
            return false;
        };
        loop {
            let Some(flags) = slot.header.flags_for(handle.generation) else {
                return false;
            };
            if flags & OCCUPIED == 0 {
                return false;
            }
            if flags & CANCELLED != 0 {
                return false;
            }
            if slot
                .header
                .flags
                .compare_exchange_weak(
                    flags,
                    flags | CANCELLED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return slot.header.generation.load(Ordering::Acquire) == handle.generation;
            }
        }
    }

    #[must_use]
    pub fn is_cancelled(&self, handle: SlotHandle) -> bool {
        self.slots.get(handle.index).is_some_and(|slot| {
            slot.header
                .flags_for(handle.generation)
                .is_some_and(|flags| flags & CANCELLED != 0)
        })
    }

    /// Reclaim `handle`'s slot: clear its contents and return the index to
    /// the free list. Advances the slot generation so any outstanding
    /// stale handle (a dropped guard fired after reuse) is provably
    /// invalidated.
    pub fn release(&self, handle: SlotHandle) {
        let Some(slot) = self.slots.get(handle.index) else {
            return;
        };
        let mut payload = slot.payload.lock();
        if slot.header.generation.load(Ordering::Acquire) != handle.generation {
            return;
        }
        payload.occupant = None;
        payload.synthetic_terminal = None;
        payload.scope_identities = None;
        // Clear the flags BEFORE advancing the generation: a reader that
        // sees the old generation must never see stale flags alongside it,
        // and one that sees the new generation re-reads anyway.
        slot.header.flags.store(0, Ordering::Release);
        // Advance the generation on release itself, not only on reuse, so
        // a guard that drops after a forced release (e.g. a child whose
        // root already released it) sees an immediate mismatch instead of
        // a window where the freed-but-not-yet-reused slot still matches
        // its stale handle.
        slot.header.generation.fetch_add(1, Ordering::AcqRel);

        // Keep the payload lock held through the free-list insertion. This
        // makes generation validation, cleanup, and reclamation one atomic
        // operation with respect to duplicate/stale releases. `reserve`
        // drops its free-list guard before taking any payload lock, so
        // this payload-then-free order cannot form an ABBA cycle.
        self.free.lock().push(handle.index);
    }

    /// The occupant of `index` at its *current* generation, regardless of
    /// whether the caller's own handle is stale. Used when force-completing
    /// outstanding children on root release.
    #[must_use]
    pub fn is_occupied(&self, index: usize) -> Option<OperationScope> {
        let slot = self.slots.get(index)?;
        if slot.header.flags.load(Ordering::Acquire) & OCCUPIED == 0 {
            return None;
        }
        slot.payload.lock().occupant
    }

    /// Whether any live, non-cancelled generation currently owns `scope`.
    /// Drain-owned lane entries retain only the scope and reservation
    /// provenance; this bounded scan lets the drain discard a fact queued
    /// before cancellation without allowing it to resurrect reducer state.
    #[must_use]
    pub fn has_active_scope(&self, scope: OperationScope) -> bool {
        self.slots.iter().any(|slot| {
            let flags = slot.header.flags.load(Ordering::Acquire);
            if flags & OCCUPIED == 0 || flags & CANCELLED != 0 {
                return false;
            }
            slot.payload.lock().occupant == Some(scope)
        })
    }

    #[must_use]
    pub fn occupied_len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.header.flags.load(Ordering::Acquire) & OCCUPIED != 0)
            .count()
    }

    #[must_use]
    pub fn current_generation(&self, index: usize) -> u64 {
        self.slots[index].header.generation.load(Ordering::Acquire)
    }

    /// Snapshot every occupied slot whose terminal claim is still
    /// available, retaining the generation and original family
    /// synthesizer. The engine calls this only after admission closes, so
    /// the collection cannot race a new reserve.
    #[must_use]
    pub(crate) fn unsettled(&self) -> Vec<UnsettledReservation> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let generation = slot.header.generation.load(Ordering::Acquire);
                let flags = slot.header.flags.load(Ordering::Acquire);
                if flags & OCCUPIED == 0 || flags & (CANCELLED | TERMINAL_CLAIMED) != 0 {
                    return None;
                }
                let payload = slot.payload.lock();
                Some(UnsettledReservation {
                    handle: SlotHandle { index, generation },
                    scope: payload.occupant?,
                    synthetic_terminal: payload.synthetic_terminal?,
                    scope_identities: payload.scope_identities.clone(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use mesh_llm_runtime_event_contracts::OperationId;

    use super::*;

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
    fn a_second_terminal_claim_is_refused_as_duplicate() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");

        assert_eq!(table.claim_terminal(handle), TerminalClaim::Claimed);
        assert_eq!(table.claim_terminal(handle), TerminalClaim::Refused);
        assert!(table.has_terminal(handle));
    }

    #[test]
    fn a_stale_handle_after_release_cannot_claim_a_terminal() {
        let table = ReservationTable::new(1);
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");
        table.release(handle);
        let _reused = table.reserve(scope).expect("reuse");

        assert_eq!(table.claim_terminal(handle), TerminalClaim::Refused);
    }

    /// The whole point of the write-once claim: under real contention on
    /// the same slot, exactly one caller may own the terminal.
    #[test]
    fn exactly_one_of_many_concurrent_claimants_wins() {
        const CLAIMANTS: usize = 16;

        let table = Arc::new(ReservationTable::new(1));
        let handle = table
            .reserve(OperationScope::root_only(OperationId::new()))
            .expect("reserve");
        let barrier = Arc::new(Barrier::new(CLAIMANTS));
        let winners = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for _ in 0..CLAIMANTS {
                let table = Arc::clone(&table);
                let barrier = Arc::clone(&barrier);
                let winners = Arc::clone(&winners);
                scope.spawn(move || {
                    barrier.wait();
                    if table.claim_terminal(handle) == TerminalClaim::Claimed {
                        winners.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });

        assert_eq!(winners.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_cancelled_reservation_refuses_a_terminal_claim_and_reports_not_live() {
        let table = ReservationTable::new(1);
        let handle = table
            .reserve(OperationScope::root_only(OperationId::new()))
            .expect("reserve");

        assert!(table.is_live(handle));
        assert!(table.mark_cancelled(handle));
        assert!(!table.is_live(handle));
        assert!(table.is_cancelled(handle));
        assert_eq!(table.claim_terminal(handle), TerminalClaim::Refused);
        assert!(
            !table.mark_cancelled(handle),
            "cancelling twice must report that this call did not do it"
        );
    }

    #[test]
    fn concurrent_duplicate_releases_return_one_slot_to_the_free_list() {
        let table = Arc::new(ReservationTable::new(1));
        let scope = OperationScope::root_only(OperationId::new());
        let handle = table.reserve(scope).expect("reserve");
        let barrier = Arc::new(Barrier::new(2));

        thread::scope(|scope| {
            for _ in 0..2 {
                let table = Arc::clone(&table);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    table.release(handle);
                });
            }
        });

        let reused = table.reserve(OperationScope::root_only(OperationId::new()));
        assert!(reused.is_ok(), "one release must return the slot");
        assert!(
            matches!(
                table.reserve(OperationScope::root_only(OperationId::new())),
                Err(ReserveError::Exhausted)
            ),
            "a duplicate release must not put the same index on the free list twice"
        );
    }

    #[test]
    fn stale_release_after_slot_reuse_cannot_free_the_new_occupant() {
        let table = ReservationTable::new(1);
        let first = table
            .reserve(OperationScope::root_only(OperationId::new()))
            .expect("first reserve");
        table.release(first);
        let second = table
            .reserve(OperationScope::root_only(OperationId::new()))
            .expect("reuse");

        table.release(first);

        assert!(
            matches!(
                table.reserve(OperationScope::root_only(OperationId::new())),
                Err(ReserveError::Exhausted)
            ),
            "a stale handle must not free a newer generation's occupant"
        );
        assert!(table.is_current(second));
    }

    /// Liveness has to be answerable without the payload lock, because a
    /// producer asks it on every submit. A held payload lock must not make
    /// `is_live`, `is_cancelled`, `has_terminal`, or `claim_terminal`
    /// wait.
    #[test]
    fn producer_path_queries_do_not_touch_the_payload_lock() {
        let table = ReservationTable::new(1);
        let handle = table
            .reserve(OperationScope::root_only(OperationId::new()))
            .expect("reserve");

        let held = table.slots[0].payload.lock();
        assert!(table.is_live(handle));
        assert!(!table.is_cancelled(handle));
        assert!(!table.has_terminal(handle));
        assert_eq!(table.claim_terminal(handle), TerminalClaim::Claimed);
        assert!(table.has_terminal(handle));
        drop(held);
    }
}
