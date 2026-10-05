//! The producer boundary: one bounded lock-free ring.
//!
//! Every producer -- a decode loop, a request task, the native reporter's
//! driver ingest -- reaches the engine through [`Ingress::push`], which
//! takes no lock at all. The single driver task pops. That is the whole
//! structure: there is no admission gate, no per-class lane a producer can
//! wait on, and no counter minted under a mutex.
//!
//! ## Why a single ring
//!
//! The four delivery classes used to have four separate bounded
//! containers, each with its own mutex, plus a shared sequence counter
//! minted under a process-global `ingress_gate`. That gate existed to make
//! "assign a sequence" and "insert into a container" one atomic step, so
//! the drain could reconstruct a total order across the four containers.
//! It also meant every producer thread in the process serialized on one
//! mutex, and that the drain -- which held the same mutex while
//! snapshotting all four containers -- blocked producers for the length of
//! a pass.
//!
//! With one ring, pop order *is* the order. Class routing happens in the
//! consumer, where it costs a producer nothing, and the sequence is
//! assigned at publication. Both the gate and the cross-lane
//! reconciliation it paid for disappear together.
//!
//! ## Ordering and sequence numbers
//!
//! Sequences are assigned when a fact publishes, not when it is submitted.
//! They are therefore contiguous and monotonic by construction: there is
//! no window in which a sequence has been minted but its fact has not been
//! placed, so no reordering to reconcile and no gap to explain.
//!
//! Under the old scheme every submission consumed a sequence regardless of
//! outcome, so a coalesced or dropped fact left a hole in the published
//! sequence space. That hole was the only evidence a drop had happened,
//! and it did not say what was dropped or why. `runtime_health` already
//! carries `dropped_progress`, `dropped_diagnostic`, and
//! `state_transition_rejected` per class, which is strictly more
//! informative, so the holes are not replaced. Spec §10 requires Rust
//! event IDs to be process-unique and monotonically assigned, which
//! contiguous numbering satisfies; it does not require gaps.
//!
//! ## Capacity and the terminal guarantee
//!
//! A terminal must never be lost to queue pressure -- losing one strands
//! an operation with no outcome. Terminals are therefore exempt from the
//! credit budget:
//!
//! * Non-terminal pushes must take one of [`NON_TERMINAL_CREDITS`],
//!   returned when the consumer pops the entry.
//! * Terminal-class items take no credit. At most one terminal can exist
//!   per occupied reservation slot, because claiming the slot's
//!   write-once terminal flag is what admits it, so terminals in flight
//!   are bounded by `RESERVATION_TABLE_CAPACITY`.
//! * [`RING_CAPACITY`] is the sum of the two.
//!
//! A full ring on a terminal push is therefore unreachable by
//! construction rather than by assumption, and the impossible case still
//! trips a `debug_assert` rather than being silently swallowed.

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_queue::ArrayQueue;
use mesh_llm_runtime_event_contracts::{DeliveryClass, OperationScope, RuntimeFact};

use super::config::{
    DIAGNOSTIC_LANE_DEPTH, RESERVATION_TABLE_CAPACITY, STATE_TRANSITION_LANE_DEPTH,
};
use super::reservation::SlotHandle;

/// Non-terminal entries that may be in flight at once. Terminals do not
/// draw on this.
///
/// Derived from the frozen per-class bounds rather than picked: the
/// previous design gave state transitions and diagnostics their own
/// bounded lanes, and one shared budget must not admit less than the sum
/// of what those two promised. Progress draws on the same budget and used
/// to have no lane of its own at all, so it can only gain.
pub const NON_TERMINAL_CREDITS: usize = STATE_TRANSITION_LANE_DEPTH + DIAGNOSTIC_LANE_DEPTH;

/// Entries the ring can hold at once.
///
/// Also derived, not picked: the credit budget plus the most terminals
/// that can exist at once. Sizing the ring this way is what makes a full
/// ring on a terminal push structurally unreachable rather than merely
/// unlikely.
pub const RING_CAPACITY: usize = RESERVATION_TABLE_CAPACITY + NON_TERMINAL_CREDITS;

/// One submitted fact, with everything the consumer needs to route,
/// validate, and publish it.
#[derive(Debug)]
pub(crate) struct IngressFact {
    pub(crate) scope: OperationScope,
    pub(crate) fact: RuntimeFact,
    /// The reservation this was submitted through, if any.
    pub(crate) handle: Option<SlotHandle>,
    /// Whether the submission was reservation-bound. Distinct from
    /// `handle.is_some()` only in that it is what the reducer is told.
    pub(crate) reserved: bool,
    /// Whether the engine synthesized this terminal rather than a producer
    /// submitting it.
    pub(crate) synthesized: bool,
    pub(crate) class: DeliveryClass,
}

/// What a producer can put in the ring.
///
/// The two variants differ sharply in size, because a `RuntimeFact`
/// carries its family payload inline. Boxing it would even them out and
/// shrink the ring, at the cost of a heap allocation on the producer path
/// -- which is the one thing this boundary is built to avoid. The ring is
/// sized once at construction; `tests::footprint_is_reported_so_a_growing_fact_cannot_silently_double_it`
/// states the budget so a growing fact is a deliberate decision rather
/// than a silent one.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum IngressItem {
    Fact(IngressFact),
    /// A cancelled operation whose slot has already been released, asking
    /// the drain to evict its reducer state.
    ///
    /// Cancelling releases the slot on the calling thread -- the
    /// reservation table and its indices are all producer-reachable -- but
    /// eviction needs the publication and reducer locks, which a drain
    /// pass holds for as long as its work takes. Queuing it also puts the
    /// eviction behind facts submitted for the same scope moments earlier
    /// that are still in the ring, instead of evicting state those facts
    /// would then re-create.
    Released {
        scope: OperationScope,
    },
}

impl IngressItem {
    /// Whether this item is an undelivered terminal. Shutdown reports
    /// these separately, because a terminal left queued means an operation
    /// ended with no outcome on the stream.
    fn is_terminal_fact(&self) -> bool {
        matches!(self, Self::Fact(fact) if fact.class == DeliveryClass::Terminal)
    }

    /// Whether this item is exempt from the credit budget.
    fn is_terminal_class(&self) -> bool {
        match self {
            // Bounded the same way a terminal is -- at most one per
            // occupied slot -- and losing one would strand reducer state
            // for an operation that has already gone away.
            Self::Released { .. } => true,
            Self::Fact(fact) => fact.class == DeliveryClass::Terminal,
        }
    }
}

/// Why a push did not make it into the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PushError {
    /// The non-terminal credit budget is exhausted, or the ring itself is
    /// full. The caller maps this to its class's capacity outcome.
    Full,
}

/// The bounded lock-free ring plus its credit budget.
#[derive(Debug)]
pub(crate) struct Ingress {
    ring: ArrayQueue<IngressItem>,
    /// Remaining non-terminal credits. Decremented before a push, restored
    /// if the push fails, and returned when the consumer pops.
    credits: AtomicUsize,
    /// Terminal facts currently waiting for the consumer. Tracked rather
    /// than derived, because the ring cannot be inspected by class, and
    /// shutdown has to report undelivered terminals separately from other
    /// queued work.
    queued_terminals: AtomicUsize,
}

impl Ingress {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::with_capacity(RING_CAPACITY, NON_TERMINAL_CREDITS)
    }

    /// Test seam for capacity behavior. Production uses [`Self::new`].
    #[must_use]
    pub(crate) fn with_capacity(ring_capacity: usize, credits: usize) -> Self {
        Self {
            ring: ArrayQueue::new(ring_capacity.max(1)),
            credits: AtomicUsize::new(credits),
            queued_terminals: AtomicUsize::new(0),
        }
    }

    /// Place `item` in the ring. Never blocks and never allocates.
    ///
    /// Terminal-class items bypass the credit budget; see the module
    /// documentation for why that cannot overflow the ring.
    pub(crate) fn push(&self, item: IngressItem) -> Result<(), PushError> {
        let terminal_fact = item.is_terminal_fact();
        if item.is_terminal_class() {
            return self
                .ring
                .push(item)
                .map(|()| {
                    if terminal_fact {
                        self.queued_terminals.fetch_add(1, Ordering::AcqRel);
                    }
                })
                .map_err(|_| {
                    // Unreachable by the capacity arithmetic asserted
                    // above. Loud in a debug build rather than assumed
                    // away; in release the caller still reports it through
                    // `terminal_delivery_failed`, which is on the wire.
                    debug_assert!(
                        false,
                        "a terminal push found the ring full, which the \
                         capacity budget is supposed to make impossible"
                    );
                    PushError::Full
                });
        }
        if !self.take_credit() {
            return Err(PushError::Full);
        }
        self.ring.push(item).map_err(|_| {
            self.return_credits(1);
            PushError::Full
        })
    }

    /// Pop up to `max` items in push order, returning the credits the
    /// non-terminal ones were holding.
    ///
    /// Single-consumer by contract: only the driver's drain pass calls
    /// this, serialized by `drain_gate`.
    pub(crate) fn pop_up_to(&self, max: usize, out: &mut Vec<IngressItem>) {
        let mut credits_returned = 0usize;
        let mut terminals_popped = 0usize;
        while out.len() < max {
            let Some(item) = self.ring.pop() else {
                break;
            };
            if item.is_terminal_fact() {
                terminals_popped += 1;
            } else if !item.is_terminal_class() {
                credits_returned += 1;
            }
            out.push(item);
        }
        if credits_returned > 0 {
            self.return_credits(credits_returned);
        }
        if terminals_popped > 0 {
            self.queued_terminals
                .fetch_sub(terminals_popped, Ordering::AcqRel);
        }
    }

    /// Terminal facts currently waiting for the consumer.
    #[must_use]
    pub(crate) fn queued_terminals(&self) -> usize {
        self.queued_terminals.load(Ordering::Acquire)
    }

    /// Items currently waiting for the consumer.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.ring.len()
    }

    fn take_credit(&self) -> bool {
        self.credits
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |available| {
                available.checked_sub(1)
            })
            .is_ok()
    }

    fn return_credits(&self, count: usize) {
        self.credits.fetch_add(count, Ordering::AcqRel);
    }
}

impl Default for Ingress {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, OperationId, RequestEventKind,
    };

    use super::*;

    fn item(class: DeliveryClass) -> IngressItem {
        IngressItem::Fact(IngressFact {
            scope: OperationScope::root_only(OperationId::new()),
            fact: match class {
                DeliveryClass::Terminal => {
                    RuntimeFact::Request(FamilyFact::new(RequestEventKind::RequestCompleted))
                }
                _ => RuntimeFact::NativeRuntime(FamilyFact::new(
                    NativeRuntimeEventKind::RuntimeInitialized,
                )),
            },
            handle: None,
            reserved: false,
            synthesized: false,
            class,
        })
    }

    #[test]
    fn footprint_is_reported_so_a_growing_fact_cannot_silently_double_it() {
        let entry = std::mem::size_of::<IngressItem>();
        let total = entry * RING_CAPACITY;
        assert!(
            total <= 8 * 1024 * 1024,
            "the ring is {total} bytes ({entry} per entry). It is allocated \
             once per engine at construction, which is why the budget is \
             stated rather than discovered: if RuntimeFact has grown enough \
             to breach this, decide deliberately between a smaller ring and \
             a larger budget."
        );
    }

    /// Non-terminal pushes are bounded by the credit budget, and draining
    /// returns the credits so the ring recovers instead of wedging.
    #[test]
    fn non_terminal_pushes_are_bounded_by_credits_and_recover_on_pop() {
        let ingress = Ingress::with_capacity(16, 4);

        for _ in 0..4 {
            assert_eq!(ingress.push(item(DeliveryClass::Diagnostic)), Ok(()));
        }
        assert_eq!(
            ingress.push(item(DeliveryClass::Diagnostic)),
            Err(PushError::Full)
        );

        let mut popped = Vec::new();
        ingress.pop_up_to(usize::MAX, &mut popped);
        assert_eq!(popped.len(), 4);
        assert_eq!(ingress.push(item(DeliveryClass::Diagnostic)), Ok(()));
    }

    /// The terminal guarantee: a terminal is admitted even with every
    /// non-terminal credit spent, because losing one strands an operation
    /// with no outcome.
    #[test]
    fn a_terminal_is_admitted_with_the_credit_budget_exhausted() {
        let ingress = Ingress::with_capacity(16, 2);

        for _ in 0..2 {
            assert_eq!(ingress.push(item(DeliveryClass::StateTransition)), Ok(()));
        }
        assert_eq!(
            ingress.push(item(DeliveryClass::Progress)),
            Err(PushError::Full),
            "the credit budget is spent"
        );
        assert_eq!(
            ingress.push(item(DeliveryClass::Terminal)),
            Ok(()),
            "a terminal takes no credit"
        );
        assert_eq!(ingress.queued_terminals(), 1);
    }

    /// Popping tracks terminal facts separately, because shutdown reports
    /// an undelivered terminal as a stranded operation rather than as
    /// ordinary queued work.
    #[test]
    fn queued_terminals_tracks_only_terminal_facts() {
        let ingress = Ingress::with_capacity(16, 8);
        ingress.push(item(DeliveryClass::Terminal)).expect("push");
        ingress
            .push(IngressItem::Released {
                scope: OperationScope::root_only(OperationId::new()),
            })
            .expect("push");
        ingress.push(item(DeliveryClass::Diagnostic)).expect("push");

        assert_eq!(ingress.queued_terminals(), 1);
        assert_eq!(ingress.len(), 3);

        let mut popped = Vec::new();
        ingress.pop_up_to(usize::MAX, &mut popped);
        assert_eq!(ingress.queued_terminals(), 0);
    }
}
