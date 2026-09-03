//! The host runtime-event engine: admission, the write-once terminal slot,
//! and the minimal acknowledgement seam a reducer (task 4) drains.

mod drain;
mod lanes;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mesh_llm_runtime_event_contracts::{
    OperationId, OperationScope, ProcessInstanceId, RuntimeEventIngress, RuntimeFact, SubmitOutcome,
};

use super::config::RESERVATION_TABLE_CAPACITY;
use super::health::EngineHealth;
use super::reducer::ReducerSnapshot;
use super::replay::ReplayBuffer;
use super::reservation::{ReservationTable, SlotHandle};
use super::subscribers::SubscriberRegistry;
use super::wake::WakeList;

use lanes::{DiagnosticLane, StateLane};

/// Builds the family-correct synthesized terminal fact for a dropped guard
/// or shutdown. Engine layer stays family-agnostic; callers (family
/// adapters in later tasks) supply the right `Terminal`-class kind with
/// `outcome: Unknown` and `reason: TerminalNotDelivered` already set.
pub type SyntheticTerminal = fn() -> RuntimeFact;

pub struct RuntimeEventEngine {
    table: ReservationTable,
    wake: WakeList,
    replay: ReplayBuffer,
    subscribers: SubscriberRegistry,
    health: EngineHealth,
    children_by_root: Mutex<HashMap<OperationId, Vec<usize>>>,
    shutting_down: AtomicBool,
    rebuild_generation: AtomicU64,
    state_lane: StateLane,
    diagnostic_lane: DiagnosticLane,
    reducer_state: Mutex<Arc<ReducerSnapshot>>,
    process_instance: ProcessInstanceId,
}

impl RuntimeEventEngine {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_capacity(RESERVATION_TABLE_CAPACITY)
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            table: ReservationTable::new(capacity),
            wake: WakeList::new(),
            replay: ReplayBuffer::new(),
            subscribers: SubscriberRegistry::new(),
            health: EngineHealth::default(),
            children_by_root: Mutex::new(HashMap::new()),
            shutting_down: AtomicBool::new(false),
            rebuild_generation: AtomicU64::new(0),
            state_lane: StateLane::default(),
            diagnostic_lane: DiagnosticLane::default(),
            reducer_state: Mutex::new(ReducerSnapshot::empty()),
            process_instance: ProcessInstanceId::new(),
        })
    }

    /// This engine's process-local identity: the first component of the
    /// wire cursor grammar `rt1:<process-instance-uuid>:<sequence>`. Minted
    /// once per engine instance and never changes for its lifetime.
    #[must_use]
    pub fn process_instance(&self) -> ProcessInstanceId {
        self.process_instance
    }

    /// The highest ingress sequence ever minted by this engine's wake list,
    /// or `None` if no terminal-class fact has been submitted yet. Read-only
    /// (never advances the counter): used by the v1 SSE route to classify a
    /// resumption cursor as in-window, evicted, or an out-of-range future
    /// sequence, without consuming a real sequence number to do it.
    #[must_use]
    pub fn highest_known_sequence(&self) -> Option<u64> {
        let next = self.wake.peek_next_sequence();
        (next > 0).then(|| next - 1)
    }

    /// Snapshot of the reducer's current, fully-applied state. Cheap: an
    /// `Arc` clone, never a copy of the underlying map.
    #[must_use]
    pub fn reducer_snapshot(&self) -> Arc<ReducerSnapshot> {
        Arc::clone(
            &self
                .reducer_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    pub(super) fn reducer_state(&self) -> &Mutex<Arc<ReducerSnapshot>> {
        &self.reducer_state
    }

    pub(super) fn state_lane(&self) -> &StateLane {
        &self.state_lane
    }

    pub(super) fn diagnostic_lane(&self) -> &DiagnosticLane {
        &self.diagnostic_lane
    }

    #[must_use]
    pub fn health(&self) -> &EngineHealth {
        &self.health
    }

    #[must_use]
    pub fn replay(&self) -> &ReplayBuffer {
        &self.replay
    }

    #[must_use]
    pub fn subscribers(&self) -> &SubscriberRegistry {
        &self.subscribers
    }

    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    pub fn reserve_root(
        self: &Arc<Self>,
        operation: OperationId,
        synthetic_terminal: SyntheticTerminal,
    ) -> Option<OperationReservation> {
        self.reserve_scope(OperationScope::root_only(operation), synthetic_terminal)
    }

    pub fn reserve_child(
        self: &Arc<Self>,
        root: OperationId,
        child: mesh_llm_runtime_event_contracts::ChildOperationId,
        synthetic_terminal: SyntheticTerminal,
    ) -> Option<OperationReservation> {
        self.reserve_scope(OperationScope::with_child(root, child), synthetic_terminal)
    }

    fn reserve_scope(
        self: &Arc<Self>,
        scope: OperationScope,
        synthetic_terminal: SyntheticTerminal,
    ) -> Option<OperationReservation> {
        match self.table.reserve(scope) {
            Ok(handle) => {
                if let OperationScope::Child { root, .. } = scope {
                    self.children_by_root
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .entry(root)
                        .or_default()
                        .push(handle.index);
                }
                Some(OperationReservation {
                    engine: Arc::clone(self),
                    scope,
                    handle,
                    synthetic_terminal,
                })
            }
            Err(super::reservation::ReserveError::Exhausted) => {
                self.health.bump_reservation_exhausted();
                None
            }
        }
    }

    fn submit(
        &self,
        scope: OperationScope,
        handle: Option<SlotHandle>,
        fact: RuntimeFact,
    ) -> SubmitOutcome {
        use mesh_llm_runtime_event_contracts::DeliveryClass;
        match fact.delivery_class() {
            DeliveryClass::Terminal => lanes::submit_terminal(self, scope, handle, fact),
            DeliveryClass::Progress => lanes::submit_progress(self, handle, fact),
            DeliveryClass::StateTransition => lanes::submit_state_transition(self, fact),
            DeliveryClass::Diagnostic => lanes::submit_diagnostic(self, fact),
        }
    }

    /// A `RuntimeEventIngress` bound to `scope` with no slot. Used by an
    /// exhaustion-degraded caller (`reserve_*` returned `None`) so primary
    /// work still proceeds; a `Terminal`-class fact submitted here always
    /// reports `TerminalDeliveryFailed` because there is no slot to own it.
    #[must_use]
    pub fn unreserved_ingress(self: &Arc<Self>, scope: OperationScope) -> UnreservedIngress {
        UnreservedIngress {
            engine: Arc::clone(self),
            scope,
        }
    }

    pub(super) fn table(&self) -> &ReservationTable {
        &self.table
    }

    /// Count of currently-occupied reservation slots. Test-only: a linear
    /// scan over the table's full capacity, fine for the small capacities
    /// used in tests but never a production hot path.
    #[cfg(test)]
    #[must_use]
    pub fn occupied_count(&self) -> usize {
        (0..self.table.capacity())
            .filter(|&index| self.table.is_occupied(index).is_some())
            .count()
    }

    /// Test-only observability into the `StateTransition` lane, which
    /// (unlike Terminal-class facts) never reaches `replay()` -- it is a
    /// bounded latest-value-wins map keyed by kind, not part of the
    /// reducer-applied stream. Mirrors `occupied_count()`'s own
    /// test-only-extension precedent (task 9) rather than widening any
    /// production accessor.
    #[cfg(test)]
    #[must_use]
    pub fn state_lane_kinds(&self) -> Vec<&'static str> {
        self.state_lane.kinds()
    }

    pub(super) fn wake(&self) -> &WakeList {
        &self.wake
    }
}

/// An operation-ID-bound admission guard: the only way to obtain a
/// [`ScopedIngress`]. Dropping this guard before a terminal fact was
/// submitted synthesizes one with `terminal_not_delivered`/`unknown`.
#[must_use = "dropping without submitting a terminal synthesizes terminal_not_delivered"]
pub struct OperationReservation {
    engine: Arc<RuntimeEventEngine>,
    scope: OperationScope,
    handle: SlotHandle,
    synthetic_terminal: SyntheticTerminal,
}

impl OperationReservation {
    #[must_use]
    pub fn scope(&self) -> OperationScope {
        self.scope
    }

    #[must_use]
    pub fn ingress(&self) -> ScopedIngress {
        ScopedIngress {
            engine: Arc::clone(&self.engine),
            scope: self.scope,
            handle: self.handle,
        }
    }

    /// Explicit pre-work cancellation: releases the reservation without a
    /// terminal (no synthesis, no wake entry).
    pub fn cancel(self) {
        self.engine.table().release(self.handle);
        drain::cascade_children(&self.engine, self.scope);
        std::mem::forget(self);
    }
}

impl Drop for OperationReservation {
    fn drop(&mut self) {
        if !self.engine.table().has_terminal(self.handle) {
            let record = super::reservation::TerminalRecord {
                fact: (self.synthetic_terminal)(),
                synthesized: true,
            };
            if self.engine.table().write_terminal(self.handle, record) {
                self.engine.wake().push_next(self.handle);
            }
        }
    }
}

pub struct ScopedIngress {
    engine: Arc<RuntimeEventEngine>,
    scope: OperationScope,
    handle: SlotHandle,
}

impl RuntimeEventIngress for ScopedIngress {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome {
        self.engine.submit(self.scope, Some(self.handle), fact)
    }
}

pub struct UnreservedIngress {
    engine: Arc<RuntimeEventEngine>,
    scope: OperationScope,
}

impl RuntimeEventIngress for UnreservedIngress {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome {
        self.engine.submit(self.scope, None, fact)
    }
}
