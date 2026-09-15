//! Drain-owned lane state.
//!
//! Delivery classes used to be separated at the producer boundary: four
//! bounded containers, each with its own mutex, each written by whichever
//! thread happened to be emitting an event. That is what made a producer
//! wait on a lane, and what forced the drain to reconstruct a total order
//! across four containers under a shared gate.
//!
//! Now all four classes arrive through one ring in submission order, and
//! this module is the consumer-side state that remains. Only a drain pass
//! touches it, serialized by `drain_gate`, so it needs no producer-visible
//! locking of its own:
//!
//! * **Terminal** and **diagnostic** facts pass straight through in the
//!   pass that popped them. Neither accumulates.
//! * **State transitions** coalesce per `(OperationScope, kind)` within a
//!   pass. A repeated key keeps the newest value at the newest position,
//!   because the newest value is what a consumer wants and the newest
//!   position is where it actually arrived.
//! * **Progress** is the only class that persists across passes: one
//!   latest value per operation, exported at most once per
//!   `PROGRESS_EXPORT_INTERVAL`. Everything in between is a superseded
//!   snapshot and is counted as a drop rather than published.

use std::collections::HashMap;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{DeliveryClass, OperationScope, RuntimeFact};

use crate::runtime_events::config::PROGRESS_EXPORT_INTERVAL;
use crate::runtime_events::ingress::IngressFact;
use crate::runtime_events::reservation::SlotHandle;

/// One fact on its way to the reducer, with the provenance the reducer and
/// the liveness check both need.
#[derive(Debug)]
pub(crate) struct PendingFact {
    pub(crate) scope: OperationScope,
    pub(crate) fact: RuntimeFact,
    pub(crate) handle: Option<SlotHandle>,
    pub(crate) reserved: bool,
    pub(crate) synthesized: bool,
    /// Whether this fact's reservation was still valid when the pass
    /// routed it. Decided at routing time, before any of this pass's
    /// releases, so a terminal cannot invalidate facts that preceded it.
    pub(crate) live: bool,
}

impl From<IngressFact> for PendingFact {
    fn from(entry: IngressFact) -> Self {
        Self {
            scope: entry.scope,
            fact: entry.fact,
            handle: entry.handle,
            reserved: entry.reserved,
            synthesized: entry.synthesized,
            // Overwritten by the router, which owns the decision.
            live: true,
        }
    }
}

/// A state-transition coalescing key: per operation scope AND kind, never
/// globally by kind alone -- two different operations reporting the same
/// kind must never overwrite each other.
type StateKey = (OperationScope, &'static str);

/// What one drain pass accumulated, ready to apply in arrival order.
#[derive(Debug, Default)]
pub(crate) struct PassBatch {
    /// Facts to apply, in arrival order. `None` marks a state transition
    /// superseded by a later one for the same key in this same pass.
    entries: Vec<Option<PendingFact>>,
    /// Where each live state key currently sits in `entries`.
    state_positions: HashMap<StateKey, usize>,
    /// Progress values superseded before their export window came due.
    pub(crate) superseded_progress: usize,
}

impl PassBatch {
    /// Add one popped fact, coalescing it against this pass if its class
    /// says to.
    pub(crate) fn push(&mut self, entry: PendingFact, class: DeliveryClass) {
        if class != DeliveryClass::StateTransition {
            self.entries.push(Some(entry));
            return;
        }
        let key: StateKey = (entry.scope, entry.fact.kind_id());
        let position = self.entries.len();
        if let Some(previous) = self.state_positions.insert(key, position) {
            // Keep the newest value at the newest position: superseding in
            // place would publish a value under an older arrival order it
            // never had.
            self.entries[previous] = None;
        }
        self.entries.push(Some(entry));
    }

    /// Consume the batch in arrival order.
    pub(crate) fn drain(self) -> impl Iterator<Item = PendingFact> {
        self.entries.into_iter().flatten()
    }
}

/// Progress values waiting for their export window, one per operation.
///
/// This is the only lane that survives a drain pass, because its whole
/// purpose is to rate-limit: a generation emitting ten times a second
/// should publish ten frames a second regardless of how often the driver
/// happens to tick.
#[derive(Debug, Default)]
pub(crate) struct ProgressLane {
    latest: HashMap<OperationScope, PendingFact>,
    /// Arrival order of the keys in `latest`, so a flush publishes in the
    /// order the operations last reported rather than in map order.
    order: Vec<OperationScope>,
    last_flush: Option<Instant>,
}

impl ProgressLane {
    /// Record `entry` as this operation's latest progress. Returns whether
    /// it superseded a value that had not yet been exported.
    pub(crate) fn record(&mut self, entry: PendingFact) -> bool {
        let scope = entry.scope;
        let superseded = self.latest.insert(scope, entry).is_some();
        if !superseded {
            self.order.push(scope);
        }
        superseded
    }

    /// Whether the export window is due at `now`. The first call is always
    /// due, matching `EngineHealth`'s identical never-published-yet
    /// convention.
    pub(crate) fn is_due(&self, now: Instant) -> bool {
        self.last_flush
            .is_none_or(|previous| now.duration_since(previous) >= PROGRESS_EXPORT_INTERVAL)
    }

    /// Take every held value and open a new export window.
    ///
    /// The window moves whenever it is due, held values or not, so the
    /// cadence is anchored to drain passes rather than to traffic. An
    /// operation that starts reporting mid-window therefore waits at most
    /// one full interval, instead of publishing immediately and resetting
    /// the cadence for everyone else.
    pub(crate) fn take_due(&mut self, now: Instant) -> Vec<PendingFact> {
        if !self.is_due(now) {
            return Vec::new();
        }
        self.last_flush = Some(now);
        let order = std::mem::take(&mut self.order);
        let mut latest = std::mem::take(&mut self.latest);
        order
            .into_iter()
            .filter_map(|scope| latest.remove(&scope))
            .collect()
    }

    /// Take every held value regardless of the export window. Shutdown
    /// uses this so a final progress snapshot is not stranded by a cadence
    /// that will never come due again.
    pub(crate) fn take_all(&mut self) -> Vec<PendingFact> {
        let order = std::mem::take(&mut self.order);
        let mut latest = std::mem::take(&mut self.latest);
        order
            .into_iter()
            .filter_map(|scope| latest.remove(&scope))
            .collect()
    }

    /// Forget `scope`'s held progress. Called when a reservation is
    /// cancelled or released, so a settled operation cannot publish a
    /// progress frame afterwards.
    pub(crate) fn forget(&mut self, scope: OperationScope) {
        if self.latest.remove(&scope).is_some() {
            self.order.retain(|held| *held != scope);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.latest.len()
    }
}
