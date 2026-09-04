//! Per-delivery-class submit handling.
//!
//! Terminal writes are the only path that touches the reservation table's
//! write-once slot and the wake list; state-transition, diagnostic, and
//! progress facts route through their own bounded structures below, each
//! drained fully by `engine::drain` (task 4,
//! `.omo/plans/event-system-fixes.md`). Every submit function here mints
//! from the SAME shared counter (`wake.rs::next_sequence`) exactly once
//! per call, regardless of outcome -- there is no second counter.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use mesh_llm_runtime_event_contracts::{OperationScope, RuntimeFact, SubmitOutcome};

use super::RuntimeEventEngine;
use crate::runtime_events::config::{DIAGNOSTIC_LANE_DEPTH, STATE_TRANSITION_LANE_DEPTH};
use crate::runtime_events::reservation::{SlotHandle, TerminalRecord};

/// A state-transition lane key: coalescing is per operation scope AND
/// kind, never globally by kind alone (review defect D2) -- two different
/// operations reporting the same kind must never overwrite each other.
type StateLaneKey = (OperationScope, &'static str);

/// Per-engine bounded latest-value lane, keyed by `(OperationScope, kind)`:
/// a repeat key coalesces in place; a new key past the depth ceiling
/// evicts the oldest distinct key rather than reporting a drop (state
/// transitions have no `Dropped*` outcome). Each held value carries the
/// ingress sequence it was minted with.
#[derive(Default)]
pub(crate) struct StateLane {
    entries: Mutex<VecDeque<StateLaneKey>>,
    /// `(fact, ingress_sequence, reserved)` -- `reserved` (R1 fix, task
    /// 6-fix, `.omo/plans/event-system-fixes.md`) is threaded from
    /// `submit_state_transition`'s own `handle.is_some()` and carried all
    /// the way to the reducer's `ReducerInput::reserved`.
    latest: Mutex<HashMap<StateLaneKey, (RuntimeFact, u64, bool)>>,
}

impl StateLane {
    /// Test-only: the kinds currently held (latest-value-wins) in this
    /// lane, across every scope. Backs `RuntimeEventEngine::state_lane_kinds()`.
    #[cfg(test)]
    pub(super) fn kinds(&self) -> Vec<&'static str> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(_, kind)| *kind)
            .collect()
    }

    /// Drain every currently-held key, oldest first, clearing the lane.
    /// A submission for the same key that arrives after this call starts
    /// fresh (`Accepted`, not `Coalesced`), matching the terminal lane's
    /// own "drained means gone" contract.
    pub(super) fn drain(&self) -> Vec<(OperationScope, RuntimeFact, u64, bool)> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut latest = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries
            .drain(..)
            .filter_map(|key| {
                latest
                    .remove(&key)
                    .map(|(fact, sequence, reserved)| (key.0, fact, sequence, reserved))
            })
            .collect()
    }
}

/// Per-engine bounded diagnostic queue: strict FIFO, each entry carrying
/// the scope, ingress sequence, and reservation provenance (R1 fix, task
/// 6-fix -- see [`StateLane`]'s identical addition) it was submitted with.
#[derive(Default)]
pub(crate) struct DiagnosticLane {
    queue: Mutex<VecDeque<(OperationScope, RuntimeFact, u64, bool)>>,
}

impl DiagnosticLane {
    /// Drain every currently-queued diagnostic, oldest first, clearing the
    /// queue.
    pub(super) fn drain(&self) -> Vec<(OperationScope, RuntimeFact, u64, bool)> {
        self.queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect()
    }
}

pub(super) fn submit_terminal(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let Some(handle) = handle else {
        engine.wake().next_ingress_sequence();
        engine.health().bump_terminal_delivery_failed();
        return SubmitOutcome::TerminalDeliveryFailed;
    };
    if engine.table().occupant(handle) != Some(scope) {
        engine.wake().next_ingress_sequence();
        engine.health().bump_terminal_delivery_failed();
        return SubmitOutcome::TerminalDeliveryFailed;
    }
    let record = TerminalRecord {
        fact,
        synthesized: false,
    };
    if engine.table().write_terminal(handle, record) {
        // Mint-and-enqueue as ONE atomic step (unchanged `push_next`):
        // splitting these across two lock acquisitions would let a
        // later-minted concurrent submission's push overtake an
        // earlier-minted one's, breaking the wake list's FIFO ==
        // ingress-sequence-order invariant under concurrent terminal
        // writes to different scopes.
        engine.wake().push_next(handle);
        SubmitOutcome::Accepted
    } else {
        engine.wake().next_ingress_sequence();
        engine.health().bump_terminal_delivery_failed();
        SubmitOutcome::TerminalDeliveryFailed
    }
}

pub(super) fn submit_progress(
    engine: &RuntimeEventEngine,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let sequence = engine.wake().next_ingress_sequence();
    match handle {
        Some(handle) if engine.table().coalesce_progress(handle, fact, sequence) => {
            SubmitOutcome::Coalesced
        }
        _ => {
            engine.health().bump_dropped_progress();
            SubmitOutcome::DroppedProgress
        }
    }
}

pub(super) fn submit_state_transition(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    reserved: bool,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let sequence = engine.wake().next_ingress_sequence();
    let lane = engine.state_lane();
    let key: StateLaneKey = (scope, fact.kind_id());
    // Lock `entries` THEN `latest` -- the SAME order `StateLane::drain`
    // above uses. Taking `latest` first (as this function used to) is an
    // AB-BA inversion against `drain`'s `entries`-then-`latest` order: a
    // concurrent drainer holding `entries` while waiting on `latest` and a
    // submitter holding `latest` while waiting on `entries` deadlock each
    // other. `inference::skippy::runtime_events::tests::concurrent_roots`
    // (task 5) was the first test to actually drive concurrent drain +
    // state-transition submits and surfaced this as an intermittent hang.
    let mut entries = lane
        .entries
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut latest = lane
        .latest
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if latest.insert(key, (fact, sequence, reserved)).is_some() {
        return SubmitOutcome::Coalesced;
    }
    entries.push_back(key);
    if entries.len() > STATE_TRANSITION_LANE_DEPTH
        && let Some(evicted) = entries.pop_front()
    {
        latest.remove(&evicted);
    }
    SubmitOutcome::Accepted
}

pub(super) fn submit_diagnostic(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    reserved: bool,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let sequence = engine.wake().next_ingress_sequence();
    let mut queue = engine
        .diagnostic_lane()
        .queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if queue.len() >= DIAGNOSTIC_LANE_DEPTH {
        drop(queue);
        engine.health().bump_dropped_diagnostic();
        return SubmitOutcome::DroppedDiagnostic;
    }
    queue.push_back((scope, fact, sequence, reserved));
    SubmitOutcome::Accepted
}
