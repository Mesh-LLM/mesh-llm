//! Per-delivery-class submit handling.
//!
//! Terminal writes are the only path that touches the reservation table's
//! write-once slot and the wake list. The other three lanes are minimal,
//! genuinely bounded stand-ins: task 4 owns their real reducer semantics.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use mesh_llm_runtime_event_contracts::{OperationScope, RuntimeFact, SubmitOutcome};

use super::RuntimeEventEngine;
use crate::runtime_events::config::{DIAGNOSTIC_LANE_DEPTH, STATE_TRANSITION_LANE_DEPTH};
use crate::runtime_events::reservation::{SlotHandle, TerminalRecord};

/// Per-engine bounded latest-value lane: a repeat kind coalesces in place, a
/// new kind past the depth ceiling evicts the oldest distinct kind rather
/// than reporting a drop (state transitions have no `Dropped*` outcome).
#[derive(Default)]
pub(crate) struct StateLane {
    entries: Mutex<VecDeque<&'static str>>,
    latest: Mutex<HashMap<&'static str, RuntimeFact>>,
}

impl StateLane {
    /// Test-only: the set of kinds currently held (latest-value-wins) in
    /// this lane. Backs `RuntimeEventEngine::state_lane_kinds()`.
    #[cfg(test)]
    pub(super) fn kinds(&self) -> Vec<&'static str> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .copied()
            .collect()
    }
}

/// Per-engine bounded diagnostic queue.
#[derive(Default)]
pub(crate) struct DiagnosticLane {
    queue: Mutex<VecDeque<RuntimeFact>>,
}

pub(super) fn submit_terminal(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let Some(handle) = handle else {
        engine.health().bump_terminal_delivery_failed();
        return SubmitOutcome::TerminalDeliveryFailed;
    };
    if engine.table().occupant(handle) != Some(scope) {
        engine.health().bump_terminal_delivery_failed();
        return SubmitOutcome::TerminalDeliveryFailed;
    }
    let record = TerminalRecord {
        fact,
        synthesized: false,
    };
    if engine.table().write_terminal(handle, record) {
        engine.wake().push_next(handle);
        SubmitOutcome::Accepted
    } else {
        engine.health().bump_terminal_delivery_failed();
        SubmitOutcome::TerminalDeliveryFailed
    }
}

pub(super) fn submit_progress(
    engine: &RuntimeEventEngine,
    handle: Option<SlotHandle>,
    fact: RuntimeFact,
) -> SubmitOutcome {
    match handle {
        Some(handle) if engine.table().coalesce_progress(handle, fact) => SubmitOutcome::Coalesced,
        _ => {
            engine.health().bump_dropped_progress();
            SubmitOutcome::DroppedProgress
        }
    }
}

pub(super) fn submit_state_transition(
    engine: &RuntimeEventEngine,
    fact: RuntimeFact,
) -> SubmitOutcome {
    let lane = engine.state_lane();
    let kind = fact.kind_id();
    let mut latest = lane
        .latest
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if latest.insert(kind, fact).is_some() {
        return SubmitOutcome::Coalesced;
    }
    let mut entries = lane
        .entries
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    entries.push_back(kind);
    if entries.len() > STATE_TRANSITION_LANE_DEPTH {
        if let Some(evicted) = entries.pop_front() {
            latest.remove(evicted);
        }
    }
    SubmitOutcome::Accepted
}

pub(super) fn submit_diagnostic(engine: &RuntimeEventEngine, fact: RuntimeFact) -> SubmitOutcome {
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
    queue.push_back(fact);
    SubmitOutcome::Accepted
}
