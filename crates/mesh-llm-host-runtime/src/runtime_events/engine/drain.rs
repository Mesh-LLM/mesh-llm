//! Drain wake entries in ingress-sequence order and apply each through the
//! transactional reducer: only an accepted fact appends a replay frame and
//! fans out to subscribers, so a rejected input never appears on the
//! stream. The reservation is released after the reducer has settled the
//! outcome either way, matching the release-after-ack contract.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{EventSequence, OperationId, OperationScope};

use super::RuntimeEventEngine;
use crate::runtime_events::reducer::{ReduceOutcome, ReducerInput, apply};
use crate::runtime_events::replay::ReplayFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrainReport {
    pub applied: usize,
    pub left_queued: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShutdownReport {
    pub applied: usize,
    pub started_with: usize,
    pub remaining_after_deadline: usize,
}

impl RuntimeEventEngine {
    /// Drain and apply every currently queued wake entry.
    pub fn drain(&self) -> DrainReport {
        self.drain_up_to(None)
    }

    /// Drain and apply at most `max` wake entries, leaving the rest queued.
    /// `None` drains everything currently queued.
    pub fn drain_up_to(&self, max: Option<usize>) -> DrainReport {
        let entries = match max {
            Some(limit) => self.wake().drain_up_to(limit),
            None => self.wake().drain_all(),
        };
        let mut applied = 0;
        for entry in entries {
            let handle = entry.handle;
            if !self.table().is_current(handle) {
                continue;
            }
            let Some(scope) = self.table().is_occupied(handle.index) else {
                continue;
            };
            self.apply_and_publish(handle, scope, entry.ingress_sequence);
            self.table().release(handle);
            cascade_children(self, scope);
            applied += 1;
        }
        DrainReport {
            applied,
            left_queued: self.wake().len(),
        }
    }

    fn apply_and_publish(
        &self,
        handle: super::super::reservation::SlotHandle,
        scope: OperationScope,
        ingress_sequence: u64,
    ) {
        let Some(record) = self.table().terminal_record(handle) else {
            return;
        };
        let fact = Arc::new(record.fact.clone());
        let input = ReducerInput {
            scope,
            ingress_sequence,
            native_sequence: None,
            wall_clock_hint: None,
            synthesized: record.synthesized,
            fact: record.fact,
        };
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ReduceOutcome::Applied(next) = apply(&reducer_state, input) else {
            self.health.bump_reducer_rejected();
            return;
        };
        *reducer_state = next;
        drop(reducer_state);

        let frame = ReplayFrame {
            sequence: EventSequence::new(ingress_sequence),
            rebuild_generation: self.rebuild_generation.load(Ordering::Acquire),
            scope,
            fact,
            recorded_at: Instant::now(),
        };
        if self.replay.push(frame.clone()) {
            self.health.bump_replay_evicted();
        }
        self.subscribers.publish(frame);
    }

    /// Increment `rebuild_generation` and evict every retained replay frame,
    /// simulating a reducer crash/restart recovering into a fresh window.
    pub fn rebuild(&self) -> u64 {
        let generation = self.rebuild_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.health.set_rebuild_generation(generation);
        let evicted = self.replay.evict_all();
        for _ in 0..evicted {
            self.health.bump_replay_evicted();
        }
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let crate::runtime_events::reducer::RebuildOutcome::Rebuilt(next) =
            crate::runtime_events::reducer::rebuild(&reducer_state, generation)
        {
            *reducer_state = next;
        }
        generation
    }

    /// Begin shutdown: block new admission, then drain at most `budget`
    /// wake entries (`None` drains fully). Entries left queued past the
    /// budget are recorded as shutdown-degraded rather than silently
    /// dropped, matching the deadline-degradation rule.
    pub fn shutdown(&self, budget: Option<usize>) -> ShutdownReport {
        self.shutting_down.store(true, Ordering::Release);
        let started_with = self.wake().len();
        let report = self.drain_up_to(budget);
        let remaining = self.wake().len();
        if remaining > 0 {
            self.health.bump_shutdown_degraded();
            for _ in 0..remaining {
                self.health.bump_terminal_delivery_failed();
            }
        }
        ShutdownReport {
            applied: report.applied,
            started_with,
            remaining_after_deadline: remaining,
        }
    }
}

/// Force-complete every outstanding child reservation under `root` when
/// `root`'s own reservation is released, bounding child lifetime by root
/// lifetime. Best-effort: the child's own guard, if it later drops, will
/// see a mismatched generation and no-op.
pub(super) fn cascade_children(engine: &RuntimeEventEngine, scope: OperationScope) {
    let OperationScope::Root(root) = scope else {
        return;
    };
    let indices = take_children(engine, root);
    for index in indices {
        force_complete_child(engine, index);
    }
}

fn take_children(engine: &RuntimeEventEngine, root: OperationId) -> Vec<usize> {
    engine
        .children_by_root
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&root)
        .unwrap_or_default()
}

fn force_complete_child(engine: &RuntimeEventEngine, index: usize) {
    if engine.table().is_occupied(index).is_none() {
        return;
    }
    let generation = engine.table().current_generation(index);
    let handle = super::super::reservation::SlotHandle { index, generation };
    engine.health.bump_terminal_delivery_failed();
    engine.table().release(handle);
}
