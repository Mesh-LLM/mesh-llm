//! Deferred native-runtime resolution facts.
//!
//! Host startup resolves and loads the native runtime before the runtime
//! event engine is installed. Facts produced in that window would otherwise
//! be discarded, leaving the reducer without any `NativeRuntime` state for
//! the whole process lifetime. The resolution records its steps here instead
//! and the engine installer replays them under a fresh reservation.
//!
//! Bounded: one pending resolution (latest wins), at most
//! [`MAX_DEFERRED_STEPS`] steps each.

use std::sync::{Arc, Mutex, PoisonError};

use mesh_llm_runtime_event_contracts::{FactData, NativeRuntimeEventKind, OperationId};

use super::{submit, synthetic_terminal};
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::runtime_event_engine;

pub(super) const MAX_DEFERRED_STEPS: usize = 8;

pub(super) type DeferredStep = (NativeRuntimeEventKind, FactData);

static PENDING: Mutex<Option<Vec<DeferredStep>>> = Mutex::new(None);

/// Steps recorded by a resolution that began without an installed engine.
#[derive(Default)]
pub(super) struct DeferredSteps {
    steps: Vec<DeferredStep>,
}

impl DeferredSteps {
    pub(super) fn record(&mut self, kind: NativeRuntimeEventKind, data: FactData) {
        if self.steps.len() < MAX_DEFERRED_STEPS {
            self.steps.push((kind, data));
        }
    }

    /// Hands the finished resolution to the engine: immediately when one was
    /// installed meanwhile, otherwise at the next [`replay_deferred_resolution`].
    pub(super) fn finish(self) {
        if self.steps.is_empty() {
            return;
        }
        // Hold the slot across the engine check: the installer publishes the
        // engine before it takes this lock, so a resolution either sees the
        // engine here or is stored before the installer's replay reads it.
        let mut pending = PENDING.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(engine) = runtime_event_engine() else {
            *pending = Some(self.steps);
            return;
        };
        drop(pending);
        replay_steps(&engine, self.steps);
    }
}

fn replay_steps(engine: &Arc<RuntimeEventEngine>, steps: Vec<DeferredStep>) {
    let Some(root) = engine.reserve_root(OperationId::new(), synthetic_terminal) else {
        return;
    };
    for (kind, data) in steps {
        submit(&root, kind, data);
    }
}

/// Replays a resolution that completed before `engine` was installed. Call
/// right after installing the process runtime event engine.
pub(crate) fn replay_deferred_resolution(engine: &Arc<RuntimeEventEngine>) {
    let pending = PENDING
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(steps) = pending {
        replay_steps(engine, steps);
    }
}

#[cfg(test)]
pub(super) fn clear_pending() {
    PENDING
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
}

#[cfg(test)]
pub(super) fn has_pending() -> bool {
    PENDING
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_some()
}
