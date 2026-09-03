//! Pure, immutable reducer state.
//!
//! `ReducerSnapshot` is never mutated in place: every transition produces a
//! fresh `Arc<ReducerSnapshot>` (clone-on-write over a small bounded map),
//! so a rejected input structurally cannot leave a partially applied
//! snapshot visible to any reader.

use std::collections::HashMap;
use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{FactData, OperationScope, Outcome};

/// Per-operation reduced view. Preserved across rebuild for "last-valid"
/// continuity even when the operation never settles cleanly.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationState {
    pub scope: OperationScope,
    pub has_applied: bool,
    pub settled: bool,
    pub degraded: bool,
    pub last_outcome: Option<Outcome>,
    pub last_progress_current: Option<u64>,
    pub last_ingress_sequence: u64,
    pub last_native_sequence: Option<u64>,
    pub native_gap_count: u64,
}

impl OperationState {
    fn new(scope: OperationScope) -> Self {
        Self {
            scope,
            has_applied: false,
            settled: false,
            degraded: false,
            last_outcome: None,
            last_progress_current: None,
            last_ingress_sequence: 0,
            last_native_sequence: None,
            native_gap_count: 0,
        }
    }
}

/// Immutable, `Arc`-shared reducer state. Cheap to hand to readers; a
/// writer never mutates an existing instance, only produces a new one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReducerSnapshot {
    operations: HashMap<OperationScope, OperationState>,
    pub rebuild_generation: u64,
}

impl ReducerSnapshot {
    #[must_use]
    pub fn empty() -> Arc<Self> {
        Arc::new(Self::default())
    }

    #[must_use]
    pub fn operation(&self, scope: OperationScope) -> Option<&OperationState> {
        self.operations.get(&scope)
    }

    #[must_use]
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub(super) fn get_or_default(&self, scope: OperationScope) -> OperationState {
        self.operations
            .get(&scope)
            .cloned()
            .unwrap_or_else(|| OperationState::new(scope))
    }

    /// Produce a new snapshot with `scope`'s state replaced. The receiver is
    /// left untouched; callers swap the shared `Arc` only after this
    /// succeeds, which is what makes application transactional.
    #[must_use]
    pub(super) fn with_operation(&self, scope: OperationScope, state: OperationState) -> Self {
        let mut operations = self.operations.clone();
        operations.insert(scope, state);
        Self {
            operations,
            rebuild_generation: self.rebuild_generation,
        }
    }

    #[must_use]
    pub(super) fn with_generation(&self, generation: u64) -> Self {
        Self {
            operations: self.operations.clone(),
            rebuild_generation: generation,
        }
    }

    #[must_use]
    pub(super) fn degrade_unsettled(&self) -> Self {
        let operations = self
            .operations
            .iter()
            .map(|(scope, state)| {
                let mut next = state.clone();
                if !next.settled {
                    next.degraded = true;
                }
                (*scope, next)
            })
            .collect();
        Self {
            operations,
            rebuild_generation: self.rebuild_generation,
        }
    }
}

/// Why a reducer input was not applied. The stream never observes a
/// rejected input: no replay frame, no subscriber fan-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// Same-or-earlier ingress sequence already applied for this scope.
    Duplicate,
    /// Progress regressed against the last-applied progress value.
    StaleProgress,
    /// A second terminal for an already-settled operation.
    ContradictoryTerminal,
    /// A non-terminal fact for an already-settled operation.
    OperationSettled,
}

pub(super) fn outcome_of(data: &FactData) -> Option<Outcome> {
    data.outcome
}

pub(super) fn progress_of(data: &FactData) -> Option<u64> {
    data.progress.map(|progress| progress.current)
}
