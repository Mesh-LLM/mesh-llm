//! Pure transactional application: `(snapshot, input) -> Applied | Rejected`.
//!
//! No I/O, no locking. Ordering is ingress-sequence only — native sequence
//! and any wall-clock hint are recorded as data, never consulted to decide
//! acceptance or ordering, per spec §10-11.

use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{OperationScope, RuntimeFact};

use super::state::{OperationState, ReducerSnapshot, RejectReason, outcome_of, progress_of};

/// One fact to reduce, already assigned its process-local ingress sequence
/// by the engine's wake list (or synthesized by a test driving the reducer
/// directly). `native_sequence` and `wall_clock_hint` are optional,
/// producer-supplied and inert for ordering/acceptance decisions.
#[derive(Debug, Clone)]
pub struct ReducerInput {
    pub scope: OperationScope,
    pub ingress_sequence: u64,
    pub native_sequence: Option<u64>,
    pub wall_clock_hint: Option<i64>,
    pub synthesized: bool,
    pub fact: RuntimeFact,
}

#[derive(Debug, Clone)]
pub enum ReduceOutcome {
    Applied(Arc<ReducerSnapshot>),
    Rejected(RejectReason),
}

/// Apply `input` against `snapshot`, returning a fresh snapshot on success.
/// `snapshot` itself is never mutated: rejection leaves the caller's `Arc`
/// exactly as it was, which is the whole transactional guarantee.
#[must_use]
pub fn apply(snapshot: &Arc<ReducerSnapshot>, input: ReducerInput) -> ReduceOutcome {
    let current = snapshot.get_or_default(input.scope);

    if current.has_applied && input.ingress_sequence <= current.last_ingress_sequence {
        return ReduceOutcome::Rejected(RejectReason::Duplicate);
    }

    let data = input.fact.data();
    let incoming_outcome = outcome_of(data);
    let incoming_progress = progress_of(data);

    if current.settled {
        return match incoming_outcome {
            Some(_) => ReduceOutcome::Rejected(RejectReason::ContradictoryTerminal),
            None => ReduceOutcome::Rejected(RejectReason::OperationSettled),
        };
    }

    if let Some(incoming) = incoming_progress
        && let Some(previous) = current.last_progress_current
        && incoming < previous
    {
        return ReduceOutcome::Rejected(RejectReason::StaleProgress);
    }

    let next = advance(&current, &input, incoming_outcome, incoming_progress);
    let next_domain = snapshot.domain().apply_fact(&input.fact);
    ReduceOutcome::Applied(Arc::new(snapshot.with_operation(
        input.scope,
        next,
        next_domain,
    )))
}

fn advance(
    current: &OperationState,
    input: &ReducerInput,
    incoming_outcome: Option<mesh_llm_runtime_event_contracts::Outcome>,
    incoming_progress: Option<u64>,
) -> OperationState {
    let mut next = current.clone();
    next.has_applied = true;
    next.last_ingress_sequence = input.ingress_sequence;
    next.last_native_sequence = resolve_native_sequence(&mut next, input.native_sequence);

    if let Some(progress) = incoming_progress {
        next.last_progress_current = Some(progress);
    }
    if let Some(outcome) = incoming_outcome {
        next.settled = true;
        next.last_outcome = Some(outcome);
        next.degraded = next.degraded || input.synthesized;
    }
    next
}

/// Track the native-sequence high-water mark and flag (never act on) a gap.
fn resolve_native_sequence(state: &mut OperationState, incoming: Option<u64>) -> Option<u64> {
    let Some(incoming) = incoming else {
        return state.last_native_sequence;
    };
    if let Some(previous) = state.last_native_sequence
        && incoming > previous + 1
    {
        state.native_gap_count += 1;
    }
    Some(incoming)
}
