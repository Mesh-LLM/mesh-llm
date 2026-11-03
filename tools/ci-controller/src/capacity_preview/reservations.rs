//! Source-only reservation transitions over caller-supplied state.
//!
//! This model is not an admission service. It does not authenticate its input,
//! implement a store, or dispatch, cancel, or release work. A real implementation
//! must commit each transition with an atomic compare-and-swap shared by every
//! scheduler and use complete, authenticated observations and an independently
//! approved catalog. The version check alone cannot supply those guarantees.

use super::{Active, Pending, Preview, PreviewError, Snapshot, Work, preview};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// Version of the entire supplied state, not an authenticated revision.
    pub revision: u64,
    pub snapshot: Snapshot,
    /// Proposed work that must be charged even before a start is observed.
    pub reserved: Vec<Work>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transition {
    /// A proposed state change; it has not been stored or authorized.
    Reserve { work: Work, next: Box<Model> },
    /// No pending request can fit at the head of the supplied queue.
    Wait,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingSupersession {
    /// Older, never-reserved requests removed from the supplied queue model.
    pub removed: Vec<Work>,
    pub next: Box<Model>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    Preview(PreviewError),
    RevisionMismatch,
    RevisionOverflow,
    UnreconciledSupersession,
    ReservationOverCapacity,
    UnknownReservation,
    ReservationMismatch,
}

impl From<PreviewError> for ModelError {
    fn from(value: PreviewError) -> Self {
        Self::Preview(value)
    }
}

fn expected_revision(model: &Model, expected: u64) -> Result<u64, ModelError> {
    if model.revision != expected {
        return Err(ModelError::RevisionMismatch);
    }
    model
        .revision
        .checked_add(1)
        .ok_or(ModelError::RevisionOverflow)
}

/// Charge every reservation before evaluating the remaining FIFO queue.
///
/// Reservations are modeled as pending demand, never as observed runners.
/// Every reservation must remain charged before later work is considered.
fn preview_model(model: &Model, now: u64) -> Result<Preview, ModelError> {
    let mut snapshot = model.snapshot.clone();
    let mut pending = model
        .reserved
        .iter()
        .cloned()
        .map(|work| Pending {
            work,
            supersedes: None,
        })
        .collect::<Vec<_>>();
    pending.append(&mut snapshot.pending);
    snapshot.pending = pending;

    let result = preview(&snapshot, now)?;
    if result
        .superseded_pending
        .iter()
        .any(|id| model.reserved.iter().any(|reserved| reserved.id == *id))
    {
        return Err(ModelError::UnreconciledSupersession);
    }
    if result.candidates.len() < model.reserved.len()
        || !result
            .candidates
            .iter()
            .zip(&model.reserved)
            .all(|(candidate, work)| candidate == &work.id)
    {
        return Err(ModelError::ReservationOverCapacity);
    }
    Ok(result)
}

fn first_candidate(model: &Model, now: u64) -> Result<Option<String>, ModelError> {
    if model
        .snapshot
        .pending
        .iter()
        .any(|work| work.supersedes.is_some())
    {
        return Err(ModelError::UnreconciledSupersession);
    }
    let result = preview_model(model, now)?;
    Ok(result.candidates.get(model.reserved.len()).cloned())
}

/// Propose one FIFO reservation from a caller-supplied snapshot.
///
/// `expected` is a local precondition. Before any live use, an external store
/// must atomically replace the exact expected state with `next`; two callers
/// evaluating the same revision otherwise both receive a proposal. This
/// repository provides no production store or workflow caller.
pub fn propose(model: &Model, expected: u64, now: u64) -> Result<Transition, ModelError> {
    let next_revision = expected_revision(model, expected)?;
    let Some(candidate) = first_candidate(model, now)? else {
        return Ok(Transition::Wait);
    };
    let mut next = model.clone();
    let pending = next.snapshot.pending.remove(0);
    if pending.work.id != candidate {
        return Err(ModelError::ReservationMismatch);
    }
    next.revision = next_revision;
    next.reserved.push(pending.work.clone());
    Ok(Transition::Reserve {
        work: pending.work,
        next: Box::new(next),
    })
}

/// Model a supplied start observation without freeing its reserved capacity.
///
/// The observation and its runner class are caller assertions, not attested
/// facts. This transition is also subject to an atomic expected-state commit.
/// There is deliberately no expiry or completion transition: capacity cannot
/// be released until a separately designed authenticated reconciler proves it.
pub fn observe_start(
    model: &Model,
    expected: u64,
    now: u64,
    observed: Active,
) -> Result<Model, ModelError> {
    let next_revision = expected_revision(model, expected)?;
    first_candidate(model, now)?;
    let index = model
        .reserved
        .iter()
        .position(|work| work.id == observed.work.id)
        .ok_or(ModelError::UnknownReservation)?;
    if model.reserved[index] != observed.work {
        return Err(ModelError::ReservationMismatch);
    }
    let mut next = model.clone();
    next.reserved.remove(index);
    next.snapshot.active.push(observed);
    next.revision = next_revision;
    first_candidate(&next, now)?;
    Ok(next)
}

/// Propose removing only explicitly superseded, still-pending PR work.
///
/// This operates entirely on supplied state. A complete, authenticated queue
/// and atomic participation by all schedulers are prerequisites for live use.
/// Reserved or active predecessors are never removed by this transition;
/// they require separately observed cancellation and reconciliation. Main,
/// release, warmer and control work are preserved.
pub fn propose_pending_supersession(
    model: &Model,
    expected: u64,
    now: u64,
) -> Result<Option<PendingSupersession>, ModelError> {
    let next_revision = expected_revision(model, expected)?;
    let result = preview_model(model, now)?;
    if !result.active_cancellation_candidates.is_empty() {
        return Err(ModelError::UnreconciledSupersession);
    }
    let ids = result
        .superseded_pending
        .into_iter()
        .collect::<BTreeSet<_>>();
    if ids.is_empty() {
        return Ok(None);
    }
    let pending_ids = model
        .snapshot
        .pending
        .iter()
        .map(|pending| pending.work.id.as_str())
        .collect::<BTreeSet<_>>();
    if ids.iter().any(|id| !pending_ids.contains(id.as_str())) {
        return Err(ModelError::UnreconciledSupersession);
    }
    let mut next = model.clone();
    let mut removed = Vec::new();
    next.snapshot.pending.retain(|pending| {
        if ids.contains(&pending.work.id) {
            removed.push(pending.work.clone());
            false
        } else {
            true
        }
    });
    for pending in &mut next.snapshot.pending {
        if let Some(target) = &pending.supersedes {
            if !ids.contains(target) {
                return Err(ModelError::UnreconciledSupersession);
            }
            pending.supersedes = None;
        }
    }
    next.revision = next_revision;
    first_candidate(&next, now)?;
    Ok(Some(PendingSupersession {
        removed,
        next: Box::new(next),
    }))
}

#[cfg(test)]
#[path = "reservations_tests.rs"]
mod tests;
