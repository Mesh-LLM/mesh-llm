//! Source-only receipts for replaying committed reservation transitions.
//!
//! This is not a store or an admission service. It assumes a future store can
//! atomically retain the entire model and receipt history, and that callers
//! reuse a globally unique operation ID for retries. Neither property is
//! established here. Input state and start claims remain unauthenticated.

use super::reservations::{
    Model, ModelError, Transition, observe_start, propose, propose_pending_supersession,
};
use super::{Active, Work, identifier};
use std::collections::BTreeMap;

/// The exact request bound to an operation ID, including its supplied time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Reserve { now: u64 },
    RecordStart { now: u64, observed: Box<Active> },
    SupersedePending { now: u64 },
}

/// A stored model change, not proof that work was authorized or ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Receipt {
    Reserved { work: Work, revision: u64 },
    StartRecorded { revision: u64 },
    PendingSuperseded { removed: Vec<Work>, revision: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub request: Request,
    pub receipt: Receipt,
}

/// The model and all committed operation receipts must be one atomic value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Journal {
    pub model: Model,
    pub receipts: BTreeMap<String, Entry>,
}

/// Contract required of a future shared store; no implementation is supplied.
///
/// Both methods must be linearizable across all participants. A successful
/// compare/exchange durably commits the *entire* replacement, including the
/// model and append-only receipts. A false result writes nothing and means the
/// expected value did not match. An error
/// can occur before or after commitment. All receipts must be retained for as
/// long as an operation ID could be retried; there is no safe eviction rule or
/// bounded storage implementation here. Every participant must use the same
/// journal and validated transitions, and the store must prevent rollback or
/// replacement of committed history. The store must also authenticate its
/// state and participants before any live use.
pub trait AtomicJournalStore {
    type Error;

    fn load(&self) -> Result<Journal, Self::Error>;

    fn compare_exchange(
        &self,
        expected: &Journal,
        replacement: &Journal,
    ) -> Result<bool, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Applied(Receipt),
    /// The identical request was already committed under this operation ID.
    Replayed(Receipt),
    /// No state change was committed or recorded; a later attempt may differ.
    Wait,
    /// No state change was committed or recorded; a later attempt may differ.
    NoSupersession,
    /// Another writer changed the state before this request could commit.
    Contended,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JournalError<E> {
    InvalidOperationId,
    ReusedOperationId,
    Load(E),
    /// The commit may have happened. Retry only with the same ID and request.
    CommitUnknown(E),
    Model(ModelError),
}

/// Apply or replay one supplied request without dispatching any work.
///
/// A caller must preserve the operation ID and complete request across an
/// uncertain acknowledgement. A committed receipt is returned on replay,
/// even if later operations have changed the model. Reusing a committed ID
/// for a different request is rejected. A reserve request selects the next
/// work at commit time; callers must use the returned receipt, not assume an
/// earlier uncommitted candidate. `Wait` and `NoSupersession` are not recorded
/// and may change on a later attempt. IDs must not contain secrets. This cannot
/// compensate for a non-atomic store, incomplete history, reused IDs or
/// unauthenticated input.
pub fn apply<S: AtomicJournalStore>(
    store: &S,
    operation_id: &str,
    request: Request,
) -> Result<Outcome, JournalError<S::Error>> {
    if !identifier(operation_id) {
        return Err(JournalError::InvalidOperationId);
    }
    let current = store.load().map_err(JournalError::Load)?;
    if let Some(entry) = current.receipts.get(operation_id) {
        return if entry.request == request {
            Ok(Outcome::Replayed(entry.receipt.clone()))
        } else {
            Err(JournalError::ReusedOperationId)
        };
    }
    let (model, receipt) = match &request {
        Request::Reserve { now } => {
            match propose(&current.model, current.model.revision, *now)
                .map_err(JournalError::Model)?
            {
                Transition::Wait => return Ok(Outcome::Wait),
                Transition::Reserve { work, next } => {
                    let receipt = Receipt::Reserved {
                        work,
                        revision: next.revision,
                    };
                    (*next, receipt)
                }
            }
        }
        Request::RecordStart { now, observed } => {
            let next = observe_start(
                &current.model,
                current.model.revision,
                *now,
                observed.as_ref().clone(),
            )
            .map_err(JournalError::Model)?;
            let receipt = Receipt::StartRecorded {
                revision: next.revision,
            };
            (next, receipt)
        }
        Request::SupersedePending { now } => {
            let Some(proposal) =
                propose_pending_supersession(&current.model, current.model.revision, *now)
                    .map_err(JournalError::Model)?
            else {
                return Ok(Outcome::NoSupersession);
            };
            let receipt = Receipt::PendingSuperseded {
                removed: proposal.removed,
                revision: proposal.next.revision,
            };
            (*proposal.next, receipt)
        }
    };
    let mut next = current.clone();
    next.model = model;
    next.receipts.insert(
        operation_id.to_owned(),
        Entry {
            request,
            receipt: receipt.clone(),
        },
    );
    if store
        .compare_exchange(&current, &next)
        .map_err(JournalError::CommitUnknown)?
    {
        Ok(Outcome::Applied(receipt))
    } else {
        Ok(Outcome::Contended)
    }
}

#[cfg(test)]
#[path = "journal_tests.rs"]
mod tests;
