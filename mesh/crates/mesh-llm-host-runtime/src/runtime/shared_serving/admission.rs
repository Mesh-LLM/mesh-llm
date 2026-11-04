//! Bounded, non-preemptive round-robin admission across independent memberships.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use skippy_inference_api::{
    CancellationToken, InferenceError, InferenceErrorKind, InferenceExecutionLease,
};
use tokio::sync::oneshot;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct MemberId(u64);

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct AdmissionStatus {
    pub accepting: bool,
    pub active: usize,
    pub queued: usize,
}

#[derive(Debug)]
struct Member {
    accepting: bool,
    retired: bool,
    queue_limit: usize,
    queue: VecDeque<u64>,
    active: usize,
}

#[derive(Debug)]
struct Ticket {
    member: MemberId,
    ready: Option<oneshot::Sender<()>>,
    cancellation: CancellationToken,
}

#[derive(Debug, Default)]
struct State {
    next_id: u64,
    members: BTreeMap<MemberId, Member>,
    tickets: BTreeMap<u64, Ticket>,
    rotation: VecDeque<MemberId>,
    active: usize,
    queued: usize,
}

#[derive(Debug)]
pub(super) struct Admission {
    concurrency: usize,
    queue_limit: usize,
    state: Mutex<State>,
}

#[derive(Debug)]
pub(super) struct Lease {
    admission: Arc<Admission>,
    ticket: u64,
}

impl InferenceExecutionLease for Lease {}

impl Drop for Lease {
    /// Release this ticket only when the final execution-lease owner is dropped.
    fn drop(&mut self) {
        self.admission.release(self.ticket);
    }
}

/// Report admission pressure as HTTP 429 with a one-second retry hint.
fn unavailable(message: &str) -> InferenceError {
    InferenceError::from_kind(
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        InferenceErrorKind::RateLimit,
        message,
    )
    .with_retry_after_secs(1)
}

impl Admission {
    /// Create a shared execution budget and a bounded aggregate waiting queue.
    ///
    /// Zero concurrency is invalid; zero queue capacity permits immediate work only.
    pub(super) fn new(concurrency: usize, queue_limit: usize) -> anyhow::Result<Arc<Self>> {
        anyhow::ensure!(concurrency > 0, "shared concurrency must be positive");
        Ok(Arc::new(Self {
            concurrency,
            queue_limit,
            state: Mutex::default(),
        }))
    }

    /// Register an accepting membership with its own waiting limit, without
    /// increasing the host's aggregate execution or queue capacity.
    pub(super) fn register(&self, queue_limit: usize) -> MemberId {
        let mut state = self.state.lock().expect("admission lock poisoned");
        state.next_id += 1;
        let id = MemberId(state.next_id);
        state.members.insert(
            id,
            Member {
                accepting: true,
                retired: false,
                queue_limit,
                queue: VecDeque::new(),
                active: 0,
            },
        );
        id
    }

    /// Wait for this membership's turn and return a lease on shared capacity.
    ///
    /// Cancellation or dropping the waiting future releases its queued ticket.
    /// Admitted work keeps its slot until every clone of the lease is dropped.
    pub(super) async fn acquire(
        self: &Arc<Self>,
        member: MemberId,
        cancellation: CancellationToken,
    ) -> Result<Arc<Lease>, InferenceError> {
        if cancellation.is_cancelled() {
            return Err(InferenceError::cancelled(
                "request cancelled before admission",
            ));
        }
        let (lease, ready) = self.enqueue(member, cancellation.clone())?;
        tokio::select! {
            biased;
            () = cancellation.cancelled() => Err(InferenceError::cancelled("request cancelled while queued")),
            result = ready => {
                result.map_err(|_| unavailable("membership is not accepting requests"))?;
                Ok(lease)
            }
        }
    }

    /// Enforce both waiting limits, enqueue a FIFO ticket, and dispatch ready work.
    /// Existing waiters take precedence over a newly arriving request.
    fn enqueue(
        self: &Arc<Self>,
        member: MemberId,
        cancellation: CancellationToken,
    ) -> Result<(Arc<Lease>, oneshot::Receiver<()>), InferenceError> {
        let mut state = self.state.lock().expect("admission lock poisoned");
        let attachment = state
            .members
            .get(&member)
            .filter(|entry| entry.accepting)
            .ok_or_else(|| unavailable("membership is not accepting requests"))?;
        let must_queue = state.active >= self.concurrency || state.queued > 0;
        if must_queue
            && (state.queued >= self.queue_limit
                || attachment.queue.len() >= attachment.queue_limit)
        {
            return Err(unavailable("shared serving queue is full"));
        }
        state.next_id += 1;
        let ticket = state.next_id;
        let (send, ready) = oneshot::channel();
        let attachment = state
            .members
            .get_mut(&member)
            .expect("member checked above");
        let needs_turn = attachment.queue.is_empty();
        attachment.queue.push_back(ticket);
        if needs_turn {
            state.rotation.push_back(member);
        }
        state.queued += 1;
        state.tickets.insert(
            ticket,
            Ticket {
                member,
                ready: Some(send),
                cancellation,
            },
        );
        self.dispatch(&mut state);
        Ok((
            Arc::new(Lease {
                admission: self.clone(),
                ticket,
            }),
            ready,
        ))
    }

    /// Fill free execution slots by rotating among memberships with queued work.
    /// Disconnected waiters are removed without consuming native capacity.
    fn dispatch(&self, state: &mut State) {
        while state.active < self.concurrency {
            let Some(member_id) = state.rotation.pop_front() else {
                break;
            };
            let member = state
                .members
                .get_mut(&member_id)
                .expect("rotation member exists");
            let Some(ticket_id) = member.queue.pop_front() else {
                continue;
            };
            if !member.queue.is_empty() {
                state.rotation.push_back(member_id);
            }
            state.queued -= 1;
            let ticket = state
                .tickets
                .get_mut(&ticket_id)
                .expect("queued ticket exists");
            let ready = ticket.ready.take().expect("queued ticket has sender");
            // A disconnected waiter never consumes native capacity.
            if ready.send(()).is_err() {
                state.tickets.remove(&ticket_id);
                continue;
            }
            member.active += 1;
            state.active += 1;
        }
    }

    /// Remove a queued or active ticket, reclaim its budget, and wake the next turn.
    /// Retired memberships remain tracked until their last active ticket finishes.
    fn release(&self, ticket_id: u64) {
        let mut state = self.state.lock().expect("admission lock poisoned");
        let Some(ticket) = state.tickets.remove(&ticket_id) else {
            return;
        };
        let member = state
            .members
            .get_mut(&ticket.member)
            .expect("ticket member exists");
        if ticket.ready.is_some() {
            member.queue.retain(|id| *id != ticket_id);
            if member.queue.is_empty() {
                state.rotation.retain(|id| *id != ticket.member);
            }
            state.queued -= 1;
        } else {
            member.active -= 1;
            state.active -= 1;
        }
        if state
            .members
            .get(&ticket.member)
            .is_some_and(|member| member.retired && member.active == 0)
        {
            state.members.remove(&ticket.member);
        }
        self.dispatch(&mut state);
    }

    /// Pausing rejects queued/new requests but lets admitted execution finish.
    /// Detach additionally cancels active execution; its leases still own slots.
    pub(super) fn pause(&self, member_id: MemberId, cancel_active: bool) {
        let mut state = self.state.lock().expect("admission lock poisoned");
        let Some(member) = state.members.get_mut(&member_id) else {
            return;
        };
        member.accepting = false;
        let queued = std::mem::take(&mut member.queue);
        state.queued -= queued.len();
        state.rotation.retain(|id| *id != member_id);
        for id in queued {
            state.tickets.remove(&id);
        }
        if cancel_active {
            for ticket in state
                .tickets
                .values()
                .filter(|ticket| ticket.member == member_id)
            {
                ticket.cancellation.cancel();
            }
        }
        self.dispatch(&mut state);
    }

    /// Reopen admission for a paused membership; retired memberships stay closed.
    pub(super) fn resume(&self, member: MemberId) {
        if let Some(member) = self
            .state
            .lock()
            .expect("admission lock poisoned")
            .members
            .get_mut(&member)
        {
            member.accepting = !member.retired;
        }
    }

    /// Permanently close admission and signal cancellation of active requests.
    /// Their leases retain capacity until the workers actually release them.
    pub(super) fn retire(&self, member_id: MemberId) {
        self.pause(member_id, true);
        let mut state = self.state.lock().expect("admission lock poisoned");
        if let Some(member) = state.members.get_mut(&member_id) {
            member.retired = true;
            if member.active == 0 {
                state.members.remove(&member_id);
            }
        }
    }

    /// Snapshot a membership's admission counters, or an inactive default if absent.
    pub(super) fn status(&self, member: MemberId) -> AdmissionStatus {
        self.state
            .lock()
            .expect("admission lock poisoned")
            .members
            .get(&member)
            .map(|member| AdmissionStatus {
                accepting: member.accepting,
                active: member.active,
                queued: member.queue.len(),
            })
            .unwrap_or_default()
    }

    /// Count execution slots still held, including workers awaiting cancellation.
    pub(super) fn active(&self) -> usize {
        self.state.lock().expect("admission lock poisoned").active
    }

    /// Snapshot aggregate active and queued counts under the same lock.
    pub(super) fn totals(&self) -> (usize, usize) {
        let state = self.state.lock().expect("admission lock poisoned");
        (state.active, state.queued)
    }
}
