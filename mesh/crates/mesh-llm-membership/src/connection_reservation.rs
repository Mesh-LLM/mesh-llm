//! Single-owner peer handshakes and cancellation-safe waiter cleanup.

use crate::state::MembershipState;
use anyhow::Result;
use iroh::EndpointId;
use tokio::sync::watch;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingConnectionAttemptId(pub u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PendingConnectionOutcome {
    Admitted,
    Failed(String),
}

impl PendingConnectionOutcome {
    pub fn into_result(self, peer_id: EndpointId) -> Result<()> {
        match self {
            Self::Admitted => Ok(()),
            Self::Failed(message) => {
                anyhow::bail!(
                    "connection attempt to {} failed: {message}",
                    peer_id.fmt_short()
                )
            }
        }
    }
}

#[derive(Clone)]
pub struct PendingConnectionHandshake {
    pub attempt_id: PendingConnectionAttemptId,
    pub outcome_rx: watch::Receiver<Option<PendingConnectionOutcome>>,
}

impl PendingConnectionHandshake {
    pub fn waiter(&self, peer_id: EndpointId) -> PendingConnectionWaiter {
        PendingConnectionWaiter {
            peer_id,
            attempt_id: self.attempt_id,
            outcome_rx: self.outcome_rx.clone(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.outcome_rx.has_changed().is_ok()
    }
}

pub struct PendingConnectionAttemptOwner {
    pub peer_id: EndpointId,
    pub attempt_id: PendingConnectionAttemptId,
    pub outcome_tx: watch::Sender<Option<PendingConnectionOutcome>>,
}

pub struct PendingConnectionWaiter {
    pub peer_id: EndpointId,
    pub attempt_id: PendingConnectionAttemptId,
    pub outcome_rx: watch::Receiver<Option<PendingConnectionOutcome>>,
}

pub enum PendingConnectionReservation {
    Owner(PendingConnectionAttemptOwner),
    Waiter(PendingConnectionWaiter),
}

impl MembershipState {
    pub fn pending_connection_is_active(&mut self, peer_id: EndpointId) -> bool {
        let Some(pending) = self.pending_connections.get(&peer_id) else {
            return false;
        };
        if pending.is_active() {
            return true;
        }
        self.pending_connections.remove(&peer_id);
        false
    }
    pub fn reserve_pending_connection(
        &mut self,
        peer_id: EndpointId,
    ) -> PendingConnectionReservation {
        if let Some(pending) = self
            .pending_connection_is_active(peer_id)
            .then(|| self.pending_connections.get(&peer_id))
            .flatten()
        {
            return PendingConnectionReservation::Waiter(pending.waiter(peer_id));
        }

        let attempt_id = PendingConnectionAttemptId(self.next_pending_connection_attempt);
        self.next_pending_connection_attempt = self.next_pending_connection_attempt.wrapping_add(1);
        let (outcome_tx, outcome_rx) = watch::channel(None);
        self.pending_connections.insert(
            peer_id,
            PendingConnectionHandshake {
                attempt_id,
                outcome_rx,
            },
        );
        PendingConnectionReservation::Owner(PendingConnectionAttemptOwner {
            peer_id,
            attempt_id,
            outcome_tx,
        })
    }

    pub fn remove_pending_connection_if_attempt(
        &mut self,
        peer_id: EndpointId,
        attempt_id: PendingConnectionAttemptId,
    ) {
        if self
            .pending_connections
            .get(&peer_id)
            .is_some_and(|pending| pending.attempt_id == attempt_id)
        {
            self.pending_connections.remove(&peer_id);
        }
    }
}

pub async fn finish_pending_connection(
    state: &tokio::sync::Mutex<MembershipState>,
    owner: PendingConnectionAttemptOwner,
    outcome: PendingConnectionOutcome,
) {
    owner.outcome_tx.send_replace(Some(outcome));
    let mut state = state.lock().await;
    if state
        .pending_connections
        .get(&owner.peer_id)
        .is_some_and(|pending| pending.attempt_id == owner.attempt_id)
    {
        state.pending_connections.remove(&owner.peer_id);
    }
}

pub async fn await_pending_connection(
    state: &tokio::sync::Mutex<MembershipState>,
    mut waiter: PendingConnectionWaiter,
) -> Result<()> {
    loop {
        let outcome = waiter.outcome_rx.borrow().clone();
        if let Some(outcome) = outcome {
            return outcome.into_result(waiter.peer_id);
        }
        if waiter.outcome_rx.changed().await.is_err() {
            state
                .lock()
                .await
                .remove_pending_connection_if_attempt(waiter.peer_id, waiter.attempt_id);
            anyhow::bail!(
                "connection attempt to {} ended without a terminal result",
                waiter.peer_id.fmt_short()
            )
        }
    }
}
