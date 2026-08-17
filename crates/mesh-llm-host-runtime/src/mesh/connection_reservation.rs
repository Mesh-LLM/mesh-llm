use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PendingConnectionAttemptId(pub(crate) u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PendingConnectionOutcome {
    Admitted,
    Failed(String),
}

impl PendingConnectionOutcome {
    pub(crate) fn into_result(self, peer_id: EndpointId) -> Result<()> {
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
pub(crate) struct PendingConnectionHandshake {
    pub(crate) attempt_id: PendingConnectionAttemptId,
    pub(crate) outcome_rx: watch::Receiver<Option<PendingConnectionOutcome>>,
}

impl PendingConnectionHandshake {
    pub(crate) fn waiter(&self, peer_id: EndpointId) -> PendingConnectionWaiter {
        PendingConnectionWaiter {
            peer_id,
            attempt_id: self.attempt_id,
            outcome_rx: self.outcome_rx.clone(),
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.outcome_rx.has_changed().is_ok()
    }
}

pub(crate) struct PendingConnectionAttemptOwner {
    pub(crate) peer_id: EndpointId,
    pub(crate) attempt_id: PendingConnectionAttemptId,
    pub(crate) outcome_tx: watch::Sender<Option<PendingConnectionOutcome>>,
}

pub(crate) struct PendingConnectionWaiter {
    pub(crate) peer_id: EndpointId,
    pub(crate) attempt_id: PendingConnectionAttemptId,
    pub(crate) outcome_rx: watch::Receiver<Option<PendingConnectionOutcome>>,
}

pub(crate) enum PendingConnectionReservation {
    Owner(PendingConnectionAttemptOwner),
    Waiter(PendingConnectionWaiter),
}

impl MeshState {
    pub(crate) fn pending_connection_is_active(&mut self, peer_id: EndpointId) -> bool {
        let Some(pending) = self.pending_connections.get(&peer_id) else {
            return false;
        };
        if pending.is_active() {
            return true;
        }
        self.pending_connections.remove(&peer_id);
        false
    }
}
