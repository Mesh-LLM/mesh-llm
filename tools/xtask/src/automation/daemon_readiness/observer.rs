use super::{
    Rejection,
    correlation::{self, RequestId},
    http::{Endpoint, Request, Transfer, TransferError},
};
use crate::process::{ObservedLine, ProbeContext, ProbeDecision, ReadinessProbe};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/daemon/observer.rs"]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    StatusRetry { next: Duration },
    StatusInFlight,
    ModelsInFlight,
    Attribution,
    Complete,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Facts {
    pub(super) phase: Phase,
    pub(super) status: Option<Transfer>,
    pub(super) models: Option<Transfer>,
    pub(super) status_attempts: u64,
}

pub(crate) struct Observer {
    requests: SyncSender<Request>,
    responses: Receiver<Result<Transfer, TransferError>>,
    request_id: RequestId,
    terminals: [bool; 200],
    maximum_attempts: u64,
    facts: Facts,
}

impl Observer {
    pub(crate) fn new(
        requests: SyncSender<Request>,
        responses: Receiver<Result<Transfer, TransferError>>,
        request_id: RequestId,
        maximum_attempts: u64,
    ) -> Self {
        Self {
            requests,
            responses,
            request_id,
            maximum_attempts,
            terminals: [false; 200],
            facts: Facts {
                phase: Phase::StatusRetry {
                    next: Duration::ZERO,
                },
                status: None,
                models: None,
                status_attempts: 0,
            },
        }
    }

    pub(super) fn facts(&self) -> Facts {
        self.facts
    }

    fn send(&self, endpoint: Endpoint, remaining: Duration) -> ProbeDecision<Rejection> {
        let deadline = Instant::now() + remaining.min(Duration::from_secs(5));
        match self.requests.try_send(Request { endpoint, deadline }) {
            Ok(()) => ProbeDecision::Pending,
            Err(_) => ProbeDecision::Rejected(Rejection::WorkerFailed),
        }
    }

    fn candidate(&mut self) -> ProbeDecision<Rejection> {
        if self.facts.models.is_some_and(|transfer| {
            transfer
                .status_code
                .checked_sub(200)
                .and_then(|index| self.terminals.get(usize::from(index)))
                .copied()
                .unwrap_or(false)
        }) {
            self.facts.phase = Phase::Complete;
            ProbeDecision::Candidate
        } else {
            ProbeDecision::Pending
        }
    }

    fn receive(&mut self, context: ProbeContext) -> ProbeDecision<Rejection> {
        let result = match self.responses.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return ProbeDecision::Pending,
            Err(TryRecvError::Disconnected) => {
                return ProbeDecision::Rejected(Rejection::WorkerFailed);
            }
        };
        match result {
            Err(TransferError::Rejected(reason)) => ProbeDecision::Rejected(reason),
            Err(
                TransferError::Transport | TransferError::Timeout | TransferError::HttpStatus(_),
            ) => match self.facts.phase {
                Phase::StatusInFlight => {
                    self.facts.phase = Phase::StatusRetry {
                        next: context.elapsed + Duration::from_secs(1),
                    };
                    ProbeDecision::Pending
                }
                Phase::ModelsInFlight => ProbeDecision::Rejected(Rejection::ModelsTransferFailed),
                Phase::StatusRetry { .. } | Phase::Attribution | Phase::Complete => {
                    ProbeDecision::Rejected(Rejection::WorkerFailed)
                }
            },
            Ok(transfer) => match self.facts.phase {
                Phase::StatusInFlight => {
                    self.facts.status = Some(transfer);
                    self.facts.phase = Phase::ModelsInFlight;
                    self.send(
                        Endpoint::Models {
                            request_id: self.request_id,
                        },
                        context.remaining,
                    )
                }
                Phase::ModelsInFlight => {
                    self.facts.models = Some(transfer);
                    self.facts.phase = Phase::Attribution;
                    self.candidate()
                }
                Phase::StatusRetry { .. } | Phase::Attribution | Phase::Complete => {
                    ProbeDecision::Rejected(Rejection::WorkerFailed)
                }
            },
        }
    }
}

impl ReadinessProbe for Observer {
    type Rejection = Rejection;

    fn line(&mut self, line: ObservedLine<'_>) -> ProbeDecision<Rejection> {
        match self.facts.phase {
            Phase::ModelsInFlight | Phase::Attribution => {
                if let Some(status) = correlation::classify(line, self.request_id) {
                    self.terminals[usize::from(status - 200)] = true;
                }
                self.candidate()
            }
            Phase::StatusRetry { .. } | Phase::StatusInFlight | Phase::Complete => {
                ProbeDecision::Pending
            }
        }
    }

    fn tick(&mut self, context: ProbeContext) -> ProbeDecision<Rejection> {
        match self.facts.phase {
            Phase::StatusRetry { next } => {
                if context.elapsed < next || self.facts.status_attempts >= self.maximum_attempts {
                    return ProbeDecision::Pending;
                }
                self.facts.status_attempts += 1;
                self.facts.phase = Phase::StatusInFlight;
                self.send(
                    Endpoint::Status {
                        leader_pid: context.pid,
                    },
                    context.remaining,
                )
            }
            Phase::StatusInFlight | Phase::ModelsInFlight => self.receive(context),
            Phase::Attribution | Phase::Complete => ProbeDecision::Pending,
        }
    }
}
