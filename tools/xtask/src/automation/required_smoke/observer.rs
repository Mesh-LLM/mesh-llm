use super::{Check, Progress, Rejection, Session, http};
use crate::{
    automation::daemon_readiness::observer::Observer as Daemon,
    process::{ObservedLine, ProbeContext, ProbeDecision, ReadinessProbe},
};
use std::{
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
    time::{Duration, Instant},
};

pub(super) struct Observer {
    pub daemon: Daemon,
    pub requests: SyncSender<http::Request>,
    pub responses: Receiver<http::ResultBody>,
    pub ports: crate::automation::daemon_readiness::ports::Ports,
    pub headless: bool,
    pub started: Instant,
    pub gated: bool,
    pub in_flight: bool,
    pub next: Duration,
}

impl Observer {
    pub(super) fn decision(
        &mut self,
        context: ProbeContext,
        session: &mut Session,
    ) -> Result<ProbeDecision<Rejection>, Rejection> {
        session.tick(self.started.elapsed())?;
        if !self.gated {
            match self.daemon.tick(context) {
                ProbeDecision::Pending => return Ok(ProbeDecision::Pending),
                ProbeDecision::Rejected(_) => return Err(Rejection::ProcessFailure),
                ProbeDecision::Candidate => self.gated = true,
            }
        }
        if self.in_flight {
            match self.responses.try_recv() {
                Ok(result) => {
                    self.in_flight = false;
                    let check = session.expected()?.ok_or(Rejection::OutOfOrder)?;
                    match session.observe((check, result.transfer()), self.started.elapsed())? {
                        Progress::Pending => self.next = context.elapsed + Duration::from_secs(1),
                        Progress::Advanced(_) | Progress::Complete => (),
                    }
                }
                Err(TryRecvError::Empty) => return Ok(ProbeDecision::Pending),
                Err(TryRecvError::Disconnected) => return Err(Rejection::WorkerFailed),
            }
        }
        let check = match session.expected()? {
            None => return Ok(ProbeDecision::Candidate),
            Some(Check::HeadlessModels) if !self.headless => return Ok(ProbeDecision::Candidate),
            Some(check) => check,
        };
        if context.elapsed < self.next {
            return Ok(ProbeDecision::Pending);
        }
        let port = match check {
            Check::Runtime
            | Check::RuntimeAttestation
            | Check::HeadlessStatus
            | Check::HeadlessAttestation
            | Check::InspectAttestation => self.ports.console,
            Check::Models | Check::HeadlessModels | Check::Chat | Check::Stream | Check::Auto => {
                self.ports.api
            }
        };
        let remaining = session.remaining(self.started.elapsed())?;
        self.requests
            .try_send(http::Request {
                check,
                port,
                model: session.model_id().unwrap_or_default().into(),
                deadline: Instant::now() + remaining.min(context.remaining),
            })
            .map_err(|_| Rejection::WorkerFailed)?;
        self.in_flight = true;
        Ok(ProbeDecision::Pending)
    }
}

impl Observer {
    pub(super) fn line(&mut self, line: ObservedLine<'_>) -> ProbeDecision<Rejection> {
        if self.gated {
            return ProbeDecision::Pending;
        }
        match self.daemon.line(line) {
            ProbeDecision::Candidate => {
                self.gated = true;
                ProbeDecision::Pending
            }
            ProbeDecision::Pending => ProbeDecision::Pending,
            ProbeDecision::Rejected(_) => ProbeDecision::Rejected(Rejection::ProcessFailure),
        }
    }
}
