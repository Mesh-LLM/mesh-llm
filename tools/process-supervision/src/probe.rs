use super::control::Leader;
use super::observed::{self, Decision, Observation};
use super::{
    Cancellation, Completion, Failure, Limits, ObservedLine, Outcome, OutputFiles, ProcessReport,
    ProcessSpec, Readiness,
};
use std::time::Duration;

/// A candidate is provisional until the owner admits its live leader before expiry.
pub enum ProbeDecision<Rejection> {
    Pending,
    Candidate,
    Rejected(Rejection),
}

/// Identity and budgets from the owner's single monotonic start clock.
#[derive(Debug, Clone, Copy)]
pub struct ProbeContext {
    pub pid: u32,
    pub elapsed: Duration,
    pub remaining: Duration,
}

/// Trusted, scoped repository observer. Callbacks must not block, perform I/O,
/// spawn threads, log, panic, retain raw lines, or signal/wait/reap processes.
/// Lines only classify bounded borrowed bytes; ticks may exchange bounded messages
/// with an adapter-owned worker. Keep per-run typed facts, not raw output history.
/// No callback runs after a candidate/rejection or during cleanup. These work and
/// privacy requirements need adapter tests; Rust cannot enforce callback behavior.
pub trait ReadinessProbe {
    type Rejection;
    fn line(&mut self, line: ObservedLine<'_>) -> ProbeDecision<Self::Rejection>;
    fn tick(&mut self, context: ProbeContext) -> ProbeDecision<Self::Rejection>;
}

/// Probe readiness always requests a stop after admission. Supply None/Exit in
/// Limits; that existing specification remains unchanged for ordinary supervise.
pub struct Probe<'a, Observer> {
    pub observer: &'a mut Observer,
    pub deadline: Duration,
}

impl<Observer> Probe<'_, Observer> {
    pub(super) fn validate(&self, limits: &Limits) -> Result<(), Failure> {
        match (&limits.readiness, limits.completion) {
            (Readiness::None, Completion::Exit) => (),
            (Readiness::None, Completion::StopAfterReady)
            | (Readiness::Line { .. } | Readiness::ObservedLines { .. }, _) => {
                return Err(Failure::InvalidSpec("probe requires None/Exit limits"));
            }
        }
        if self.deadline.is_zero() || self.deadline > limits.execution {
            return Err(Failure::InvalidSpec("invalid probe deadline"));
        }
        Ok(())
    }
}

/// Adapter rejection stays typed and separate from process/output failures.
pub struct ProbeReport<Rejection> {
    pub process: ProcessReport,
    pub rejection: Option<Rejection>,
}

/// Runs a scoped stateful probe through the same owned spawn, admission, and
/// shutdown engine as supervise. No observer runs if validation or spawn fails.
pub fn supervise_with_probe<Observer: ReadinessProbe>(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    files: OutputFiles,
    probe: Probe<'_, Observer>,
) -> Result<ProbeReport<Observer::Rejection>, Failure> {
    probe.validate(limits)?;
    let mut rejection = None;
    let process = super::supervisor::supervise_owned(
        spec,
        limits,
        cancellation,
        files,
        |child, output, started| {
            let admission = Admission {
                limits,
                cancellation,
                elapsed: || started.elapsed(),
                deadline: probe.deadline,
                pid: child.id(),
            };
            let mut session = Session::new(probe.observer, &admission);
            let decision = session.monitor(child, output);
            rejection = session.rejection;
            match decision {
                Decision::Admitted(((), elapsed)) => (
                    Outcome::Ready,
                    true,
                    Some(super::report::StopObservation::Probe(elapsed)),
                ),
                Decision::Terminal(outcome) => (outcome, false, None),
                Decision::Failed(error) => {
                    output.failure = Some(error);
                    (Outcome::IoFailure, false, None)
                }
            }
        },
    )?;
    Ok(ProbeReport { process, rejection })
}

pub(super) trait ProbeLines {
    fn poll_probe(&mut self, callback: &mut dyn FnMut(ObservedLine<'_>)) -> Result<(), Failure>;
}

pub(super) struct Admission<'a, Clock> {
    pub(super) limits: &'a Limits,
    pub(super) cancellation: &'a Cancellation,
    pub(super) elapsed: Clock,
    pub(super) deadline: Duration,
    pub(super) pid: u32,
}

impl<Clock: Fn() -> Duration> Admission<'_, Clock> {
    fn check(&self) -> Result<Duration, Outcome> {
        if self.cancellation.is_cancelled() {
            return Err(Outcome::Cancelled);
        }
        let elapsed = (self.elapsed)();
        if elapsed >= self.limits.execution {
            return Err(Outcome::Deadline);
        }
        if elapsed >= self.deadline {
            return Err(Outcome::ReadinessDeadline);
        }
        Ok(elapsed)
    }
}

pub(super) struct Session<'a, Observer: ReadinessProbe, Clock> {
    observer: &'a mut Observer,
    admission: &'a Admission<'a, Clock>,
    decision: ProbeDecision<Observer::Rejection>,
    frozen: bool,
    pub(super) rejection: Option<Observer::Rejection>,
}

impl<'a, Observer: ReadinessProbe, Clock: Fn() -> Duration> Session<'a, Observer, Clock> {
    pub(super) fn new(observer: &'a mut Observer, admission: &'a Admission<'a, Clock>) -> Self {
        Self {
            observer,
            admission,
            decision: ProbeDecision::Pending,
            frozen: false,
            rejection: None,
        }
    }

    fn observing(&self) -> bool {
        !self.frozen
            && self.admission.check().is_ok()
            && matches!(self.decision, ProbeDecision::Pending)
    }

    pub(super) fn monitor(
        &mut self,
        child: &mut impl Leader,
        output: &mut impl ProbeLines,
    ) -> Decision<((), Duration)> {
        let decision = observed::monitor_with(
            child,
            &mut Cycle {
                session: self,
                output,
            },
        );
        self.frozen = true;
        if let ProbeDecision::Rejected(reason) =
            std::mem::replace(&mut self.decision, ProbeDecision::Pending)
        {
            self.rejection = Some(reason);
        }
        decision
    }
}

struct Cycle<'a, 'scope, Observer: ReadinessProbe, Clock, Output> {
    session: &'a mut Session<'scope, Observer, Clock>,
    output: &'a mut Output,
}

impl<Observer: ReadinessProbe, Clock: Fn() -> Duration, Output: ProbeLines> Observation
    for Cycle<'_, '_, Observer, Clock, Output>
{
    type Candidate = ();

    fn check(&self) -> Result<Duration, Outcome> {
        let elapsed = self.session.admission.check()?;
        match self.session.decision {
            ProbeDecision::Rejected(_) => Err(Outcome::ObservationRejected),
            ProbeDecision::Pending | ProbeDecision::Candidate => Ok(elapsed),
        }
    }

    fn poll(&mut self) -> Result<Option<()>, Failure> {
        let session = &mut self.session;
        if let Ok(elapsed) = session.admission.check()
            && session.observing()
        {
            session.decision = session.observer.tick(ProbeContext {
                pid: session.admission.pid,
                elapsed,
                remaining: session.admission.deadline.saturating_sub(elapsed),
            });
        }
        self.output.poll_probe(&mut |line| {
            if session.observing() {
                session.decision = session.observer.line(line);
            }
        })?;
        match session.decision {
            ProbeDecision::Candidate => Ok(Some(())),
            ProbeDecision::Pending | ProbeDecision::Rejected(_) => Ok(None),
        }
    }
}
