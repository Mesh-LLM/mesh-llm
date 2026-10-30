use super::control::{Leader, POLL};
use super::{Cancellation, Failure, Limits, Outcome, Readiness, ReadinessObservation, Stream};
use std::time::Duration;

#[cfg(test)]
#[path = "observed_tests.rs"]
mod tests;

pub(super) trait Lines {
    fn poll_lines(&mut self, readiness: &Readiness) -> Result<Option<Stream>, Failure>;
}

pub(super) struct Admission<'a, Clock> {
    pub(super) limits: &'a Limits,
    pub(super) cancellation: &'a Cancellation,
    pub(super) elapsed: Clock,
}

pub(super) enum Decision<Candidate = ReadinessObservation> {
    Admitted(Candidate),
    Terminal(Outcome),
    Failed(Failure),
}

pub(super) trait Observation {
    type Candidate;
    fn check(&self) -> Result<Duration, Outcome>;
    fn poll(&mut self) -> Result<Option<Self::Candidate>, Failure>;
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
        match &self.limits.readiness {
            Readiness::ObservedLines { deadline, .. } | Readiness::Line { deadline, .. }
                if elapsed >= *deadline =>
            {
                Err(Outcome::ReadinessDeadline)
            }
            Readiness::None | Readiness::Line { .. } | Readiness::ObservedLines { .. } => {
                Ok(elapsed)
            }
        }
    }
}

pub(super) fn monitor(
    child: &mut impl Leader,
    output: &mut impl Lines,
    admission: &Admission<'_, impl Fn() -> Duration>,
) -> Decision {
    struct LineObservation<'a, Output, Clock> {
        output: &'a mut Output,
        admission: &'a Admission<'a, Clock>,
    }
    impl<Output: Lines, Clock: Fn() -> Duration> Observation for LineObservation<'_, Output, Clock> {
        type Candidate = Stream;
        fn check(&self) -> Result<Duration, Outcome> {
            self.admission.check()
        }
        fn poll(&mut self) -> Result<Option<Stream>, Failure> {
            self.output.poll_lines(&self.admission.limits.readiness)
        }
    }
    match monitor_with(child, &mut LineObservation { output, admission }) {
        Decision::Admitted((stream, elapsed)) => {
            Decision::Admitted(ReadinessObservation { stream, elapsed })
        }
        Decision::Terminal(outcome) => Decision::Terminal(outcome),
        Decision::Failed(error) => Decision::Failed(error),
    }
}

pub(super) fn monitor_with<Source: Observation>(
    child: &mut impl Leader,
    source: &mut Source,
) -> Decision<(Source::Candidate, Duration)> {
    loop {
        if let Err(outcome) = source.check() {
            return Decision::Terminal(outcome);
        }
        let candidate = match source.poll() {
            Ok(candidate) => candidate,
            Err(error) => return Decision::Failed(error),
        };
        if let Err(outcome) = source.check() {
            return Decision::Terminal(outcome);
        }
        let exited = child.exited();
        let elapsed = match source.check() {
            Ok(elapsed) => elapsed,
            Err(outcome) => return Decision::Terminal(outcome),
        };
        match exited {
            Ok(true) => return Decision::Terminal(Outcome::EarlyExit),
            Err(error) => return Decision::Failed(error),
            Ok(false) => (),
        }
        if let Some(candidate) = candidate {
            return Decision::Admitted((candidate, elapsed));
        }
        std::thread::sleep(POLL);
    }
}
