use super::{Disposition, Launch, MemberId, MemberReport, MemberState, Snapshot};
use crate::process::capture::Capture;
use crate::process::output::{Output, output_file};
use crate::process::{
    Failure, GracefulRequest, Limits, Outcome, ProcessReport, Readiness, ReadinessStop, Stream,
    control, platform,
};
use std::time::{Duration, Instant};
pub(super) struct Member {
    pub id: MemberId,
    pub child: platform::OwnedChild,
    pub output: Option<Output>,
    pub started: Instant,
    pub deadline: Duration,
    pub admitted: Option<Duration>,
    pub report: Option<MemberReport>,
}
impl Member {
    pub fn spawn(launch: Launch, limits: &Limits) -> Result<Self, Failure> {
        if launch.readiness_deadline.is_zero() || launch.readiness_deadline > limits.execution {
            return Err(Failure::InvalidSpec("invalid retained readiness deadline"));
        }
        let secrets = launch.spec.secrets()?;
        let mut command = launch.spec.command()?;
        let stdout_file = output_file(launch.files.stdout.as_deref())?;
        let stderr_file = output_file(launch.files.stderr.as_deref())?;
        let started = Instant::now();
        let mut child = platform::OwnedChild::spawn(&mut command)?;
        let stdout = child
            .stdout()
            .ok_or(Failure::InvalidSpec("stdout pipe missing"))?;
        let stderr = child
            .stderr()
            .ok_or(Failure::InvalidSpec("stderr pipe missing"))?;
        let output = Output {
            stdout: Capture::new(
                stdout,
                Stream::Stdout,
                stdout_file,
                limits.retained_bytes_per_stream,
                secrets.clone(),
            )?,
            stderr: Capture::new(
                stderr,
                Stream::Stderr,
                stderr_file,
                limits.retained_bytes_per_stream,
                secrets,
            )?,
            failure: None,
        };
        Ok(Self {
            id: launch.member,
            child,
            output: Some(output),
            started,
            deadline: launch.readiness_deadline,
            admitted: None,
            report: None,
        })
    }
    pub fn snapshot(&self) -> Snapshot {
        let state = match (&self.report, self.admitted) {
            (Some(_), _) => MemberState::IntentionalStop,
            (None, Some(elapsed)) => MemberState::Ready { elapsed },
            (None, None) => MemberState::Starting,
        };
        Snapshot {
            member: self.id,
            pid: self.child.id(),
            started: self.started,
            state,
        }
    }
    pub fn finish(&mut self, limits: &Limits, disposition: Disposition, mut observe: impl FnMut()) {
        let Some(mut output) = self.output.take() else {
            return;
        };
        let live_stop = matches!(
            disposition,
            Disposition::IntentionalStop | Disposition::SessionCleanup
        ) && self.admitted.is_some();
        let (cleanup, status, request) = if live_stop {
            control::shutdown_ready(&mut self.child, limits, || {
                output.poll(&Readiness::None);
                observe();
            })
        } else {
            let (cleanup, status) = control::shutdown(&mut self.child, limits, || {
                output.poll(&Readiness::None);
                observe();
            });
            (cleanup, status, GracefulRequest::SkippedInactiveTree)
        };
        let accepted = matches!(request, GracefulRequest::RequestedAfterLiveObservation);
        let disposition = match disposition {
            Disposition::SessionCleanup if !accepted => Disposition::Failure(match &request {
                GracefulRequest::LeaderExitedBeforeRequest => Outcome::EarlyExit,
                GracefulRequest::RequestedAfterLiveObservation
                | GracefulRequest::LeaderObservationFailed(_)
                | GracefulRequest::SkippedInactiveTree
                | GracefulRequest::TreeObservationFailed
                | GracefulRequest::RequestFailed(_) => Outcome::IoFailure,
            }),
            Disposition::IntentionalStop
                if !accepted
                    || !cleanup.complete
                    || cleanup.graceful_signal_failed
                    || cleanup.failure.is_some() =>
            {
                Disposition::Failure(Outcome::IoFailure)
            }
            other => other,
        };
        let until = Instant::now() + limits.forced_shutdown;
        while !output.stdout.eof() || !output.stderr.eof() {
            output.poll(&Readiness::None);
            observe();
            if output.failure.is_some() {
                break;
            }
            if Instant::now() >= until {
                output.failure = Some(Failure::CleanupDeadline);
                break;
            }
            std::thread::sleep(control::POLL);
        }
        observe();
        let readiness_stop = match self.admitted {
            Some(elapsed) if live_stop => ReadinessStop::ProbeAdmitted { elapsed, request },
            Some(_) | None => ReadinessStop::NotAdmitted,
        };
        let outcome = match disposition {
            Disposition::IntentionalStop | Disposition::SessionCleanup => Outcome::Ready,
            Disposition::Failure(outcome) => outcome,
        };
        let (stdout, stdout_failure) = output.stdout.finish();
        let (stderr, stderr_failure) = output.stderr.finish();
        self.report = Some(MemberReport {
            member: self.id,
            admitted: self.admitted,
            disposition,
            process: ProcessReport {
                pid: self.child.id(),
                outcome,
                status,
                ready: self.admitted.is_some(),
                readiness_stop,
                elapsed: self.started.elapsed(),
                stdout,
                stderr,
                cleanup,
                failure: output.failure.or(stdout_failure).or(stderr_failure),
            },
        });
    }
}
