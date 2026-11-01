use super::{Cleanup, Failure, GracefulRequest, Limits};
use std::process::ExitStatus;
use std::thread;
use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "stop_tests.rs"]
mod stop_tests;

pub(super) const POLL: Duration = Duration::from_millis(5);

pub(super) trait Leader {
    fn exited(&mut self) -> Result<bool, Failure>;
}

pub(super) trait Tree: Leader {
    fn active(&mut self) -> Result<bool, Failure>;
    fn graceful(&mut self) -> Result<(), Failure>;
    fn force(&mut self) -> Result<(), Failure>;
    fn reap(&mut self) -> Result<Option<ExitStatus>, Failure>;
}

pub(super) fn shutdown(
    tree: &mut impl Tree,
    limits: &Limits,
    drain: impl FnMut(),
) -> (Cleanup, Option<ExitStatus>) {
    let (cleanup, status, _) = shutdown_owned(tree, limits, drain, StopKind::Cleanup);
    (cleanup, status)
}

pub(super) fn shutdown_ready(
    tree: &mut impl Tree,
    limits: &Limits,
    drain: impl FnMut(),
) -> (Cleanup, Option<ExitStatus>, GracefulRequest) {
    shutdown_owned(tree, limits, drain, StopKind::Readiness)
}

enum StopKind {
    Cleanup,
    Readiness,
}

fn shutdown_owned(
    tree: &mut impl Tree,
    limits: &Limits,
    mut drain: impl FnMut(),
    kind: StopKind,
) -> (Cleanup, Option<ExitStatus>, GracefulRequest) {
    let mut cleanup = Cleanup::default();
    let mut request = GracefulRequest::SkippedInactiveTree;
    match tree.active() {
        Ok(false) => (),
        Ok(true) => {
            (request, cleanup.graceful_signal_failed) = request_stop(tree, kind);
            match wait_empty(tree, Instant::now() + limits.graceful_shutdown, &mut drain) {
                Ok(true) => (),
                Ok(false) => cleanup.forced = true,
                Err(error) => {
                    cleanup.failure = Some(error);
                    cleanup.forced = true;
                }
            }
        }
        Err(error) => {
            request = GracefulRequest::TreeObservationFailed;
            cleanup.failure = Some(error);
            cleanup.forced = true;
        }
    }
    if cleanup.forced
        && let Err(error) = tree.force()
    {
        cleanup.failure = Some(error);
        return (cleanup, None, request);
    }
    let until = Instant::now() + limits.forced_shutdown;
    match wait_empty(tree, until, &mut drain) {
        Ok(true) => (),
        Ok(false) => {
            cleanup.failure = Some(Failure::CleanupDeadline);
            return (cleanup, None, request);
        }
        Err(error) => {
            cleanup.failure = Some(error);
            return (cleanup, None, request);
        }
    }
    loop {
        drain();
        match tree.reap() {
            Ok(Some(status)) => {
                cleanup.complete = true;
                return (cleanup, Some(status), request);
            }
            Ok(None) => (),
            Err(error) => {
                cleanup.failure = Some(error);
                return (cleanup, None, request);
            }
        }
        if Instant::now() >= until {
            cleanup.failure = Some(Failure::CleanupDeadline);
            return (cleanup, None, request);
        }
        thread::sleep(POLL);
    }
}

fn request_stop(tree: &mut impl Tree, kind: StopKind) -> (GracefulRequest, bool) {
    let rejected = match kind {
        StopKind::Cleanup => None,
        StopKind::Readiness => match tree.exited() {
            Ok(false) => None,
            Ok(true) => Some(GracefulRequest::LeaderExitedBeforeRequest),
            Err(error) => Some(GracefulRequest::LeaderObservationFailed(error)),
        },
    };
    let signal = tree.graceful();
    let failed = signal.is_err();
    let request = match rejected {
        Some(rejected) => rejected,
        None => match signal {
            Ok(()) => GracefulRequest::RequestedAfterLiveObservation,
            Err(error) => GracefulRequest::RequestFailed(error),
        },
    };
    (request, failed)
}

fn wait_empty(
    tree: &mut impl Tree,
    until: Instant,
    drain: &mut impl FnMut(),
) -> Result<bool, Failure> {
    loop {
        drain();
        if !tree.active()? {
            return Ok(true);
        }
        if Instant::now() >= until {
            return Ok(false);
        }
        thread::sleep(POLL);
    }
}

impl Leader for super::platform::OwnedChild {
    fn exited(&mut self) -> Result<bool, Failure> {
        self.exited()
    }
}

impl Tree for super::platform::OwnedChild {
    fn active(&mut self) -> Result<bool, Failure> {
        self.active()
    }
    fn graceful(&mut self) -> Result<(), Failure> {
        self.graceful()
    }
    fn force(&mut self) -> Result<(), Failure> {
        self.force()
    }
    fn reap(&mut self) -> Result<Option<ExitStatus>, Failure> {
        self.reap()
    }
}
