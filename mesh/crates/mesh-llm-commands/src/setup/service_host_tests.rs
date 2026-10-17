use super::service_host::{
    Backoff, ConsoleHost, FIRST_RESTART_DELAY, HostOutcome, MAX_RESTARTS, RestartDecision,
    RunningServe, STABLE_RUN, ServeExit, ServeLauncher, supervise,
};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::io;
use std::rc::Rc;
use std::time::Duration;

const QUICK: Duration = Duration::from_secs(1);

fn backoff() -> Backoff {
    Backoff::new(FIRST_RESTART_DELAY, MAX_RESTARTS, STABLE_RUN)
}

fn restart_after(seconds: u64, restart: u32) -> RestartDecision {
    RestartDecision::RestartAfter {
        delay: Duration::from_secs(seconds),
        restart,
    }
}

#[test]
fn backoff_doubles_from_five_seconds_then_gives_up() {
    let mut backoff = backoff();
    let failure = ServeExit::Failure(Some(1));

    let decisions: Vec<_> = (0..6).map(|_| backoff.after_exit(failure, QUICK)).collect();

    assert_eq!(
        decisions,
        vec![
            restart_after(5, 1),
            restart_after(10, 2),
            restart_after(20, 3),
            restart_after(40, 4),
            restart_after(80, 5),
            RestartDecision::GiveUp,
        ]
    );
}

#[test]
fn backoff_counts_afresh_after_a_stable_run() {
    let mut backoff = backoff();
    let failure = ServeExit::Failure(None);
    for _ in 0..3 {
        backoff.after_exit(failure, QUICK);
    }

    assert_eq!(backoff.after_exit(failure, STABLE_RUN), restart_after(5, 1));
}

#[test]
fn backoff_stops_on_a_clean_exit() {
    let mut backoff = backoff();
    backoff.after_exit(ServeExit::Failure(Some(1)), QUICK);

    assert_eq!(
        backoff.after_exit(ServeExit::Success, QUICK),
        RestartDecision::Stop
    );
}

/// A serve that exits with `exit` after `polls` checks, or never.
struct FakeServe {
    polls: Option<usize>,
    exit: ServeExit,
    stopped: Rc<Cell<bool>>,
}

impl RunningServe for FakeServe {
    fn try_exit(&mut self) -> io::Result<Option<ServeExit>> {
        match &mut self.polls {
            Some(0) => Ok(Some(self.exit)),
            Some(polls) => {
                *polls -= 1;
                Ok(None)
            }
            None => Ok(None),
        }
    }

    fn stop(&mut self) {
        self.stopped.set(true);
    }
}

#[derive(Default)]
struct FakeLauncher {
    runs: VecDeque<(Option<usize>, ServeExit)>,
    launches: usize,
    stopped: Rc<Cell<bool>>,
}

impl FakeLauncher {
    fn exiting(exits: impl IntoIterator<Item = ServeExit>) -> Self {
        Self {
            runs: exits.into_iter().map(|exit| (Some(0), exit)).collect(),
            ..Self::default()
        }
    }
}

impl ServeLauncher for FakeLauncher {
    type Running = FakeServe;

    fn launch(&mut self) -> io::Result<FakeServe> {
        self.launches += 1;
        let (polls, exit) = self
            .runs
            .pop_front()
            .expect("the test planned every launch");
        Ok(FakeServe {
            polls,
            exit,
            stopped: Rc::clone(&self.stopped),
        })
    }
}

/// Records each wait; reports the console host gone from wait `gone_from` on.
#[derive(Default)]
struct FakeHost {
    gone_from: Option<usize>,
    waits: RefCell<Vec<Duration>>,
}

impl ConsoleHost for FakeHost {
    fn gone_within(&self, timeout: Duration) -> bool {
        let mut waits = self.waits.borrow_mut();
        waits.push(timeout);
        self.gone_from.is_some_and(|from| waits.len() > from)
    }
}

#[test]
fn supervise_restarts_a_failing_serve_until_it_exits_cleanly() {
    let failure = ServeExit::Failure(Some(1));
    let mut launcher = FakeLauncher::exiting([failure, failure, ServeExit::Success]);
    let host = FakeHost::default();
    let mut log = Vec::new();

    let outcome = supervise(&mut launcher, &host, &mut backoff(), &mut log).expect("supervise");

    assert_eq!(outcome, HostOutcome::ServeExited);
    assert_eq!(launcher.launches, 3);
    assert_eq!(
        *host.waits.borrow(),
        vec![Duration::from_secs(5), Duration::from_secs(10)]
    );
    let log = String::from_utf8(log).expect("log is UTF-8");
    assert!(log.contains("mesh-llm serve exited with code 1; restarting in 5 s (restart 1 of 5)"));
    assert!(log.contains("mesh-llm serve exited with code 1; restarting in 10 s (restart 2 of 5)"));
    assert!(log.contains("mesh-llm serve exited cleanly"));
}

#[test]
fn supervise_gives_up_after_the_restart_limit() {
    let failure = ServeExit::Failure(Some(101));
    let mut launcher = FakeLauncher::exiting([failure; 6]);
    let host = FakeHost::default();
    let mut log = Vec::new();

    let outcome = supervise(&mut launcher, &host, &mut backoff(), &mut log).expect("supervise");

    assert_eq!(outcome, HostOutcome::GaveUp(failure));
    assert_eq!(launcher.launches, 6);
    let log = String::from_utf8(log).expect("log is UTF-8");
    assert!(log.contains("exited with code 101; giving up after 5 restarts in a row"));
}

#[test]
fn supervise_stops_serve_when_the_task_ends() {
    let mut launcher = FakeLauncher {
        runs: VecDeque::from([(None, ServeExit::Success)]),
        ..FakeLauncher::default()
    };
    let host = FakeHost {
        gone_from: Some(2),
        ..FakeHost::default()
    };
    let mut log = Vec::new();

    let outcome = supervise(&mut launcher, &host, &mut backoff(), &mut log).expect("supervise");

    assert_eq!(outcome, HostOutcome::TaskEnded);
    assert!(launcher.stopped.get(), "the running serve must be stopped");
    assert_eq!(launcher.launches, 1);
    assert!(
        String::from_utf8(log)
            .expect("log is UTF-8")
            .contains("the scheduled task ended; stopped mesh-llm serve")
    );
}

#[test]
fn supervise_does_not_restart_once_the_task_ended_during_the_backoff() {
    let mut launcher = FakeLauncher::exiting([ServeExit::Failure(Some(1))]);
    let host = FakeHost {
        gone_from: Some(0),
        ..FakeHost::default()
    };
    let mut log = Vec::new();

    let outcome = supervise(&mut launcher, &host, &mut backoff(), &mut log).expect("supervise");

    assert_eq!(outcome, HostOutcome::TaskEnded);
    assert_eq!(launcher.launches, 1);
    assert_eq!(*host.waits.borrow(), vec![Duration::from_secs(5)]);
}
