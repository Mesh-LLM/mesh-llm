//! `mesh-llm service-host`: what the Windows background service runs.
//!
//! The scheduled task starts it under `conhost.exe --headless`. Ending the task
//! terminates only that console host, and Windows leaves the processes it
//! hosted running, so the service host watches its parent and exits once it
//! has gone; a kill-on-close job object then takes `mesh-llm serve` down with
//! it. The node's stdout and stderr go to a log file, and a failed node is
//! restarted with an exponential backoff, up to a fixed number of restarts.

// Only Windows runs the service host; elsewhere its tests are the only users.
#![cfg_attr(not(windows), allow(dead_code))]

use std::io::{self, Write};
use std::time::{Duration, Instant};

/// Delay before the first restart; each further restart doubles it.
pub(crate) const FIRST_RESTART_DELAY: Duration = Duration::from_secs(5);
/// Restarts allowed in a row before the host gives up.
pub(crate) const MAX_RESTARTS: u32 = 5;
/// A node that ran this long counts as healthy: its next failure starts the
/// count again, so a crash days apart is not the fifth of a crash loop.
pub(crate) const STABLE_RUN: Duration = Duration::from_secs(10 * 60);
/// How often a running node is checked while waiting on the console host.
const HOST_POLL: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServeExit {
    Success,
    /// A failure, with the exit code when the process had one.
    Failure(Option<i32>),
}

impl ServeExit {
    fn describe(self) -> String {
        match self {
            Self::Success => "exited cleanly".to_string(),
            Self::Failure(Some(code)) => format!("exited with code {code}"),
            Self::Failure(None) => "ended without an exit code".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RestartDecision {
    Stop,
    RestartAfter { delay: Duration, restart: u32 },
    GiveUp,
}

pub(crate) struct Backoff {
    first_delay: Duration,
    max_restarts: u32,
    stable_run: Duration,
    restarts: u32,
}

impl Backoff {
    pub(crate) fn new(first_delay: Duration, max_restarts: u32, stable_run: Duration) -> Self {
        Self {
            first_delay,
            max_restarts,
            stable_run,
            restarts: 0,
        }
    }

    pub(crate) fn max_restarts(&self) -> u32 {
        self.max_restarts
    }

    /// What to do once `mesh-llm serve` has exited after running `ran_for`.
    pub(crate) fn after_exit(&mut self, exit: ServeExit, ran_for: Duration) -> RestartDecision {
        if exit == ServeExit::Success {
            return RestartDecision::Stop;
        }
        if ran_for >= self.stable_run {
            self.restarts = 0;
        }
        if self.restarts >= self.max_restarts {
            return RestartDecision::GiveUp;
        }
        let delay = self
            .first_delay
            .saturating_mul(2_u32.saturating_pow(self.restarts));
        self.restarts += 1;
        RestartDecision::RestartAfter {
            delay,
            restart: self.restarts,
        }
    }
}

/// Starts one `mesh-llm serve`.
pub(crate) trait ServeLauncher {
    type Running: RunningServe;
    fn launch(&mut self) -> io::Result<Self::Running>;
}

pub(crate) trait RunningServe {
    fn try_exit(&mut self) -> io::Result<Option<ServeExit>>;
    fn stop(&mut self);
}

/// The console host the scheduled task runs the service host under.
pub(crate) trait ConsoleHost {
    /// Waits up to `timeout`; true once the console host has gone.
    fn gone_within(&self, timeout: Duration) -> bool;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostOutcome {
    /// `mesh-llm serve` exited cleanly.
    ServeExited,
    /// The task ended, so the node was stopped.
    TaskEnded,
    /// The node kept failing.
    GaveUp(ServeExit),
}

pub(crate) fn supervise<L: ServeLauncher>(
    launcher: &mut L,
    host: &dyn ConsoleHost,
    backoff: &mut Backoff,
    log: &mut dyn Write,
) -> io::Result<HostOutcome> {
    loop {
        note(log, "starting mesh-llm serve");
        let started = Instant::now();
        let mut serve = launcher.launch()?;
        let exit = loop {
            if let Some(exit) = serve.try_exit()? {
                break exit;
            }
            if host.gone_within(HOST_POLL) {
                serve.stop();
                note(log, "the scheduled task ended; stopped mesh-llm serve");
                return Ok(HostOutcome::TaskEnded);
            }
        };
        match backoff.after_exit(exit, started.elapsed()) {
            RestartDecision::Stop => {
                note(log, "mesh-llm serve exited cleanly");
                return Ok(HostOutcome::ServeExited);
            }
            RestartDecision::GiveUp => {
                note(
                    log,
                    &format!(
                        "mesh-llm serve {}; giving up after {} restarts in a row",
                        exit.describe(),
                        backoff.max_restarts()
                    ),
                );
                return Ok(HostOutcome::GaveUp(exit));
            }
            RestartDecision::RestartAfter { delay, restart } => {
                note(
                    log,
                    &format!(
                        "mesh-llm serve {}; restarting in {} s (restart {restart} of {})",
                        exit.describe(),
                        delay.as_secs(),
                        backoff.max_restarts()
                    ),
                );
                if host.gone_within(delay) {
                    note(log, "the scheduled task ended before the restart");
                    return Ok(HostOutcome::TaskEnded);
                }
            }
        }
    }
}

// The log is the only place the service host can report to, so a failed
// write is not worth stopping the node over.
pub(crate) fn note(log: &mut dyn Write, message: &str) {
    let now = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%:z");
    let _ = writeln!(log, "[mesh-llm service-host {now}] {message}");
    let _ = log.flush();
}

/// The service host's process id and start time, written beside its log so
/// `uninstall` can stop a host that outlived its task.
pub(crate) fn render_service_host_record(pid: u32, started_at_unix: i64) -> String {
    format!("{pid} {started_at_unix}\n")
}

pub(crate) fn parse_service_host_record(record: &str) -> Option<(u32, i64)> {
    let mut fields = record.split_whitespace();
    let pid = fields.next()?.parse().ok()?;
    let started_at_unix = fields.next()?.parse().ok()?;
    fields.next().is_none().then_some((pid, started_at_unix))
}

/// Entry point of the hidden `mesh-llm service-host` subcommand.
pub fn run_service_host() -> anyhow::Result<()> {
    #[cfg(windows)]
    {
        super::service_host_windows::run()
    }
    #[cfg(not(windows))]
    {
        anyhow::bail!("`mesh-llm service-host` only runs under the Windows background service")
    }
}
