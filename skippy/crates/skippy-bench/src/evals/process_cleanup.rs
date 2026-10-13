//! Owned eval child lifecycle; result polling always ends with bounded cleanup.
use super::CommandOutcome;
use anyhow::{Context, Result, bail};
#[cfg(unix)]
use std::io;
use std::{
    process::{Child, Command, ExitStatus},
    thread,
    time::{Duration, Instant},
};

const TERMINATION_GRACE: Duration = Duration::from_secs(5);
const REAP_BUDGET: Duration = Duration::from_secs(1);

pub(super) fn configure_child_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    #[cfg(not(unix))]
    let _ = command;
}

pub(super) fn wait_with_timeout(
    child: &mut Child,
    timeout: Option<Duration>,
) -> Result<CommandOutcome> {
    wait_with_timeout_observed(child, timeout, || Ok(()))
}

/// Observe this eval's concrete capture before polling and after cleanup.
/// Observation refusal takes the same owned cleanup path as polling failure.
pub(super) fn wait_with_timeout_observed(
    child: &mut Child,
    timeout: Option<Duration>,
    mut observe: impl FnMut() -> Result<()>,
) -> Result<CommandOutcome> {
    let result = wait_polled(child, timeout, TERMINATION_GRACE, |child| {
        observe()?;
        child.try_wait().context("poll harness command")
    });
    let final_observation = observe();
    match (result, final_observation) {
        (Err(error), Err(observation)) => Err(error.context(format!(
            "final eval capture observation also failed: {observation:#}"
        ))),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(outcome), Ok(())) => Ok(outcome),
    }
}

enum Completion {
    Exited(ExitStatus),
    TimedOut,
}

fn wait_polled(
    child: &mut Child,
    timeout: Option<Duration>,
    grace: Duration,
    mut poll: impl FnMut(&mut Child) -> Result<Option<ExitStatus>>,
) -> Result<CommandOutcome> {
    let completion = poll_completion(child, timeout, &mut poll);
    // A direct child's exit says nothing about its remaining process group.
    // Polling errors also take this path, before their original error returns.
    let needs_grace = !matches!(completion, Ok(Completion::Exited(_)));
    let cleanup = cleanup_child(child, needs_grace, grace);
    match (completion, cleanup) {
        (Err(error), Err(cleanup)) => {
            Err(error.context(format!("eval cleanup also failed: {cleanup:#}")))
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(Completion::Exited(status)), Ok(())) => Ok(CommandOutcome {
            exit_status: status.code(),
            success: status.success(),
            timed_out: false,
        }),
        (Ok(Completion::TimedOut), Ok(())) => Ok(CommandOutcome {
            exit_status: None,
            success: false,
            timed_out: true,
        }),
    }
}

fn poll_completion(
    child: &mut Child,
    timeout: Option<Duration>,
    poll: &mut impl FnMut(&mut Child) -> Result<Option<ExitStatus>>,
) -> Result<Completion> {
    let started = Instant::now();
    loop {
        if let Some(status) = poll(child).context("poll harness command")? {
            return Ok(Completion::Exited(status));
        }
        if timeout.is_some_and(|budget| started.elapsed() >= budget) {
            return Ok(Completion::TimedOut);
        }
        let pause = timeout.map_or(Duration::from_millis(250), |budget| {
            budget
                .saturating_sub(started.elapsed())
                .min(Duration::from_millis(250))
        });
        thread::sleep(pause);
    }
}

fn cleanup_child(child: &mut Child, needs_grace: bool, grace: Duration) -> Result<()> {
    let graceful = if needs_grace {
        signal_child(child, false).and_then(|()| wait_grace(child, grace))
    } else {
        Ok(())
    };
    // Always force the owned group, even if the direct child exited during
    // grace, or TERM/grace polling failed. Do not return before this attempt.
    let forced = signal_child(child, true);
    let reaped = reap_child(child, REAP_BUDGET);
    graceful?;
    forced?;
    reaped
}

fn wait_grace(child: &mut Child, grace: Duration) -> Result<()> {
    let deadline = Instant::now() + grace;
    loop {
        if child
            .try_wait()
            .context("poll terminated harness command")?
            .is_some()
            || Instant::now() >= deadline
        {
            return Ok(());
        }
        thread::sleep(
            deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(100)),
        );
    }
}

fn reap_child(child: &mut Child, budget: Duration) -> Result<()> {
    let deadline = Instant::now() + budget;
    loop {
        if child.try_wait().context("reap harness command")?.is_some() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("harness command reaping exceeded cleanup deadline");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(unix)]
fn signal_child(child: &mut Child, force: bool) -> Result<()> {
    let group = -libc::pid_t::try_from(child.id()).context("child PID overflow")?;
    let signal = if force { libc::SIGKILL } else { libc::SIGTERM };
    // SAFETY: callers configure this freshly spawned child's owned process group.
    if unsafe { libc::kill(group, signal) } == -1 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            // Attempt direct-child cleanup too, but never hide failed group custody.
            let _ = child.kill();
            return Err(error.into());
        }
        // If the original group disappeared while the direct child escaped
        // it, still terminate/reap our direct child. Escaped descendants are
        // outside this process-group guarantee.
        match child.try_wait() {
            Ok(Some(_)) => {}
            _ => child
                .kill()
                .context("terminate harness command after group exit")?,
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn signal_child(child: &mut Child, _force: bool) -> Result<()> {
    // std::process provides direct-child termination only; no Windows job
    // object is created here and descendant cleanup is not claimed.
    match child.try_wait() {
        Ok(Some(_)) => Ok(()),
        _ => child.kill().context("terminate harness command"),
    }
}

#[cfg(all(test, unix))]
#[path = "process_cleanup/tests.rs"]
mod tests;
