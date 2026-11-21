//! Lifetime ownership for synchronous hardware probe commands.
//!
//! A survey runs synchronously on one thread. Its scope follows nested Mesh and
//! Skippy collectors on that thread without changing their public query APIs.
//! Cancellation prevents later fallback probes from starting, kills the current
//! process group/job and reaps its leader before returning.

use process_wrap::std::CommandWrap;
use std::cell::RefCell;
use std::io::{self, Read, Seek};
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

mod child;
use child::ProbeChild;

#[cfg(test)]
mod tests;

thread_local! {
    static CURRENT: RefCell<Option<Arc<ProbeScope>>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct State {
    cancelled: bool,
    child: Option<ProbeChild>,
}

/// An independently cancellable, sequential hardware survey.
#[derive(Default)]
pub struct ProbeScope {
    state: Mutex<State>,
    changed: Condvar,
}

impl ProbeScope {
    /// Run synchronous collectors with this survey's process ownership.
    pub fn run<T>(self: &Arc<Self>, collect: impl FnOnce() -> T) -> T {
        struct Restore(Option<Arc<ProbeScope>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.set(self.0.take());
            }
        }
        let _restore = Restore(CURRENT.replace(Some(Arc::clone(self))));
        collect()
    }

    /// Stop accepting probes, then kill and reap the in-flight tool tree.
    pub fn cancel(&self) -> io::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.cancelled = true;
        self.changed.notify_all();
        if let Some(child) = state.child.as_mut() {
            child.stop()?;
            state.child = None;
        }
        Ok(())
    }

    pub fn is_cancelled(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .cancelled
    }

    fn output(&self, command: &mut Command, first_line: bool) -> io::Result<Output> {
        // Files avoid pipe-reader threads that can outlive the probe, and allow
        // cancellation even when a tool closes stdout before it stops running.
        let (mut stdout, mut stderr) = match (tempfile::tempfile(), tempfile::tempfile()) {
            (Ok(stdout), Ok(stderr)) => (stdout, stderr),
            // shortcut: without a usable temp directory the probe runs as before
            // this scope existed (uncancellable), so the survey still reports
            // hardware; a missing TMPDIR is rare on hosts that serve models.
            _ => return command.stdin(Stdio::null()).output(),
        };
        command.stdin(Stdio::null());
        command.stdout(stdout.try_clone()?);
        command.stderr(stderr.try_clone()?);
        let mut wrapped = CommandWrap::from(std::mem::replace(command, Command::new("")));
        #[cfg(unix)]
        wrapped.wrap(process_wrap::std::ProcessGroup::leader());
        #[cfg(windows)]
        wrapped.wrap(process_wrap::std::JobObject);

        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.cancelled {
            *command = wrapped.into_command();
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "hardware survey stopped",
            ));
        }
        let spawned = wrapped.spawn();
        *command = wrapped.into_command();
        state.child = Some(ProbeChild::new(spawned?));
        let deadline = Instant::now() + Duration::from_secs(120);
        let status = loop {
            if state.cancelled {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "hardware survey stopped",
                ));
            }
            let sampled = if first_line {
                sample(&stdout)?.contains(&b'\n')
            } else {
                false
            };
            if sampled || Instant::now() >= deadline {
                if let Some(child) = state.child.as_mut() {
                    let status = child.stop()?;
                    state.child = None;
                    if sampled {
                        break status;
                    }
                }
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "hardware probe timed out",
                ));
            }
            if let Some(child) = state.child.as_mut() {
                match child.try_finish() {
                    Ok(Some(status)) => {
                        state.child = None;
                        break status;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        child.stop()?;
                        state.child = None;
                        return Err(error);
                    }
                }
            }
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(10))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        };
        drop(state);
        stdout.rewind()?;
        stderr.rewind()?;
        let mut output = Output {
            status,
            stdout: Vec::new(),
            stderr: Vec::new(),
        };
        stdout.read_to_end(&mut output.stdout)?;
        stderr.read_to_end(&mut output.stderr)?;
        Ok(output)
    }
}

/// Run a tool under the current survey, or a standalone scope for a direct query.
pub fn output(command: &mut Command) -> io::Result<Output> {
    let scope = CURRENT.with_borrow(Clone::clone).unwrap_or_default();
    scope.output(command, false)
}

/// Whether the current survey was stopped, so partial results are not published.
pub fn is_cancelled() -> bool {
    CURRENT.with_borrow(|scope| scope.as_ref().is_some_and(|scope| scope.is_cancelled()))
}

/// Sample a streaming probe, then kill and reap it instead of waiting for EOF.
/// Returns only the first line, so a malformed later record cannot spoil it.
pub fn first_line(command: &mut Command) -> io::Result<Vec<u8>> {
    let scope = CURRENT.with_borrow(Clone::clone).unwrap_or_default();
    scope
        .output(command, true)
        .map(|output| first_record(output.stdout))
}

fn first_record(mut stdout: Vec<u8>) -> Vec<u8> {
    if let Some(end) = stdout.iter().position(|&byte| byte == b'\n') {
        stdout.truncate(end);
    }
    stdout
}

/// Read what a live tool has written so far without moving the shared file
/// cursor, which would make the tool's next write overwrite captured bytes.
#[cfg(unix)]
fn sample(file: &std::fs::File) -> io::Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let mut bytes = vec![0; usize::try_from(file.metadata()?.len()).map_err(io::Error::other)?];
    let mut filled = 0;
    while filled < bytes.len() {
        match file.read_at(&mut bytes[filled..], filled as u64)? {
            0 => break,
            read => filled += read,
        }
    }
    bytes.truncate(filled);
    Ok(bytes)
}

// shortcut: Windows positional reads move the cursor; the only streaming probe
// (tegrastats) is Linux-only, so this path is not used there today.
#[cfg(not(unix))]
fn sample(file: &std::fs::File) -> io::Result<Vec<u8>> {
    let mut file = file.try_clone()?;
    file.rewind()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}
