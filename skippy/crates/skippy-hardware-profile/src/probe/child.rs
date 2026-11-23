//! Keep the leader waitable until its last process-tree signal.

use process_wrap::std::ChildWrapper;
use std::io;
use std::process::ExitStatus;

pub(super) struct ProbeChild {
    child: Box<dyn ChildWrapper>,
    status: Option<ExitStatus>,
}

impl ProbeChild {
    pub(super) fn new(child: Box<dyn ChildWrapper>) -> Self {
        Self {
            child,
            status: None,
        }
    }

    pub(super) fn stop(&mut self) -> io::Result<ExitStatus> {
        if let Some(status) = self.status {
            return Ok(status);
        }
        if let Err(error) = self.child.start_kill() {
            // A leader that already exited with an empty group leaves nothing
            // to signal (ESRCH on Linux, EPERM on macOS for a zombie-only
            // group); the wait below still reaps it. Any other error is real.
            if !(empty_group(&error) && self.leader_exited()) {
                return Err(error);
            }
        }
        // process-wrap's job-object `wait` ends in an INFINITE completion-port
        // read that blocks forever once the job's last-process packet was
        // already posted (#3067). Terminating the job kills every member, so
        // waiting on the leader alone is sufficient and always returns.
        #[cfg(windows)]
        let status = self.child.inner_mut().wait()?;
        #[cfg(not(windows))]
        let status = self.child.wait()?;
        self.status = Some(status);
        Ok(status)
    }

    /// Whether the leader has exited, without reaping it.
    fn leader_exited(&mut self) -> bool {
        #[cfg(unix)]
        let exited = {
            use rustix::process::{Pid, WaitId, WaitIdOptions, waitid};
            i32::try_from(self.child.id())
                .ok()
                .and_then(Pid::from_raw)
                .is_some_and(|pid| {
                    matches!(
                        waitid(
                            WaitId::Pid(pid),
                            WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
                        ),
                        Ok(Some(_))
                    )
                })
        };
        #[cfg(not(unix))]
        let exited = matches!(self.child.inner_mut().try_wait(), Ok(Some(_)));
        exited
    }

    pub(super) fn try_finish(&mut self) -> io::Result<Option<ExitStatus>> {
        #[cfg(unix)]
        let exited = {
            use rustix::process::{Pid, WaitId, WaitIdOptions, waitid};
            let pid = Pid::from_raw(i32::try_from(self.child.id()).map_err(io::Error::other)?)
                .ok_or_else(|| io::Error::other("invalid probe PID"))?;
            // Do not reap here: a descendant may still be alive, and reaping
            // would allow the group ID to be reused before the final signal.
            waitid(
                WaitId::Pid(pid),
                WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
            )?
            .is_some()
        };
        // Poll the leader only: the job-object `try_wait` drains the job's
        // completion port, which would hide the packets a later wait needs.
        #[cfg(not(unix))]
        let exited = self.child.inner_mut().try_wait()?.is_some();
        if exited {
            // Windows retains the job handle even after its leader exits.
            self.stop().map(Some)
        } else {
            Ok(None)
        }
    }
}

/// Signalling a group whose leader already exited finds no live member:
/// ESRCH on Linux, EPERM on macOS while only the unreaped leader remains.
fn empty_group(error: &io::Error) -> bool {
    #[cfg(unix)]
    let empty = matches!(
        rustix::io::Errno::from_io_error(error),
        Some(rustix::io::Errno::SRCH | rustix::io::Errno::PERM)
    );
    #[cfg(not(unix))]
    let empty = {
        let _ = error;
        false
    };
    empty
}

impl Drop for ProbeChild {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("failed to reap hardware probe: {error}");
        }
    }
}
