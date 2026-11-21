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
        self.child.start_kill()?;
        let status = self.child.wait()?;
        self.status = Some(status);
        Ok(status)
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
        #[cfg(not(unix))]
        let exited = self.child.try_wait()?.is_some();
        if exited {
            // Windows retains the job handle even after its leader exits.
            self.stop().map(Some)
        } else {
            Ok(None)
        }
    }
}

impl Drop for ProbeChild {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("failed to reap hardware probe: {error}");
        }
    }
}
