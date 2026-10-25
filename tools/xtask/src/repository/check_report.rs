//! The complete observable result of a ported repository check: both streams
//! and the exit status, emitted once so the legacy channel split survives.

use crate::command::DynResult;
use std::io::Write;

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct CheckReport {
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) code: i32,
}

impl CheckReport {
    pub(crate) fn success(stdout: String) -> Self {
        Self {
            stdout,
            ..Self::default()
        }
    }

    pub(crate) fn failure(stdout: String, stderr: String) -> Self {
        Self {
            stdout,
            stderr,
            code: 1,
        }
    }

    /// Argument errors keep the legacy argparse status of 2.
    pub(crate) fn usage(usage: &str, message: &str) -> Self {
        Self {
            stdout: String::new(),
            stderr: format!("usage: {usage}\nerror: {message}\n"),
            code: 2,
        }
    }

    /// Writes stdout then stderr and ends the process with the recorded
    /// status. A nonzero status is not an xtask error, so it bypasses the
    /// generic `error:` prefix that `main` adds to failures.
    pub(crate) fn emit(self) -> DynResult<()> {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(self.stdout.as_bytes())?;
        stdout.flush()?;
        let mut stderr = std::io::stderr().lock();
        stderr.write_all(self.stderr.as_bytes())?;
        stderr.flush()?;
        if self.code != 0 {
            std::process::exit(self.code);
        }
        Ok(())
    }
}
