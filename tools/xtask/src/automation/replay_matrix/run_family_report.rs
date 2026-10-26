use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use std::io::Write;

pub(super) struct RunFamilyReport {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    code: i32,
}

impl RunFamilyReport {
    pub(super) fn success(stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        Self {
            stdout,
            stderr,
            code: 0,
        }
    }
    pub(super) fn failure(stderr: String) -> Self {
        Self {
            stdout: Vec::new(),
            stderr: stderr.into_bytes(),
            code: 1,
        }
    }
    pub(super) fn failed_child(stdout: Vec<u8>, stderr: String) -> Self {
        Self {
            stdout,
            stderr: stderr.into_bytes(),
            code: 1,
        }
    }
    pub(super) fn from_check(report: CheckReport) -> Self {
        Self {
            stdout: report.stdout.into_bytes(),
            stderr: report.stderr.into_bytes(),
            code: report.code,
        }
    }
    pub(super) fn emit(self) -> DynResult<()> {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&self.stdout)?;
        stdout.flush()?;
        let mut stderr = std::io::stderr().lock();
        stderr.write_all(&self.stderr)?;
        stderr.flush()?;
        if self.code != 0 {
            std::process::exit(self.code);
        }
        Ok(())
    }
}
