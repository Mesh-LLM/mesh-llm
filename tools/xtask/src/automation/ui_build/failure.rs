//! Supervised UI producer failure and admission status contracts.
use crate::process::{Outcome, ProcessReport};
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
#[error("UI {step} failed; logs={logs:?}")]
pub(super) struct Failure {
    pub step: String,
    pub report: Box<ProcessReport>,
    pub logs: PathBuf,
}

impl Failure {
    pub(super) fn code(&self, signal: Option<i32>) -> i32 {
        let report = &self.report;
        if !report.cleanup.complete || report.failure.is_some() || report.cleanup.failure.is_some()
        {
            return 125;
        }
        if let Some(signal) = signal {
            return 128 + signal;
        }
        match report.outcome {
            Outcome::Deadline => 124,
            Outcome::Cancelled => 130,
            Outcome::Exited => report.status.map_or(125, child_status),
            _ => 125,
        }
    }
}

fn child_status(status: std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status
            .code()
            .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
    }
    #[cfg(windows)]
    {
        status.code().unwrap_or(125)
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum Admission {
    #[error("UI build cancelled before producer admission")]
    Cancelled,
    #[error("UI build deadline exhausted before producer admission")]
    Deadline,
}
impl Admission {
    pub(super) fn code(&self, signal: Option<i32>) -> i32 {
        if let Some(signal) = signal {
            return 128 + signal;
        }
        match self {
            Self::Cancelled => 130,
            Self::Deadline => 124,
        }
    }
}
