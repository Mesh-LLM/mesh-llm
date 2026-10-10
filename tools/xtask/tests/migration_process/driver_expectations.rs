use crate::process::{Outcome, ProcessReport};
use std::io;
use std::path::Path;
use std::time::Duration;

#[derive(Clone, Copy)]
pub enum Scenario {
    Exit,
    TreeExit,
    TreeHang,
    Flood,
}

impl Scenario {
    pub fn parse(value: &str) -> io::Result<Self> {
        match value {
            "exit" => Ok(Self::Exit),
            "tree-exit" => Ok(Self::TreeExit),
            "tree-hang" => Ok(Self::TreeHang),
            "flood" => Ok(Self::Flood),
            _ => Err(io::Error::other("unknown local QA scenario")),
        }
    }

    pub fn mode(self) -> &'static str {
        match self {
            Self::Exit => "exit",
            Self::TreeExit => "tree-exit",
            Self::TreeHang => "tree-hang",
            Self::Flood => "flood",
        }
    }

    pub fn validate(self, report: &ProcessReport, root: &Path) -> io::Result<()> {
        if !report.cleanup.complete || report.failure.is_some() || report.cleanup.failure.is_some()
        {
            return Err(io::Error::other("local QA cleanup failed"));
        }
        let status = report
            .status
            .ok_or_else(|| io::Error::other("missing leader exit status"))?;
        let expected = match self {
            Self::Exit | Self::Flood => report.outcome == Outcome::Exited && report.success(),
            Self::TreeExit => {
                report.outcome == Outcome::Exited
                    && status.code() == Some(0)
                    && report.cleanup.forced
                    && !report.success()
            }
            Self::TreeHang => {
                report.outcome == Outcome::Deadline
                    && report.cleanup.forced
                    && !report.success()
                    && timeout_status(report)
            }
        };
        if !expected {
            return Err(io::Error::other("unexpected scenario termination"));
        }
        for stream in [&report.stdout, &report.stderr] {
            if stream.bytes_retained.len() > 128 {
                return Err(io::Error::other("output retention exceeded"));
            }
        }
        match self {
            Self::Exit => {
                if !report
                    .stdout
                    .bytes_retained
                    .windows(6)
                    .any(|bytes| bytes == b"READY\n")
                    || !report
                        .stderr
                        .bytes_retained
                        .windows(11)
                        .any(|bytes| bytes == b"diagnostic\n")
                {
                    return Err(io::Error::other("missing expected exit output"));
                }
            }
            Self::Flood => {
                for stream in [&report.stdout, &report.stderr] {
                    if stream.bytes_seen < 4 * 1024 * 1024
                        || stream.bytes_retained.len() != 128
                        || !stream.truncated
                    {
                        return Err(io::Error::other("flood evidence incomplete"));
                    }
                }
            }
            Self::TreeExit | Self::TreeHang => (),
        }
        let modes: &[&str] = match self {
            Self::Exit => &["exit"],
            Self::Flood => &["flood"],
            Self::TreeExit => &["tree-exit", "branch", "leaf"],
            Self::TreeHang => &["tree-hang", "branch", "leaf"],
        };
        let mut recorded = std::collections::BTreeSet::new();
        for (index, mode) in modes.iter().enumerate() {
            let pid: u32 = std::fs::read_to_string(root.join(format!("{mode}.pid")))?
                .parse()
                .map_err(io::Error::other)?;
            if pid == 0 || !recorded.insert(pid) || (index == 0 && pid != report.pid) {
                return Err(io::Error::other("invalid scenario PID roster"));
            }
            #[cfg(unix)]
            {
                let pid = i32::try_from(pid).map_err(io::Error::other)?;
                // SAFETY: signal zero only queries each fixture-recorded PID.
                if unsafe { libc::kill(pid, 0) } == 0 {
                    return Err(io::Error::other("owned process survived driver cleanup"));
                }
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(error);
                }
            }
            println!("owned_pid={pid} cleanup_confirmed=true");
        }
        Ok(())
    }
}

#[cfg(unix)]
fn timeout_status(report: &ProcessReport) -> bool {
    use std::os::unix::process::ExitStatusExt;
    report
        .status
        .is_some_and(|status| status.signal() == Some(libc::SIGTERM))
}

#[cfg(windows)]
fn timeout_status(report: &ProcessReport) -> bool {
    let expected = if report.cleanup.graceful_signal_failed {
        1
    } else {
        -1073741510
    };
    report
        .status
        .is_some_and(|status| status.code() == Some(expected))
}

pub fn driver_limits() -> crate::process::Limits {
    crate::process::Limits {
        execution: Duration::from_secs(2),
        graceful_shutdown: Duration::from_millis(100),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 128,
        readiness: crate::process::Readiness::None,
        completion: crate::process::Completion::Exit,
    }
}
