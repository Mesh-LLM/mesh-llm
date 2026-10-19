//! `mesh-llm service-host`: what the Windows background service runs.
//!
//! The scheduled task starts it under `conhost.exe --headless`. Ending the task
//! terminates only that console host, and Windows leaves the processes it
//! hosted running, so the service host watches its parent and exits once it
//! has gone; a kill-on-close job object then takes `mesh-llm serve` down with
//! it. The node's stdout and stderr go to a log file, and a failed node is
//! restarted with an exponential backoff, up to a fixed number of restarts.

/// The service host's process id and start time, written beside its log so
/// `uninstall` can stop a host that outlived its task. Only the Windows
/// service host writes it.
#[cfg(any(windows, test))]
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
