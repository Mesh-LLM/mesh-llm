mod driver_expectations;
mod fixture;
mod observed_driver;
mod observed_fixture;
mod pipe_fixture;
#[expect(
    dead_code,
    reason = "shared process owner includes HTTPS APIs unused by this lifecycle driver"
)]
#[path = "../../src/process/mod.rs"]
pub mod process;
mod readiness_fixture;
mod retained_fixture;

use driver_expectations::{Scenario, driver_limits};
use process::*;
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("MIGRATION_PROCESS_MODE").is_some() {
        fixture::run()?;
        return Ok(());
    }
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "--help".to_owned());
    if mode == "--help" {
        println!(
            "migration_process_driver <exit|tree-exit|tree-hang|flood|observed-ready|observed-stubborn|observed-nonzero|observed-timeout>"
        );
        return Ok(());
    }
    if mode.starts_with("observed-") {
        return observed_driver::run(&mode);
    }
    let scenario = Scenario::parse(&mode)?;
    let mode = scenario.mode();
    let root = tempfile::tempdir()?;
    let mut environment = BTreeMap::from([
        ("MIGRATION_PROCESS_MODE".into(), Value::Public(mode.into())),
        (
            "MIGRATION_PROCESS_ROOT".into(),
            Value::Public(root.path().into()),
        ),
        ("MIGRATION_PROCESS_DRIVER".into(), Value::Public("1".into())),
    ]);
    for key in ["SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), Value::Public(value));
        }
    }
    let spec = ProcessSpec {
        executable: std::env::current_exe()?,
        arguments: Vec::new(),
        cwd: root.path().to_path_buf(),
        environment,
    };
    let limits = driver_limits();
    let mut sentinel_command = std::process::Command::new(std::env::current_exe()?);
    sentinel_command
        .env("MIGRATION_PROCESS_MODE", "sentinel")
        .env("MIGRATION_PROCESS_ROOT", root.path())
        .env("MIGRATION_PROCESS_DRIVER", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let mut sentinel = Sentinel(sentinel_command.spawn()?);
    let report = supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )?;
    let sentinel_alive = sentinel.0.try_wait()?.is_none();
    println!(
        "scenario={mode} pid={} outcome={:?} status={:?} elapsed_ms={} cleanup_complete={} forced={} stdout_seen={} stdout_retained={} stderr_seen={} stderr_retained={}",
        report.pid,
        report.outcome,
        report.status,
        report.elapsed.as_millis(),
        report.cleanup.complete,
        report.cleanup.forced,
        report.stdout.bytes_seen,
        report.stdout.bytes_retained.len(),
        report.stderr.bytes_seen,
        report.stderr.bytes_retained.len()
    );
    println!(
        "sentinel_pid={} sentinel_alive={sentinel_alive}",
        sentinel.0.id()
    );
    if !sentinel_alive {
        return Err("unrelated sentinel was killed".into());
    }
    scenario.validate(&report, root.path())?;
    Ok(())
}

struct Sentinel(std::process::Child);

impl Drop for Sentinel {
    fn drop(&mut self) {
        if let Err(error) = self.0.kill() {
            eprintln!("sentinel stop failed: {error}");
        }
        if let Err(error) = self.0.wait() {
            eprintln!("sentinel reap failed: {error}");
        }
    }
}
