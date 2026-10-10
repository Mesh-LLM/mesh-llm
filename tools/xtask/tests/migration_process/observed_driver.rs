use crate::process::*;
use std::collections::BTreeMap;
use std::time::Duration;

pub(super) fn run(mode: &str) -> Result<(), Box<dyn std::error::Error>> {
    if ![
        "observed-ready",
        "observed-stubborn",
        "observed-nonzero",
        "observed-timeout",
    ]
    .contains(&mode)
    {
        return Err("unknown observation QA scenario".into());
    }
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
    let limits = Limits {
        execution: Duration::from_secs(3),
        graceful_shutdown: Duration::from_millis(100),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 0,
        readiness: Readiness::ObservedLines {
            deadline: Duration::from_secs(1),
            matcher: |line| {
                line.ending == LineEnding::Lf
                    && line.bytes.strip_suffix(b"\r").unwrap_or(line.bytes) == b"READY"
            },
        },
        completion: Completion::StopAfterReady,
    };
    let mut sentinel = crate::Sentinel(
        std::process::Command::new(std::env::current_exe()?)
            .env("MIGRATION_PROCESS_MODE", "sentinel")
            .env("MIGRATION_PROCESS_ROOT", root.path())
            .env("MIGRATION_PROCESS_DRIVER", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?,
    );
    let report = supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )?;
    let sentinel_alive = sentinel.0.try_wait()?.is_none();
    let handled = root.path().join("observed-stop").is_file();
    println!(
        "scenario={mode} pid={} outcome={:?} status={:?} receipt={:?} elapsed_ms={} complete={} forced={} handler_marker={handled} success={} sentinel_pid={} sentinel_alive={sentinel_alive}",
        report.pid,
        report.outcome,
        report.status,
        report.readiness_stop,
        report.elapsed.as_millis(),
        report.cleanup.complete,
        report.cleanup.forced,
        report.success(),
        sentinel.0.id()
    );
    if !sentinel_alive
        || !report.cleanup.complete
        || report.failure.is_some()
        || report.cleanup.failure.is_some()
    {
        return Err("observation QA cleanup failed".into());
    }
    let requested = matches!(
        report.readiness_stop,
        ReadinessStop::Admitted {
            request: GracefulRequest::RequestedAfterLiveObservation,
            ..
        }
    );
    let expected = match mode {
        "observed-ready" => requested && report.success() && handled,
        "observed-stubborn" => requested && report.cleanup.forced && !report.success() && !handled,
        "observed-nonzero" => {
            requested
                && !report.success()
                && handled
                && report
                    .status
                    .is_some_and(|status| status.code() == Some(23))
        }
        "observed-timeout" => {
            report.outcome == Outcome::ReadinessDeadline
                && !report.ready
                && !requested
                && !report.success()
                && handled
        }
        _ => false,
    };
    if !expected {
        return Err("unexpected observation QA result".into());
    }
    drop(sentinel);
    root.close()?;
    Ok(())
}
