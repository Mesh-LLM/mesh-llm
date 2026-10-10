//! Owned existing CLI composition; never launches a supplied binary directly.
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    },
};
use serde_json::Value;
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(in crate::automation) fn run(
    route: &str,
    input: &Value,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<(Value, bool)> {
    let (command, admission) = match route {
        "cache-family-plan" | "cache-family-correctness" | "cache-family-cell" => (route, false),
        "cache-family-cell-admission" => ("cache-family-cell", true),
        _ => return Err("unsupported cache matrix child".into()),
    };
    let remaining = until
        .saturating_duration_since(Instant::now())
        .saturating_sub(Duration::from_secs(14));
    if remaining.is_zero() || cancel.is_cancelled() {
        return Err("cache matrix child budget/cancellation refused".into());
    }
    std::fs::create_dir(directory)?;
    let request = directory.join("input.json");
    let output = directory.join("output");
    let bytes = serde_json::to_vec(input)?;
    crate::automation::receipt_files::fresh(&request, &bytes)?;
    let environment = ["PATH", "SYSTEMROOT", "WINDIR"]
        .into_iter()
        .filter_map(|k| std::env::var_os(k).map(|v| (k.into(), Arg::Public(v))))
        .collect();
    let report = process::supervise(
        &ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: {
                let mut args = vec!["automation".into(), command.into()];
                if admission {
                    args.push("admit-worker".into());
                }
                args.extend([
                    "--input".into(),
                    request.into_os_string(),
                    "--output".into(),
                    output.clone().into_os_string(),
                ]);
                args.into_iter().map(Arg::Public).collect()
            },
            cwd: directory.into(),
            environment,
        },
        &Limits {
            execution: remaining,
            graceful_shutdown: Duration::from_secs(12),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        OutputFiles {
            stdout: Some(directory.join("child.stdout.log")),
            stderr: Some(directory.join("child.stderr.log")),
        },
    )?;
    let clean = report.success()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
        && [&report.stdout, &report.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0);
    let leaf = match route {
        "cache-family-plan" | "cache-family-cell-admission" => output,
        "cache-family-correctness" => output.join("cache-correctness-stage.json"),
        "cache-family-cell" => output.join("cell.json"),
        _ => return Err("unsupported cache matrix child".into()),
    };
    let receipt = crate::automation::receipt_files::bounded(&leaf, 64 * 1024 * 1024)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    let correlation = !matches!(route, "cache-family-cell" | "cache-family-cell-admission")
        || receipt
            .as_ref()
            .is_some_and(|r| r["request_sha256"] == super::hash(&bytes));
    Ok((receipt.unwrap_or(Value::Null), clean && correlation))
}
