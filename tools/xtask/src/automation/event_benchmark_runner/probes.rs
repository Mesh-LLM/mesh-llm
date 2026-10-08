//! Parent-owned bounded metadata probes; never run from a retained coordinator callback.
use crate::{
    command::DynResult,
    process::{self, Cancellation},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io::Read,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

fn probe_budget(remaining: Duration) -> Duration {
    // 200ms graceful + 200ms forced tree wait + 200ms separate EOF drain.
    remaining
        .saturating_sub(Duration::from_millis(600))
        .min(Duration::from_secs(5))
}

fn probe(
    executable: &Path,
    args: &[&str],
    budget: Duration,
    cancellation: &Cancellation,
) -> DynResult<Option<String>> {
    if budget.is_zero() || cancellation.is_cancelled() {
        return Ok(None);
    }
    let spec = process::ProcessSpec {
        executable: executable.to_owned(),
        cwd: executable
            .parent()
            .ok_or("probe requires parent directory")?
            .to_owned(),
        arguments: args
            .iter()
            .map(|arg| process::Value::Public((*arg).into()))
            .collect(),
        environment: BTreeMap::from([(
            OsString::from("PATH"),
            process::Value::Public("/usr/bin:/bin".into()),
        )]),
    };
    let limits = process::Limits {
        execution: budget,
        graceful_shutdown: Duration::from_millis(200),
        forced_shutdown: Duration::from_millis(200),
        retained_bytes_per_stream: 16 * 1024,
        readiness: process::Readiness::None,
        completion: process::Completion::Exit,
    };
    let report = match process::supervise_raw(
        &spec,
        &limits,
        cancellation,
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(16 * 1024),
            stderr: NonZeroUsize::new(16 * 1024),
        },
    ) {
        Ok(report) => report,
        Err(process::Failure::Io {
            operation: "spawn", ..
        }) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !report.process.cleanup.complete {
        return Err("metadata probe child cleanup incomplete".into());
    }
    if !report.process.success()
        || report.process.stdout.truncated
        || report.process.stderr.truncated
        || report.process.stdout.suppressed_lines > 0
        || report.process.stderr.suppressed_lines > 0
    {
        return Ok(None);
    }
    let bytes = report.stdout.ok_or("metadata probe stdout unavailable")?;
    let Ok(text) = std::str::from_utf8(bytes.as_bytes()) else {
        return Ok(None);
    };
    let text = text.trim();
    Ok((!text.is_empty()).then(|| text.to_owned()))
}
pub(super) fn version(
    binary: &Path,
    remaining: Duration,
    cancellation: &Cancellation,
) -> DynResult<Option<String>> {
    // Reserve cleanup time inside the caller's overall remaining budget.
    let budget = probe_budget(remaining);
    probe(binary, &["--version"], budget, cancellation)
}
pub(super) fn darwin_thermal(
    pmset: &Path,
    remaining: Duration,
    cancellation: &Cancellation,
) -> DynResult<Value> {
    let budget = probe_budget(remaining);
    Ok(
        match probe(pmset, &["-g", "therm"], budget, cancellation)? {
            Some(raw) => json!({"available":true,"source":"pmset -g therm","raw":raw}),
            None => json!({"available":false,"source":"pmset -g therm"}),
        },
    )
}
/// Bounded local reads belong inside the supervised worker, because filesystem reads can block.
pub(super) fn linux_thermal(root: &Path) -> Value {
    let Ok(entries) = std::fs::read_dir(root) else {
        return json!({"available":false,"source":"linux-sysfs"});
    };
    let mut temperatures = BTreeMap::new();
    // Fixed 64 names avoids following arbitrary directory entries or enumerating unbounded zones.
    drop(entries);
    for index in 0..64 {
        let name = format!("thermal_zone{index}");
        let path: PathBuf = root.join(&name).join("temp");
        let Ok(file) = std::fs::File::open(path) else {
            continue;
        };
        let mut bytes = Vec::new();
        if file.take(65).read_to_end(&mut bytes).is_err() || bytes.len() > 64 {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        if let Ok(value) = text.trim().parse::<i64>()
            && (-100_000..=300_000).contains(&value)
        {
            temperatures.insert(name, value);
        }
    }
    json!({"available":!temperatures.is_empty(),"source":"linux-sysfs","temperature_millidegrees_celsius":temperatures})
}
#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
