//! Bounded checkout/sysctl probes; unavailable provenance is never inferred from the binary.
use crate::{
    command::DynResult,
    process::{self, Cancellation, Value as Arg},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
fn probe(
    tool: &Path,
    args: &[&str],
    cwd: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Option<String>> {
    let execution = until
        .saturating_duration_since(Instant::now())
        .saturating_sub(Duration::from_millis(600))
        .min(Duration::from_secs(2));
    if execution.is_zero() || cancel.is_cancelled() {
        return Ok(None);
    }
    let environment = BTreeMap::from([
        ("PATH".into(), Arg::Public("/usr/bin:/bin".into())),
        ("GIT_MASTER".into(), Arg::Public("1".into())),
        ("GIT_OPTIONAL_LOCKS".into(), Arg::Public("0".into())),
    ]);
    let raw = match process::supervise_raw(
        &process::ProcessSpec {
            executable: tool.into(),
            arguments: args.iter().map(|v| Arg::Public((*v).into())).collect(),
            cwd: cwd.into(),
            environment,
        },
        &process::Limits {
            execution,
            graceful_shutdown: Duration::from_millis(200),
            forced_shutdown: Duration::from_millis(200),
            retained_bytes_per_stream: 4096,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancel,
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(4096),
            stderr: NonZeroUsize::new(4096),
        },
    ) {
        Ok(value) => value,
        Err(process::Failure::Io {
            operation: "spawn", ..
        }) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let p = &raw.process;
    if !p.cleanup.complete || p.cleanup.graceful_signal_failed || p.cleanup.failure.is_some() {
        return Err("restart metadata child cleanup failed".into());
    }
    if !p.success()
        || p.failure.is_some()
        || p.cleanup.forced
        || !p.stdout.line_capture_complete
        || !p.stderr.line_capture_complete
        || p.stdout.truncated
        || p.stderr.truncated
        || p.stdout.suppressed_lines > 0
        || p.stderr.suppressed_lines > 0
    {
        return Ok(None);
    }
    let bytes = raw
        .stdout
        .as_ref()
        .ok_or("restart raw metadata output absent")?
        .as_bytes();
    if bytes.len() as u64 != p.stdout.bytes_seen {
        return Ok(None);
    }
    let text = std::str::from_utf8(bytes)?.trim();
    Ok((!text.is_empty() && text.len() <= 4096).then(|| text.to_owned()))
}
pub(super) fn checkout(
    tool: &Path,
    cwd: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    let describe = probe(
        tool,
        &["describe", "--always", "--dirty", "--tags"],
        cwd,
        until,
        cancel,
    )?;
    let commit = probe(tool, &["rev-parse", "HEAD"], cwd, until, cancel)?
        .filter(|v| [40, 64].contains(&v.len()) && v.bytes().all(|b| b.is_ascii_hexdigit()));
    Ok(
        json!({"source_sha":commit.unwrap_or_else(||"unknown".into()),"git_describe":describe.unwrap_or_else(||"unknown".into()),"source_scope":"checkout-probe-not-binary-build-attestation"}),
    )
}
pub(super) fn darwin(
    tool: &Path,
    cwd: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    let chip = probe(
        tool,
        &["-n", "machdep.cpu.brand_string"],
        cwd,
        until,
        cancel,
    )?;
    let machine = probe(tool, &["-n", "hw.model"], cwd, until, cancel)?;
    let memory = probe(tool, &["-n", "hw.memsize"], cwd, until, cancel)?
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0);
    Ok(
        json!({"chip":chip,"machine_model":machine,"physical_memory_bytes":memory,"memory_available":memory.is_some(),"memory_source":"sysctl -n hw.memsize"}),
    )
}
