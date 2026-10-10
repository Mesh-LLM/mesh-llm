//! Exact pinned inspector argv under the existing raw capture owner.
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness,
        Value as Arg,
    },
};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn budget(until: Instant) -> DynResult<Duration> {
    let value = until
        .saturating_duration_since(Instant::now())
        .saturating_sub(Duration::from_secs(3));
    if value.is_zero() {
        return Err("MoE inspector cleanup reserve unavailable".into());
    }
    Ok(value)
}
pub(super) fn inspect(
    admitted: &Value,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    let model = admitted["model"]
        .as_str()
        .ok_or("MoE admitted model path absent")?;
    let binary = admitted["binary"]
        .as_str()
        .ok_or("MoE admitted inspector absent")?;
    let mut environment = admitted["environment"]
        .as_object()
        .ok_or("MoE admitted profile absent")?
        .iter()
        .map(|(k, v)| {
            v.as_str()
                .map(|v| (k.into(), Arg::Public(v.into())))
                .ok_or("MoE scalar profile invalid")
        })
        .collect::<Result<std::collections::BTreeMap<_, _>, _>>()?;
    let toolkits: std::collections::BTreeMap<
        String,
        crate::automation::cache_family_profile::Toolkit,
    > = serde_json::from_value(admitted["toolkit_directories"].clone())?;
    for (name, toolkit) in toolkits {
        environment.insert(name.into(), Arg::Public(toolkit.path.into_os_string()));
    }
    let mut spec = ProcessSpec {
        executable: binary.into(),
        arguments: vec![Arg::Public("inspect".into()), Arg::Public(model.into())],
        cwd: directory.into(),
        environment,
    };
    spec.environment.insert(
        "LLAMA_STAGE_BUILD_DIR".into(),
        Arg::Public(
            admitted["native_build"]
                .as_str()
                .ok_or("MoE native build absent")?
                .into(),
        ),
    );
    let report = process::supervise_raw(
        &spec,
        &Limits {
            execution: budget(until)?,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 16 * 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(16 * 1024 * 1024),
            stderr: None,
        },
    )?;
    let raw = report
        .stdout
        .as_ref()
        .ok_or("MoE raw inspector stdout absent")?;
    let p = &report.process;
    if !p.success()
        || !p.cleanup.complete
        || p.cleanup.forced
        || p.cleanup.graceful_signal_failed
        || p.cleanup.failure.is_some()
        || raw.as_bytes().len() as u64 != p.stdout.bytes_seen
        || [&p.stdout, &p.stderr]
            .iter()
            .any(|s| !s.line_capture_complete || s.truncated || s.oversized_lines > 0)
    {
        return Err("MoE inspector owned lifecycle/complete capture refused".into());
    }
    let value: Value = serde_json::from_slice(raw.as_bytes())?;
    super::publish(
        &directory.join("inspector-lifecycle.json"),
        &json!({"status":p.status.as_ref().and_then(std::process::ExitStatus::code),"stdout_bytes_seen":p.stdout.bytes_seen,"stderr_bytes_seen":p.stderr.bytes_seen,"stdout_suppressed":p.stdout.suppressed_lines,"scope":"bounded_raw_only_typed_tensor_projection_persisted"}),
    )?;
    Ok(value)
}
