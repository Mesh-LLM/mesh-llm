//! Existing process ownership, exact native argv and report correlation.
use super::admission::{self, Input, Mode, Receipt};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions,
        Readiness, Value,
    },
};
use serde_json::{Value as Json, json};
use std::{
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
fn allowance(deadline: Instant) -> DynResult<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .filter(|d| !d.is_zero())
        .ok_or_else(|| {
            "certification deadline has no execution allowance after cleanup reserve".into()
        })
}
fn environment() -> std::collections::BTreeMap<std::ffi::OsString, Value> {
    ["PATH", "SYSTEMROOT", "WINDIR"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name.into(), Value::Public(value))))
        .collect()
}
pub(in crate::automation) fn run_process(
    binary: &Path,
    args: Vec<String>,
    root: &Path,
    label: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<process::RawProcessReport> {
    if cancel.is_cancelled() {
        return Err("certification cancelled before launch".into());
    }
    let limits = Limits {
        execution: allowance(deadline)?,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1048576,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let spec = ProcessSpec {
        executable: binary.into(),
        arguments: args.into_iter().map(|a| Value::Public(a.into())).collect(),
        cwd: root.into(),
        environment: environment(),
    };
    Ok(process::supervise_raw_with_files(
        &spec,
        &limits,
        cancel,
        OutputFiles {
            stdout: Some(root.join(format!("{label}-stdout.log"))),
            stderr: Some(root.join(format!("{label}-stderr.log"))),
        },
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1048576),
            stderr: NonZeroUsize::new(1048576),
        },
    )?)
}
pub(in crate::automation) fn clean(report: &process::RawProcessReport) -> bool {
    let p = &report.process;
    p.outcome == process::Outcome::Exited
        && p.status.is_some_and(|s| s.success())
        && p.failure.is_none()
        && p.cleanup.complete
        && !p.cleanup.forced
        && !p.cleanup.graceful_signal_failed
        && p.cleanup.failure.is_none()
        && [
            (&p.stdout, report.stdout.as_ref()),
            (&p.stderr, report.stderr.as_ref()),
        ]
        .iter()
        .all(|(stream, raw)| {
            !stream.truncated
                && stream.line_capture_complete
                && raw.is_some_and(|v| v.as_bytes().len() as u64 == stream.bytes_seen)
        })
}
pub(super) fn identity(
    input: &Input,
    root: &Path,
    label: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<Input> {
    let path = root.join(format!("{label}-input.json"));
    let output = root.join(format!("{label}-receipt.json"));
    admission::publish(&path, input)?;
    let args = vec![
        "automation".into(),
        "hf-certify".into(),
        "identity-worker".into(),
        "--input".into(),
        path.to_str().ok_or("input Unicode")?.into(),
        "--output".into(),
        output.to_str().ok_or("output Unicode")?.into(),
    ];
    let report = run_process(
        &std::env::current_exe()?,
        args,
        root,
        label,
        deadline,
        cancel,
    )?;
    if !clean(&report) {
        return Err(format!("certification identity worker failed: {:?}", report.process).into());
    }
    let receipt: Receipt = serde_json::from_slice(&admission::read(&output, 1048576)?)?;
    if receipt.schema_version != 1
        || receipt.request_sha256 != admission::digest(&serde_json::to_vec(input)?)
    {
        return Err("identity receipt correlation refused".into());
    }
    receipt.admitted.validate()?;
    Ok(receipt.admitted)
}
pub(super) fn arguments(input: &Input) -> DynResult<Vec<String>> {
    let mut args = match input.mode {
        Mode::ProjectorOnly => vec!["validate-projector".into()],
        Mode::MtpAttach => vec!["validate-mtp-attach".into()],
    };
    if input.mode == Mode::MtpAttach {
        for part in &input.target_parts {
            args.extend([
                "--model".into(),
                part.path.to_str().ok_or("part Unicode")?.into(),
            ]);
        }
        args.extend([
            "--mtp-draft".into(),
            input
                .mtp_draft
                .as_ref()
                .ok_or("MTP draft")?
                .path
                .to_str()
                .ok_or("MTP Unicode")?
                .into(),
            "--layer-count".into(),
            input.layer_count.to_string(),
            "--ctx-size".into(),
            input.ctx_size.to_string(),
        ]);
        if let Some(n) = input.mtp_layer_count {
            args.extend(["--mtp-layer-count".into(), n.to_string()]);
        }
    }
    args.extend([
        "--projector".into(),
        input
            .projector
            .path
            .to_str()
            .ok_or("projector Unicode")?
            .into(),
        "--json".into(),
    ]);
    Ok(args)
}
pub(super) fn correlate(input: &Input, report: &Json) -> DynResult<()> {
    let expected: &[&str] = match input.mode {
        Mode::ProjectorOnly => &["projector", "warmup", "loaded"],
        Mode::MtpAttach => &[
            "projector",
            "model_parts",
            "mtp_draft",
            "layer_count",
            "mtp_layer_count",
            "ctx_size",
            "native_mtp_multimodal_feature",
            "session_created",
        ],
    };
    let object = report
        .as_object()
        .ok_or("native certification report must be object")?;
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err("native report unexpected or missing fields refused".into());
    }
    let path = |artifact: &admission::Artifact| serde_json::to_value(&artifact.path);
    if !report.is_object() || report.get("error").is_some() {
        return Err("native certification report object/error refused".into());
    }
    if report["projector"] != path(&input.projector)? {
        return Err("native report projector correlation refused".into());
    }
    match input.mode {
        Mode::ProjectorOnly => {
            if report["loaded"] != true || report["warmup"] != true {
                return Err("projector load/warmup not proven".into());
            }
        }
        Mode::MtpAttach => {
            let parts = input
                .target_parts
                .iter()
                .map(|p| &p.path)
                .collect::<Vec<_>>();
            if report["model_parts"] != serde_json::to_value(parts)?
                || report["mtp_draft"] != path(input.mtp_draft.as_ref().ok_or("MTP draft")?)?
                || report["layer_count"] != input.layer_count
                || report["ctx_size"] != input.ctx_size
                || report["session_created"] != true
                || report["native_mtp_multimodal_feature"] != true
                || report["mtp_layer_count"].as_u64().is_none_or(|n| {
                    n == 0 || input.mtp_layer_count.is_some_and(|v| u64::from(v) != n)
                })
            {
                return Err(
                    "native MTP report identity/feature/session correlation refused".into(),
                );
            }
        }
    }
    Ok(())
}
pub(super) fn execute(
    input: &Input,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Json,
) -> DynResult<()> {
    let admitted = identity(input, root, "before", deadline, cancel)?;
    evidence["admitted"] = serde_json::to_value(&admitted)?;
    let report = run_process(
        &admitted.binary.path,
        arguments(&admitted)?,
        root,
        "validation",
        deadline,
        cancel,
    )?;
    evidence["process"] = json!({"outcome":format!("{:?}",report.process.outcome),"status":report.process.status.and_then(|s|s.code()),"diagnostic":format!("{:?}",report.process),"stdout_suppressed_lines":report.process.stdout.suppressed_lines,"stderr_suppressed_lines":report.process.stderr.suppressed_lines});
    if !clean(&report) {
        return Err("native validation did not exit cleanly with complete bounded capture".into());
    }
    let value: Json = serde_json::from_slice(
        report
            .stdout
            .as_ref()
            .ok_or("missing native report")?
            .as_bytes(),
    )?;
    correlate(&admitted, &value)?;
    evidence["native_report"] = value;
    let final_identity = identity(input, root, "after", deadline, cancel)?;
    if final_identity != admitted {
        return Err("certification inputs changed across validation".into());
    }
    evidence["source_unchanged"] = Json::Bool(true);
    Ok(())
}
