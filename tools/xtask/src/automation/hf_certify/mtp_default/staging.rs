use super::super::{admission, bootstrap, execution};
use super::{
    check,
    contract::{Input, Staging},
};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions,
        Readiness, Value as Argument,
    },
};
use serde_json::{Value, json};
use std::{
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
pub(super) fn execute(
    input: &Input,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<Value> {
    check(deadline, cancel)?;
    if bootstrap::execution::observe(&input.staging_helper.path, deadline, cancel)?
        != input.staging_helper.sha256
    {
        return Err("staging helper pin mismatch".into());
    }
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .filter(|d| d.as_secs() > 0)
        .ok_or("staging execution/cleanup allowance exhausted")?;
    let request = Staging {
        schema_version: 1,
        checkpoint: input.checkpoint.clone(),
        tokenizer_source: input.tokenizer_source.clone(),
        tokenizer_profile: input.tokenizer_profile.path.clone(),
        tokenizer_profile_sha256: input.tokenizer_profile.sha256.clone(),
        output_directory: root.join("download"),
        credential_file: Some(input.credential_file.clone()),
        timeout_seconds: remaining.as_secs().min(86400),
        maximum_bytes: input.maximum_bytes,
    };
    let request_file = root.join("request.json");
    admission::publish(&request_file, &request)?;
    let bytes = admission::read(&request_file, 1048576)?;
    let environment = [("HF_HUB_DISABLE_IMPLICIT_TOKEN", "true")]
        .into_iter()
        .map(|(k, v)| (k.into(), Argument::Public(v.into())))
        .collect();
    let spec = ProcessSpec {
        executable: input.staging_helper.path.clone(),
        arguments: vec![
            Argument::Public("--input".into()),
            Argument::Public(request_file.as_os_str().into()),
        ],
        cwd: root.into(),
        environment,
    };
    let limits = Limits {
        execution: remaining,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1048576,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let raw = process::supervise_raw_with_files(
        &spec,
        &limits,
        cancel,
        OutputFiles {
            stdout: Some(root.join("stdout.log")),
            stderr: Some(root.join("stderr.log")),
        },
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1048576),
            stderr: NonZeroUsize::new(1048576),
        },
    )?;
    evidence["staging_process"] = super::super::publication::process_observation(&raw);
    let report =
        match std::fs::symlink_metadata(request.output_directory.join("checkpoint-stitch.json")) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            _ => serde_json::from_slice(&admission::read(
                &request.output_directory.join("checkpoint-stitch.json"),
                1048576,
            )?)?,
        };
    evidence["staging_receipt"] = report.clone();
    if !execution::clean(&raw)
        || report["schema_version"] != 1
        || report["status"] != "STAGED_NOT_CONVERTED"
        || !report["error"].is_null()
        || report["request_sha256"] != admission::digest(&serde_json::to_vec(&request)?)
        || report["request_transport_sha256"] != admission::digest(&bytes)
        || report["checkpoint_directory"] != json!(request.output_directory.join("mtp-src"))
    {
        return Err("staging child/receipt correlation refused".into());
    }
    if bootstrap::execution::observe(&input.staging_helper.path, deadline, cancel)?
        != input.staging_helper.sha256
        || admission::read(&request_file, 1048576)? != bytes
    {
        return Err("staging helper/request drift".into());
    }
    correlate(&request, &report)?;
    check(deadline, cancel)?;
    Ok(report)
}

pub(super) fn correlate(request: &Staging, receipt: &Value) -> DynResult<()> {
    let root = request.output_directory.join("mtp-src");
    let mut pins = request.checkpoint.files.clone();
    for (n, h) in &request.tokenizer_source.files {
        pins.entry(n.clone()).or_insert(h.clone());
    }
    let expected = serde_json::to_value(
        pins.iter()
            .map(|(name, hash)| admission::Artifact {
                path: root.join(name),
                sha256: hash.clone(),
            })
            .collect::<Vec<_>>(),
    )?;
    if receipt["checkpoint_files"] != expected
        || receipt["tokenizer_profile"]
            != json!({"path":request.output_directory.join("tokenizer-profile.json"),"sha256":request.tokenizer_profile_sha256})
        || receipt["checkpoint_source"]["repo"] != request.checkpoint.repo
        || receipt["checkpoint_source"]["revision"] != request.checkpoint.revision
        || receipt["checkpoint_source"]["files"] != json!(request.checkpoint.files)
        || receipt["tokenizer_source"]["repo"] != request.tokenizer_source.repo
        || receipt["tokenizer_source"]["revision"] != request.tokenizer_source.revision
        || receipt["tokenizer_source"]["files"] != json!(request.tokenizer_source.files)
    {
        return Err("checkpoint effective roster/source/profile correlation refused".into());
    }
    Ok(())
}
