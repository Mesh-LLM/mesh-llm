//! Existing confirmed layer-helper upload and read-only immutable resume child contracts.
use super::*;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
pub(in crate::automation::hf_certify) struct Context<'a> {
    pub root: &'a Path,
    pub until: Instant,
    pub cancel: &'a Cancellation,
    pub evidence: &'a mut Value,
    pub label: &'a str,
}
#[derive(Deserialize, Serialize)]
pub(super) struct UploadOptions {
    repo: String,
    revision: String,
    artifact: PathBuf,
    relative_path: String,
    credential_file: PathBuf,
    output_directory: PathBuf,
    maximum_attempts: u8,
    timeout_seconds: u64,
    dataset: bool,
    create_pr: bool,
    unlink_after_success: bool,
    admit_only: bool,
    confirm: bool,
}
#[derive(Deserialize, Serialize)]
pub(super) struct VerifyOptions {
    repo: String,
    commit: String,
    artifact: PathBuf,
    relative_path: String,
    credential_file: PathBuf,
    output_directory: PathBuf,
    timeout_seconds: u64,
}
fn seconds(until: Instant, cancel: &Cancellation) -> DynResult<u64> {
    check(until, cancel)?;
    let n = until
        .saturating_duration_since(Instant::now())
        .as_secs()
        .min(86400);
    if n <= 3 {
        return Err("quant helper cleanup margin exhausted".into());
    }
    Ok(n - 3)
}
fn invoke(
    input: &Input,
    args: Vec<String>,
    root: &Path,
    label: &str,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    pin(&input.helper, until, cancel)?;
    pin(&input.helper_source, until, cancel)?;
    let p = execution::run_process(&input.helper.path, args, root, label, until, cancel)?;
    evidence[label] = publication::process_observation(&p);
    pin(&input.helper, until, cancel)?;
    pin(&input.helper_source, until, cancel)?;
    if !execution::clean(&p) {
        return Err(
            "quant publication helper incomplete; inspect retained process/progress".into(),
        );
    }
    check(until, cancel)
}
pub(in crate::automation::hf_certify) fn commit(value: &Value) -> DynResult<String> {
    value["attempts"]
        .as_array()
        .and_then(|a| a.last())
        .and_then(|a| a["commit_oid"].as_str())
        .filter(|s| bootstrap::contract::hex(s, 40))
        .map(str::to_owned)
        .ok_or_else(|| "quant immutable publication commit absent".into())
}
pub(in crate::automation::hf_certify) fn upload(
    input: &Input,
    artifact: &Path,
    path: &str,
    unlink: bool,
    context: Context<'_>,
) -> DynResult<Value> {
    let Context {
        root,
        until,
        cancel,
        evidence,
        label,
    } = context;
    let output = root.join(label);
    let seconds = seconds(until, cancel)?;
    let options = UploadOptions {
        repo: input.target_repo.clone(),
        revision: "main".into(),
        artifact: artifact.into(),
        relative_path: path.into(),
        credential_file: input.credential_file.clone(),
        output_directory: output.clone(),
        maximum_attempts: 8,
        timeout_seconds: seconds,
        dataset: false,
        create_pr: false,
        unlink_after_success: unlink,
        admit_only: false,
        confirm: true,
    };
    let request = admission::digest(&serde_json::to_vec(&options)?);
    let mut args = vec![
        "upload".into(),
        "--repo".into(),
        input.target_repo.clone(),
        "--revision".into(),
        "main".into(),
        "--artifact".into(),
        artifact.to_string_lossy().into(),
        "--relative-path".into(),
        path.into(),
        "--credential-file".into(),
        input.credential_file.to_string_lossy().into(),
        "--output-directory".into(),
        output.to_string_lossy().into(),
        "--maximum-attempts".into(),
        "8".into(),
        "--timeout-seconds".into(),
        seconds.to_string(),
        "--confirm".into(),
    ];
    if unlink {
        args.push("--unlink-after-success".into());
    }
    let result = invoke(input, args, root, label, until, cancel, evidence);
    if output.join("progress.json").exists() {
        let progress: Value =
            serde_json::from_slice(&admission::read(&output.join("progress.json"), 1048576)?)?;
        if progress["schema_version"] != 1
            || progress["request_sha256"] != request
            || progress["status"] != "INCOMPLETE"
            || progress["publication"]["repo"] != input.target_repo
            || progress["publication"]["path"] != path
        {
            return Err("quant upload progress correlation refused".into());
        }
        evidence[format!("{label}-progress")] = progress;
    }
    let value: Value =
        serde_json::from_slice(&admission::read(&output.join("upload.json"), 1048576)?)?;
    evidence[format!("{label}-receipt")] = value.clone();
    result?;
    let p = &value["publication"];
    let attempts = p["attempts"]
        .as_array()
        .ok_or("quant upload attempts absent")?;
    if value["schema_version"] != 1
        || value["request_sha256"] != request
        || value["status"] != "PUBLISHED"
        || p["repo"] != input.target_repo
        || p["revision"] != "main"
        || p["path"] != path
        || p["completed"] != true
        || !p["error"].is_null()
        || p["source_custody_verified"] != true
        || p["unlink_requested"] != unlink
        || p["unlinked"] != unlink
        || attempts.is_empty()
        || attempts.len() > 8
        || attempts.last().unwrap()["remote_verified"] != true
        || !attempts.last().unwrap()["error"].is_null()
        || !p["identity"]["sha256"]
            .as_str()
            .is_some_and(|s| bootstrap::contract::hex(s, 64))
        || !p["identity"]["byte_size"]
            .as_u64()
            .is_some_and(|n| n > 0 && n <= 1024_u64.pow(4))
    {
        return Err("quant upload exact receipt/source/immutable verification refused".into());
    }
    commit(p)?;
    check(until, cancel)?;
    Ok(p.clone())
}
pub(in crate::automation::hf_certify) fn verify(
    input: &Input,
    artifact: &admission::Artifact,
    commit: &str,
    path: &str,
    context: Context<'_>,
) -> DynResult<()> {
    let Context {
        root,
        until,
        cancel,
        evidence,
        label,
    } = context;
    let output = root.join(label);
    let seconds = seconds(until, cancel)?;
    let options = VerifyOptions {
        repo: input.target_repo.clone(),
        commit: commit.into(),
        artifact: artifact.path.clone(),
        relative_path: path.into(),
        credential_file: input.credential_file.clone(),
        output_directory: output.clone(),
        timeout_seconds: seconds,
    };
    let request = admission::digest(&serde_json::to_vec(&options)?);
    let args = vec![
        "verify-upload".into(),
        "--repo".into(),
        input.target_repo.clone(),
        "--commit".into(),
        commit.into(),
        "--artifact".into(),
        artifact.path.to_string_lossy().into(),
        "--relative-path".into(),
        path.into(),
        "--credential-file".into(),
        input.credential_file.to_string_lossy().into(),
        "--output-directory".into(),
        output.to_string_lossy().into(),
        "--timeout-seconds".into(),
        seconds.to_string(),
    ];
    let result = invoke(input, args, root, label, until, cancel, evidence);
    let value: Value = serde_json::from_slice(&admission::read(
        &output.join("verification.json"),
        1048576,
    )?)?;
    evidence[format!("{label}-receipt")] = value.clone();
    result?;
    let v = &value["verification"];
    if value["schema_version"] != 1
        || value["request_sha256"] != request
        || value["status"] != "IMMUTABLE_VERIFIED"
        || v["repo"] != input.target_repo
        || v["commit"] != commit
        || v["path"] != path
        || v["identity"]["sha256"] != artifact.sha256
        || v["identity"]["byte_size"] != std::fs::symlink_metadata(&artifact.path)?.len()
        || v["completed"] != true
        || !v["error"].is_null()
        || v["local_custody_verified"] != true
    {
        return Err("quant remote resume immutable byte verification refused".into());
    }
    pin(artifact, until, cancel)?;
    check(until, cancel)
}
