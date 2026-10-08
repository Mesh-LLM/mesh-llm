//! Pinned regular-receipt helper child. No model-package dependency in xtask.
use super::super::{admission, bootstrap, execution};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Config {
    pub helper: admission::Artifact,
    pub helper_source: admission::Artifact,
    pub repo: String,
    pub parent_commit: String,
    pub credential_file: Option<PathBuf>,
    pub path_in_repo: String,
    #[serde(default)]
    pub credential_environment: bool,
    /// Reserved within the whole worker deadline, including existing3s child cleanup.
    #[serde(default = "default_export_budget")]
    pub export_budget_secs: u64,
}
fn default_export_budget() -> u64 {
    60
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Locator {
    pub schema_version: u32,
    pub transport_input_sha256: String,
    pub receipt_request_sha256: String,
    pub repo: String,
    pub parent_commit: String,
    pub commit_oid: String,
    pub path_in_repo: String,
    pub artifact_sha256: String,
    pub byte_size: u64,
    pub delivery_complete: bool,
}
#[derive(Serialize)]
struct Artifact {
    path: PathBuf,
    path_in_repo: String,
    sha256: String,
    byte_size: u64,
}
#[derive(Serialize)]
struct Input {
    schema_version: u32,
    repo: String,
    parent_commit: String,
    artifact: Artifact,
    receipt_request_sha256: String,
    credential_file: PathBuf,
    execution_timeout_ms: u64,
}
impl Config {
    pub(in crate::automation::hf_certify) fn validate(&self) -> DynResult<()> {
        let pieces = self.repo.split('/').collect::<Vec<_>>();
        let hex = bootstrap::contract::hex;
        if !(10..=3600).contains(&self.export_budget_secs)
            || pieces.len() != 2
            || pieces.iter().any(|p| {
                p.is_empty()
                    || matches!(*p, "." | "..")
                    || p.len() > 96
                    || !p
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            })
            || !hex(&self.parent_commit, 40)
            || !self.path_in_repo.ends_with("/native-job.json")
            || self.path_in_repo.len() > 256
            || self.path_in_repo.split('/').any(|p| {
                p.is_empty()
                    || matches!(p, "." | "..")
                    || !p
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            })
            || (self.credential_environment && self.credential_file.is_some())
            || (!self.credential_environment
                && self
                    .credential_file
                    .as_ref()
                    .is_none_or(|p| !p.is_absolute()))
            || [&self.helper, &self.helper_source]
                .iter()
                .any(|a| !a.path.is_absolute() || !hex(&a.sha256, 64))
        {
            return Err("native receipt export closed destination/pins refused".into());
        }
        Ok(())
    }
}
fn check(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err("receipt export cancelled or shared deadline expired".into());
    }
    Ok(())
}
fn pins(
    config: &Config,
    canonical: &[PathBuf; 2],
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    for (artifact, path) in [&config.helper, &config.helper_source]
        .into_iter()
        .zip(canonical)
    {
        check(deadline, cancel)?;
        if artifact.path.canonicalize()? != *path
            || bootstrap::execution::observe(path, deadline, cancel)? != artifact.sha256
        {
            return Err("receipt helper/source identity refused".into());
        }
    }
    check(deadline, cancel)
}
fn project(value: &Value, input: &Input) -> Value {
    let p = &value["publication"];
    let pin = |v: &Value| {
        v.as_str()
            .filter(|s| bootstrap::contract::hex(s, 64))
            .map(str::to_owned)
    };
    json!({"schema_version":value["schema_version"].as_u64(),"request_sha256":pin(&value["request_sha256"]),"status":value["status"].as_str().filter(|s|matches!(*s,"PUBLISHED_REGULAR_RECEIPT"|"FAILED"|"IN_PROGRESS")),"receipt_request_sha256":pin(&value["receipt_request_sha256"]),"artifact_sha256":pin(&value["artifact_sha256"]),"input_custody_verified":value["input_custody_verified"].as_bool(),"error":if value["error"].is_null(){None}else{Some("regular_receipt_operation_incomplete")},"publication":{"schema_version":p["schema_version"].as_u64(),"repo":if p["repo"]==input.repo{Some(&input.repo)}else{None},"parent_commit":if p["parent_commit"]==input.parent_commit{Some(&input.parent_commit)}else{None},"commit_oid":p["commit_oid"].as_str().filter(|s|bootstrap::contract::hex(s,40)),"completed":p["completed"].as_bool(),"source_custody_verified":p["source_custody_verified"].as_bool(),"mutation_attempted":p["mutation_attempted"].as_bool(),"remote_verified_paths":if p["remote_verified_paths"]==json!([input.artifact.path_in_repo]){json!([input.artifact.path_in_repo])}else{json!([])},"error":if p["error"].is_null(){None}else{Some("regular_operation_incomplete")}}})
}
fn correlated(value: &Value, input: &Input, hash: &str) -> DynResult<String> {
    let publication = &value["publication"];
    let oid = publication["commit_oid"]
        .as_str()
        .filter(|s| bootstrap::contract::hex(s, 40))
        .ok_or("regular receipt immutable commit absent")?;
    if value["schema_version"] != 1
        || value["request_sha256"] != hash
        || value["status"] != "PUBLISHED_REGULAR_RECEIPT"
        || value["receipt_request_sha256"] != input.receipt_request_sha256
        || value["artifact_sha256"] != input.artifact.sha256
        || value["input_custody_verified"] != true
        || publication["schema_version"] != 1
        || publication["repo"] != input.repo
        || publication["parent_commit"] != input.parent_commit
        || publication["completed"] != true
        || publication["source_custody_verified"] != true
        || publication["mutation_attempted"] != true
        || publication["remote_verified_paths"] != json!([input.artifact.path_in_repo])
        || !value["error"].is_null()
        || !publication["error"].is_null()
    {
        return Err("regular receipt process/status/correlation/custody refused".into());
    }
    Ok(oid.into())
}
fn observe(path: &Path, input: &Input, hash: &str, progress: bool) -> DynResult<Option<Value>> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Ok(m) if m.is_file() => (),
        _ => return Err("regular receipt metadata refused".into()),
    }
    let value: Value = serde_json::from_slice(&admission::read(path, 512 * 1024)?)
        .map_err(|_| "regular receipt JSON refused")?;
    if progress {
        let p = &value["publication"];
        let paths = p["remote_verified_paths"]
            .as_array()
            .ok_or("regular progress path roster refused")?;
        let valid_oid = p["commit_oid"].is_null()
            || p["commit_oid"]
                .as_str()
                .is_some_and(|s| bootstrap::contract::hex(s, 40));
        if value["schema_version"] != 1
            || value["request_sha256"] != hash
            || value["status"] != "IN_PROGRESS"
            || value["receipt_request_sha256"] != input.receipt_request_sha256
            || value["artifact_sha256"] != input.artifact.sha256
            || value["input_custody_verified"] != false
            || !value["error"].is_null()
            || p["schema_version"] != 1
            || p["repo"] != input.repo
            || p["parent_commit"] != input.parent_commit
            || p["completed"] != false
            || !p["error"].is_null()
            || !valid_oid
            || paths.len() > 1
            || (paths.len() == 1 && paths[0] != input.artifact.path_in_repo)
            || p["mutation_attempted"] != true
            || (!paths.is_empty() && p["commit_oid"].is_null())
            || (p["source_custody_verified"] != false && p["source_custody_verified"] != true)
            || (p["source_custody_verified"] == true && paths.len() != 1)
        {
            return Err("regular progress request/state correlation refused".into());
        }
    }
    Ok(Some(project(&value, input)))
}
pub(in crate::automation::hf_certify) fn execute(
    config: &Config,
    receipt_path: &Path,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    credential_override: Option<&Path>,
    evidence: &mut Value,
) -> DynResult<Locator> {
    config.validate()?;
    check(deadline, cancel)?;
    let canonical = [
        config.helper.path.canonicalize()?,
        config.helper_source.path.canonicalize()?,
    ];
    pins(config, &canonical, deadline, cancel)?;
    let bytes = admission::read(receipt_path, 1024 * 1024)?;
    let native: Value = serde_json::from_slice(&bytes)?;
    let receipt_sha = native["request_sha256"]
        .as_str()
        .filter(|s| bootstrap::contract::hex(s, 64))
        .ok_or("native receipt request hash absent")?;
    let transport = native["transport_input_sha256"]
        .as_str()
        .filter(|s| bootstrap::contract::hex(s, 64))
        .ok_or("native transport hash absent")?;
    let allowance = deadline
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .filter(|d| !d.is_zero())
        .ok_or("receipt export has no execution allowance after cleanup reserve")?;
    let input = Input {
        schema_version: 1,
        repo: config.repo.clone(),
        parent_commit: config.parent_commit.clone(),
        artifact: Artifact {
            path: receipt_path.into(),
            path_in_repo: config.path_in_repo.clone(),
            sha256: admission::digest(&bytes),
            byte_size: bytes.len() as u64,
        },
        receipt_request_sha256: receipt_sha.into(),
        credential_file: if config.credential_environment {
            credential_override
                .ok_or("private receipt publication credential absent")?
                .into()
        } else {
            config
                .credential_file
                .clone()
                .ok_or("explicit publication credential file absent")?
        },
        execution_timeout_ms: u64::try_from(allowance.as_millis())?,
    };
    let hash = admission::digest(&serde_json::to_vec(&input)?);
    let input_path = root.join("receipt-export-input.json");
    admission::publish(&input_path, &input)?;
    let output = root.join("receipt-publication");
    *evidence = json!({"request_sha256":hash,"mutation_possible":true,"process":null,"partial_progress":null,"progress_error":false,"publication":null,"final_error":false,"completed":false});
    let process = execution::run_process(
        &canonical[0],
        vec![
            "publish-regular-receipt".into(),
            "--input".into(),
            input_path.to_str().ok_or("receipt input Unicode")?.into(),
            "--output-directory".into(),
            output.to_str().ok_or("receipt output Unicode")?.into(),
        ],
        root,
        "receipt-export",
        deadline,
        cancel,
    )?;
    evidence["process"] = super::super::publication::process_observation(&process);
    let metadata = std::fs::symlink_metadata(&output)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() || output.canonicalize()? != output {
        return Err("receipt publisher output directory custody refused".into());
    }
    let progress = observe(&output.join("progress.json"), &input, &hash, true);
    let final_receipt = observe(&output.join("publication.json"), &input, &hash, false);
    for (result, key, error_key) in [
        (&progress, "partial_progress", "progress_error"),
        (&final_receipt, "publication", "final_error"),
    ] {
        match result {
            Ok(Some(value)) => evidence[key] = value.clone(),
            Err(_) => evidence[error_key] = json!(true),
            _ => (),
        }
    }
    if progress.is_err() || final_receipt.is_err() {
        return Err("regular publisher receipt metadata/correlation refused".into());
    }
    if !execution::clean(&process) {
        return Err("receipt publisher child refused; partial evidence retained".into());
    }
    let oid = correlated(&evidence["publication"], &input, &hash)?;
    pins(config, &canonical, deadline, cancel)?;
    if admission::digest(&admission::read(receipt_path, 1024 * 1024)?) != input.artifact.sha256 {
        return Err("native receipt bytes changed during export".into());
    }
    check(deadline, cancel)?;
    evidence["completed"] = json!(true);
    Ok(Locator {
        schema_version: 1,
        transport_input_sha256: transport.into(),
        receipt_request_sha256: receipt_sha.into(),
        repo: input.repo,
        parent_commit: input.parent_commit,
        commit_oid: oid,
        path_in_repo: input.artifact.path_in_repo,
        artifact_sha256: input.artifact.sha256,
        byte_size: input.artifact.byte_size,
        delivery_complete: false,
    })
}
#[cfg(test)]
#[path = "receipt_export/tests.rs"]
mod tests;
