//! Generic conversion under one Jobs deadline, with the existing immutable receipt exporter.
use super::{artifact_workspace, guard, operator};
use crate::{
    automation::{
        command_interrupt::Interrupt,
        hf_certify::{admission, bootstrap, job_worker::receipt_export},
    },
    command::DynResult,
    process::Cancellation,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    workflow: String,
    timeout_secs: u64,
    runner: admission::Artifact,
    operator: Value,
    receipt_export: receipt_export::Config,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    upload_artifact: Option<artifact_workspace::Request>,
}
impl Input {
    fn validate(&self) -> DynResult<()> {
        self.receipt_export.validate()?;
        if self.schema_version != 1
            || self.workflow != "generic-conversion"
            || !(30..=259200).contains(&self.timeout_secs)
            || !self.runner.path.is_absolute()
            || !bootstrap::contract::hex(&self.runner.sha256, 64)
            || self.operator["conversion"]["timeout_seconds"].as_u64() != Some(self.timeout_secs)
            || self.timeout_secs <= self.receipt_export.export_budget_secs + 8
            || !self.receipt_export.credential_environment
            || self.receipt_export.credential_file.is_some()
        {
            return Err("generic Jobs schema/budget/runner/explicit export secret refused".into());
        }
        operator::delivery_manifest(&self.operator, self.timeout_secs, None, false)?;
        let upload = self.operator["conversion"]["upload_only"] == true;
        if upload != self.upload_artifact.is_some() {
            return Err("generic Jobs upload-only complete mounted artifact required".into());
        }
        if let Some(artifact) = &self.upload_artifact {
            artifact.validate()?;
            let conversion = &self.operator["conversion"];
            if artifact.work_directory
                != Path::new(
                    conversion["work_directory"]
                        .as_str()
                        .ok_or("upload work path")?,
                )
                || artifact.target_prefix != conversion["target_prefix"]
                || artifact.output_basename != conversion["output_basename"]
                || artifact.requested_splits as u64
                    != conversion["expected_splits"]
                        .as_u64()
                        .ok_or("upload requested splits")?
                || artifact.timeout_seconds != self.timeout_secs
                || artifact.source_directory
                    != Path::new(conversion["source"].as_str().ok_or("upload source path")?)
            {
                return Err("generic Jobs upload workspace/request correlation refused".into());
            }
        }
        Ok(())
    }
}
fn runner(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    guard(until, cancel)?;
    let actual = std::env::current_exe()?.canonicalize()?;
    if input.runner.path.canonicalize()? != actual
        || bootstrap::execution::observe(&actual, until, cancel)? != input.runner.sha256
    {
        return Err("generic Jobs runner custody refused".into());
    }
    guard(until, cancel)
}
fn phase(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    credential: &Path,
    evidence: &mut Value,
) -> DynResult<()> {
    runner(input, until, cancel)?;
    let remaining = until
        .checked_duration_since(Instant::now())
        .map(|d| d.as_secs())
        .filter(|s| *s >= 5)
        .ok_or("generic Jobs phase allowance")?;
    if let Some(artifact) = &input.upload_artifact {
        artifact_workspace::materialize(artifact, root, until, cancel, evidence)?;
    }
    let bytes = operator::delivery_manifest(&input.operator, remaining, Some(credential), true)?;
    evidence["operator_request_sha256"] = json!(admission::digest(&bytes));
    let (observed, result) =
        operator::execute_inherited(&bytes, &root.join("operator"), until, cancel)?;
    evidence["operator"] = observed;
    result?;
    runner(input, until, cancel)?;
    guard(until, cancel)
}
fn root(output: &str, input: &Input) -> DynResult<std::path::PathBuf> {
    let requested = std::path::absolute(output)?;
    let parent = requested
        .parent()
        .ok_or("generic Jobs output parent")?
        .canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("generic Jobs output leaf")?);
    let conversion = &input.operator["conversion"];
    let source =
        Path::new(conversion["source"].as_str().ok_or("generic source path")?).canonicalize()?;
    let work = Path::new(
        conversion["work_directory"]
            .as_str()
            .ok_or("generic work path")?,
    );
    let work = if work.exists() {
        work.canonicalize()?
    } else {
        work.parent()
            .ok_or("generic work parent")?
            .canonicalize()?
            .join(work.file_name().ok_or("generic work leaf")?)
    };
    if root.starts_with(&source)
        || source.starts_with(&root)
        || root.starts_with(&work)
        || work.starts_with(&root)
    {
        return Err("generic Jobs source/work/evidence ancestry refused before mutation".into());
    }
    std::fs::create_dir(&root)?;
    Ok(root)
}
fn credential(root: &Path) -> DynResult<tempfile::NamedTempFile> {
    let token = std::env::var_os("MESH_HF_PUBLICATION_TOKEN")
        .ok_or("explicit Jobs publication secret absent")?;
    let token = token
        .to_str()
        .ok_or("Jobs publication secret Unicode refused")?;
    if token.is_empty()
        || token.len() > 8192
        || token
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        return Err("Jobs publication secret grammar refused".into());
    }
    use std::io::Write as _;
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    file.write_all(token.as_bytes())?;
    file.as_file_mut().sync_all()?;
    Ok(file)
}
fn bytes(flag: &str, value: &str) -> DynResult<Vec<u8>> {
    match flag {
        "--input" => admission::read(Path::new(value), 65536),
        "--mounted-input-environment" => super::super::job_request_transport::environment(value),
        "--input-environment" if value == "MESH_HF_JOB_INPUT" => {
            let value = std::env::var_os(value).ok_or("generic Jobs input secret absent")?;
            let value = value.to_str().ok_or("generic Jobs input Unicode refused")?;
            if value.is_empty() || value.len() > 65536 {
                return Err("generic Jobs input byte bound".into());
            }
            Ok(value.as_bytes().to_vec())
        }
        _ => Err("generic Jobs closed input transport".into()),
    }
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [flag, input, out_flag, output] = args else {
        return Err("generic-job-worker --input FILE|--input-environment MESH_HF_JOB_INPUT|--mounted-input-environment MESH_HF_JOB_REQUEST --output-directory FRESH".into());
    };
    if out_flag != "--output-directory" {
        return Err("generic Jobs closed flags".into());
    }
    let bytes = bytes(flag, input)?;
    let input: Input =
        serde_json::from_slice(&bytes).map_err(|_| "generic Jobs typed input refused")?;
    input.validate()?;
    let root = root(output, &input)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut evidence = json!({"schema_version":1,"workflow":"generic-conversion","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"transport_input_sha256":admission::digest(&bytes),"status":"FAILED","operator_request_sha256":null,"operator":null,"error":null,"image_observed":false,"cost_observed":false});
    let credential = credential(&root);
    let native_until = until - Duration::from_secs(input.receipt_export.export_budget_secs);
    let result = match credential.as_ref() {
        Ok(file) => phase(
            &input,
            &root,
            native_until,
            &cancel,
            file.path(),
            &mut evidence,
        ),
        Err(_) => Err("generic Jobs explicit credential admission failed".into()),
    };
    let native_complete = result.is_ok() && guard(native_until, &cancel).is_ok();
    if native_complete {
        evidence["status"] = json!("CONVERSION_COMPLETED");
    } else {
        evidence["error"] =
            json!("generic conversion phase incomplete; partial operator observations retained");
    }
    let native_path = root.join("native-job.json");
    let published = admission::publish(&native_path, &evidence);
    let mut export_evidence = json!({"completed":false});
    let exported = if published.is_ok() {
        receipt_export::execute(
            &input.receipt_export,
            &native_path,
            &root,
            until,
            &cancel,
            credential.as_ref().ok().map(|f| f.path()),
            &mut export_evidence,
        )
        .map(Some)
    } else {
        Err("generic Jobs native receipt publication failed".into())
    };
    let cleaned = credential.map_or(Ok(()), tempfile::NamedTempFile::close);
    let finished = interrupt.finish();
    let complete = native_complete
        && exported.is_ok()
        && cleaned.is_ok()
        && finished.is_ok()
        && guard(until, &cancel).is_ok();
    let mut locator = exported.ok().flatten();
    if let Some(value) = locator.as_mut() {
        value.delivery_complete = complete;
    }
    let delivery = json!({"schema_version":1,"workflow":"generic-conversion","status":if complete{"DELIVERED"}else{"FAILED"},"request_sha256":evidence["request_sha256"],"native_completed":native_complete,"export":export_evidence,"locator":locator,"image_observed":false,"cost_observed":false});
    admission::publish(&root.join("native-job-delivery.json"), &delivery)?;
    if let Some(value) = locator {
        println!("MESH_NATIVE_DELIVERY {}", serde_json::to_string(&value)?);
    }
    if complete {
        Ok(())
    } else {
        Err("generic Jobs conversion/export incomplete; inspect partial receipts".into())
    }
}
