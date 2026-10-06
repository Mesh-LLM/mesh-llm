//! Full default composition under one Jobs deadline and the existing immutable receipt exporter.
use super::{check as guard, contract, execute_inherited};
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
}
impl Input {
    fn validate(&self) -> DynResult<()> {
        self.receipt_export.validate()?;
        if self.schema_version != 1
            || self.workflow != "default-mtp-composition"
            || !(30..=259200).contains(&self.timeout_secs)
            || !self.runner.path.is_absolute()
            || !bootstrap::contract::hex(&self.runner.sha256, 64)
            || self.operator["overall_seconds"].as_u64() != Some(self.timeout_secs)
            || self.timeout_secs <= self.receipt_export.export_budget_secs + 8
            || !self.receipt_export.credential_environment
            || self.receipt_export.credential_file.is_some()
        {
            return Err(
                "composition Jobs schema/budget/runner/explicit export secret refused".into(),
            );
        }
        let mut op = self.operator.clone();
        if !op["credential_file"].is_null() {
            return Err("composition Jobs credential must be explicit secret".into());
        }
        op["credential_file"] = json!("/work/owned-publication-credential");
        let op: contract::Input = serde_json::from_value(op)?;
        op.validate()?;
        if op.overall_seconds != self.timeout_secs
            || op.bootstrap.timeout_seconds != self.timeout_secs
        {
            return Err("composition Jobs budget correlation".into());
        }

        Ok(())
    }
}
fn runner(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    guard(until, cancel)?;
    let actual = std::env::current_exe()?.canonicalize()?;
    if input.runner.path.canonicalize()? != actual
        || bootstrap::execution::observe_runner(&actual, until, cancel)? != input.runner.sha256
    {
        return Err("composition Jobs runner custody refused".into());
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
        .ok_or("composition Jobs phase allowance")?;
    let mut op = input.operator.clone();
    op["credential_file"] = json!(credential);
    op["overall_seconds"] = json!(remaining);
    let op: contract::Input = serde_json::from_value(op)?;
    op.validate()?;
    let bytes = serde_json::to_vec(&op)?;
    evidence["operator_request_sha256"] = json!(admission::digest(&bytes));
    let (observed, result) = execute_inherited(&bytes, &root.join("operator"), until, cancel)?;
    evidence["operator"] = observed;
    result?;
    runner(input, until, cancel)?;
    guard(until, cancel)
}
fn root(output: &str, input: &Input) -> DynResult<std::path::PathBuf> {
    let requested = std::path::absolute(output)?;
    let parent = requested
        .parent()
        .ok_or("composition Jobs output parent")?
        .canonicalize()?;
    let root = parent.join(
        requested
            .file_name()
            .ok_or("composition Jobs output leaf")?,
    );
    for key in ["target_parts", "sidecars"] {
        let values = input.operator[key]
            .as_array()
            .ok_or("composition source roster")?;
        for row in values {
            let artifact = if key == "sidecars" {
                &row["artifact"]
            } else {
                row
            };
            let source = Path::new(artifact["path"].as_str().ok_or("composition source path")?)
                .canonicalize()?;
            if source.starts_with(&root) || root.starts_with(&source) {
                return Err("composition Jobs evidence/source ancestry refused".into());
            }
        }
    }
    for key in [
        "staging_helper",
        "repository_helper",
        "repository_helper_source",
        "publisher_helper",
        "publisher_source",
        "tokenizer_profile",
    ] {
        let source = Path::new(
            input.operator[key]["path"]
                .as_str()
                .ok_or("composition helper path")?,
        )
        .canonicalize()?;
        if source.starts_with(&root) || root.starts_with(&source) {
            return Err("composition Jobs evidence/helper ancestry refused".into());
        }
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
        "--mounted-input-environment" => {
            crate::automation::hf_certify::job_request_transport::environment(value)
        }
        "--input-environment" if value == "MESH_HF_JOB_INPUT" => {
            let value = std::env::var_os(value).ok_or("composition Jobs input secret absent")?;
            let value = value
                .to_str()
                .ok_or("composition Jobs input Unicode refused")?;
            if value.is_empty() || value.len() > 65536 {
                return Err("composition Jobs input byte bound".into());
            }
            Ok(value.as_bytes().to_vec())
        }
        _ => Err("composition Jobs closed input transport".into()),
    }
}
pub(in crate::automation::hf_certify) fn run(args: &[String]) -> DynResult<()> {
    let [flag, input, out_flag, output] = args else {
        return Err("composition-job-worker --input FILE|--input-environment MESH_HF_JOB_INPUT|--mounted-input-environment MESH_HF_JOB_REQUEST --output-directory FRESH".into());
    };
    if out_flag != "--output-directory" {
        return Err("composition Jobs closed flags".into());
    }
    let bytes = bytes(flag, input)?;
    let input: Input =
        serde_json::from_slice(&bytes).map_err(|_| "composition Jobs typed input refused")?;
    input.validate()?;
    let root = root(output, &input)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut evidence = json!({"schema_version":1,"workflow":"default-mtp-composition","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"transport_input_sha256":admission::digest(&bytes),"status":"FAILED","operator_request_sha256":null,"operator":null,"error":null,"image_observed":false,"cost_observed":false});
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
        Err(_) => Err("composition Jobs explicit credential admission failed".into()),
    };
    let native_complete = result.is_ok() && guard(native_until, &cancel).is_ok();
    if native_complete {
        evidence["status"] = json!("COMPOSITION_COMPLETED");
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
        Err("composition Jobs native receipt publication failed".into())
    };
    let cleaned = credential.map_or(Ok(()), tempfile::NamedTempFile::close);
    let finished = interrupt.finish();
    let candidate = native_complete && exported.is_ok() && cleaned.is_ok() && finished.is_ok();
    let locator = exported.ok().flatten();
    let mut delivery = json!({"schema_version":1,"workflow":"default-mtp-composition","status":"FAILED","request_sha256":evidence["request_sha256"],"native_completed":native_complete,"export":export_evidence,"locator":locator,"image_observed":false,"cost_observed":false});
    let mut receipt_writer = OwnedDeliveryReceipt::new(root.join("native-job-delivery.json"));
    let complete = terminal_publication(
        &mut delivery,
        candidate,
        until,
        &cancel,
        |value| receipt_writer.publish(value),
        |value| {
            if !value["locator"].is_null() {
                println!(
                    "MESH_NATIVE_DELIVERY {}",
                    serde_json::to_string(&value["locator"])?
                );
            }
            Ok(())
        },
    )?;
    if complete {
        Ok(())
    } else {
        Err("composition Jobs/export incomplete; inspect partial receipts".into())
    }
}
/// Guard both durable publication and locator emission; late refusal retains all observations.
fn terminal_publication(
    delivery: &mut Value,
    candidate: bool,
    until: Instant,
    cancel: &Cancellation,
    mut publish: impl FnMut(&Value) -> DynResult<()>,
    mut emit: impl FnMut(&Value) -> DynResult<()>,
) -> DynResult<bool> {
    let mut admitted = candidate && guard(until, cancel).is_ok();
    mark_delivery(delivery, admitted);
    publish(delivery)?;
    if admitted && guard(until, cancel).is_err() {
        admitted = false;
        mark_delivery(delivery, false);
        publish(delivery)?;
    }
    emit(delivery)?;
    if admitted && guard(until, cancel).is_err() {
        mark_delivery(delivery, false);
        publish(delivery)?;
        emit(delivery)?;
        admitted = false;
    }
    Ok(admitted)
}
fn mark_delivery(delivery: &mut Value, complete: bool) {
    delivery["status"] = json!(if complete { "DELIVERED" } else { "FAILED" });
    if !delivery["locator"].is_null() {
        delivery["locator"]["delivery_complete"] = json!(complete);
    }
    if !complete {
        delivery["terminal_admitted"] = json!(false);
    }
}
#[cfg(test)]
#[path = "delivery/tests.rs"]
mod tests;

/// Fresh first publication; replacements belong only to the file this owner created.
struct OwnedDeliveryReceipt {
    path: std::path::PathBuf,
    file: Option<std::fs::File>,
}
impl OwnedDeliveryReceipt {
    fn new(path: std::path::PathBuf) -> Self {
        Self { path, file: None }
    }
    fn publish(&mut self, value: &Value) -> DynResult<()> {
        use std::io::Write as _;
        let bytes = serde_json::to_vec_pretty(value)?;
        if bytes.len() > 1048576 {
            return Err("composition delivery receipt exceeds 1MiB".into());
        }
        let mut staged =
            tempfile::NamedTempFile::new_in(self.path.parent().ok_or("delivery receipt parent")?)?;
        staged.write_all(&bytes)?;
        staged.as_file().sync_all()?;
        let file = if let Some(owned) = &self.file {
            self.verify_owned(owned)?;
            staged.persist(&self.path)?
        } else {
            staged.persist_noclobber(&self.path)?
        };
        self.file = Some(file);
        Ok(())
    }
    fn verify_owned(&self, owned: &std::fs::File) -> DynResult<()> {
        let current = std::fs::symlink_metadata(&self.path)?;
        if !current.is_file() {
            return Err("composition delivery receipt ownership changed".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let original = owned.metadata()?;
            if original.dev() != current.dev() || original.ino() != current.ino() {
                return Err("composition delivery receipt ownership changed".into());
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = owned;
            Err("composition delivery replacement requires Unix file identity".into())
        }
    }
}
