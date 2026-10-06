//! Complete immutable mounted conversion artifacts into a fresh original upload workspace.
use super::{contract, guard};
use crate::{
    automation::hf_certify::{admission, execution, publication},
    command::DynResult,
    process::Cancellation,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FilePin {
    pub name: String,
    pub sha256: String,
    pub byte_size: u64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub schema_version: u32,
    pub repo: String,
    pub revision: String,
    pub source_directory: PathBuf,
    pub work_directory: PathBuf,
    pub target_prefix: String,
    pub output_basename: String,
    pub requested_splits: usize,
    pub files: Vec<FilePin>,
    pub timeout_seconds: u64,
}
impl Request {
    pub(super) fn validate(&self) -> DynResult<()> {
        let mut names = BTreeSet::new();
        let repo: Vec<_> = self.repo.split('/').collect();
        if self.schema_version != 1
            || repo.len() != 2
            || !repo.iter().all(|s| contract::leaf(s))
            || !contract::hex(&self.revision, 40)
            || !self.source_directory.is_absolute()
            || !self.work_directory.is_absolute()
            || !contract::leaf(&self.target_prefix)
            || !contract::leaf(&self.output_basename)
            || !(1..=1024).contains(&self.requested_splits)
            || !(5..=259200).contains(&self.timeout_seconds)
            || self.files.is_empty()
            || self.files.len() > 1024
        {
            return Err("artifact workspace schema/immutable source/roster/budget refused".into());
        }
        for pin in &self.files {
            if !contract::leaf(&pin.name)
                || !contract::hex(&pin.sha256, 64)
                || pin.byte_size == 0
                || pin.byte_size > 4 * 1024_u64.pow(4)
                || !names.insert(&pin.name)
            {
                return Err("artifact workspace file pin/duplicate/size refused".into());
            }
        }
        if !names.contains(&"README.md".to_owned())
            || !names.contains(&"skippy-convert-manifest.json".to_owned())
        {
            return Err("artifact workspace card/manifest required".into());
        }
        Ok(())
    }
}
fn open(path: &Path) -> DynResult<File> {
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("artifact workspace regular source required".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("artifact workspace opened source not regular".into());
    }
    Ok(file)
}
fn roster(request: &Request) -> DynResult<()> {
    if request.source_directory.canonicalize()? != request.source_directory
        || !std::fs::symlink_metadata(&request.source_directory)?.is_dir()
    {
        return Err("artifact workspace canonical source directory required".into());
    }
    let mut actual = BTreeSet::new();
    for entry in std::fs::read_dir(&request.source_directory)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Err("artifact workspace flat regular complete roster required".into());
        }
        actual.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| "artifact workspace filename Unicode")?,
        );
        if actual.len() > 1024 {
            return Err("artifact workspace roster bound".into());
        }
    }
    if actual != request.files.iter().map(|p| p.name.clone()).collect() {
        return Err("artifact workspace incomplete source roster".into());
    }
    Ok(())
}
fn bytes(
    request: &Request,
    pin: &FilePin,
    destination: Option<&Path>,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    guard(until, cancel)?;
    let mut source = open(&request.source_directory.join(&pin.name))?;
    if source.metadata()?.len() != pin.byte_size {
        return Err("artifact workspace source size mismatch".into());
    }
    let mut target = destination
        .map(tempfile::NamedTempFile::new_in)
        .transpose()?;
    let mut hash = Sha256::new();
    let mut seen = 0_u64;
    let mut chunk = vec![0_u8; 1048576];
    loop {
        guard(until, cancel)?;
        let count = source.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        seen = seen
            .checked_add(count as u64)
            .ok_or("artifact workspace byte overflow")?;
        if seen > pin.byte_size {
            return Err("artifact workspace source grew".into());
        }
        hash.update(&chunk[..count]);
        if let Some(file) = target.as_mut() {
            file.write_all(&chunk[..count])?;
        }
    }
    if seen != pin.byte_size
        || hex::encode(hash.finalize()) != pin.sha256
        || source.metadata()?.len() != pin.byte_size
    {
        return Err("artifact workspace pinned bytes mismatch".into());
    }
    guard(until, cancel)?;
    if let Some(file) = target {
        file.as_file().sync_all()?;
        file.persist_noclobber(
            destination
                .ok_or("artifact workspace destination")?
                .join(&pin.name),
        )?;
    }
    Ok(())
}
fn manifest(request: &Request) -> DynResult<Value> {
    let value: Value = serde_json::from_slice(&admission::read(
        &request
            .source_directory
            .join("skippy-convert-manifest.json"),
        8 * 1048576,
    )?)?;
    let count = value["expected_splits"]
        .as_u64()
        .ok_or("artifact workspace effective count")?;
    if count < request.requested_splits as u64
        || count > 1024
        || value["output_basename"] != request.output_basename
        || value["target_prefix"] != request.target_prefix
    {
        return Err("artifact workspace effective roster/basename refused".into());
    }
    crate::automation::hf_converted_artifact::validate_converted_artifact_manifest(
        &request.source_directory,
        &value,
    )?;
    Ok(value)
}
fn execute(request: &Request, until: Instant, cancel: &Cancellation) -> DynResult<Value> {
    request.validate()?;
    guard(until, cancel)?;
    roster(request)?;
    let work_parent = request
        .work_directory
        .parent()
        .ok_or("workspace parent")?
        .canonicalize()?;
    let work = work_parent.join(request.work_directory.file_name().ok_or("workspace leaf")?);
    if work != request.work_directory
        || work.starts_with(&request.source_directory)
        || request.source_directory.starts_with(&work)
    {
        return Err("artifact workspace source/destination ancestry refused".into());
    }
    match std::fs::symlink_metadata(&work) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("artifact workspace must be fresh".into()),
    }
    for pin in &request.files {
        bytes(request, pin, None, until, cancel)?;
    }
    let manifest = manifest(request)?;
    guard(until, cancel)?;
    std::fs::create_dir(&work)?;
    std::fs::create_dir(work.join("target"))?;
    let destination = work.join("target").join(&request.target_prefix);
    std::fs::create_dir(&destination)?;
    for pin in &request.files {
        bytes(request, pin, Some(&destination), until, cancel)?;
    }
    roster(request)?;
    for pin in &request.files {
        bytes(request, pin, None, until, cancel)?;
    }
    crate::automation::hf_converted_artifact::validate_converted_artifact_manifest(
        &destination,
        &manifest,
    )?;
    guard(until, cancel)?;
    Ok(
        json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(request)?),"completed":true,"source_repo":request.repo,"source_revision":request.revision,"effective_splits":manifest["expected_splits"],"files":request.files,"artifact_directory":destination,"built":false,"converted":false}),
    )
}
pub(super) fn worker(args: &[String]) -> DynResult<()> {
    let [a, input, b, output] = args else {
        return Err("artifact-workspace-worker --input FILE --output FILE".into());
    };
    if a != "--input" || b != "--output" {
        return Err("artifact workspace closed flags".into());
    }
    let request: Request =
        serde_json::from_slice(&admission::read(Path::new(input), 8 * 1048576)?)?;
    request.validate()?;
    let output_path = std::path::absolute(output)?;
    let output_path = output_path
        .parent()
        .ok_or("artifact workspace receipt parent")?
        .canonicalize()?
        .join(
            output_path
                .file_name()
                .ok_or("artifact workspace receipt leaf")?,
        );
    let source = request.source_directory.canonicalize()?;
    let work = request
        .work_directory
        .parent()
        .ok_or("artifact workspace work parent")?
        .canonicalize()?
        .join(
            request
                .work_directory
                .file_name()
                .ok_or("artifact workspace work leaf")?,
        );
    if output_path.starts_with(&source)
        || source.starts_with(&output_path)
        || output_path.starts_with(&work)
        || work.starts_with(&output_path)
    {
        return Err("artifact workspace evidence/source/work ancestry refused".into());
    }
    match std::fs::symlink_metadata(&output_path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("artifact workspace receipt must be fresh".into()),
    }
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + Duration::from_secs(request.timeout_seconds);
    let mut receipt = json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&request)?),"completed":false,"error":null});
    let result = execute(&request, until, &cancel);
    if let Ok(value) = &result {
        receipt = value.clone();
    }
    let finish = interrupt.finish();
    let terminal = guard(until, &cancel);
    if result.is_err() || finish.is_err() || terminal.is_err() {
        receipt["completed"] = json!(false);
        receipt["error"] = json!("artifact workspace incomplete; partial copied files retained");
    }
    admission::publish(&output_path, &receipt)?;
    if receipt["completed"] == true {
        Ok(())
    } else {
        Err("artifact workspace incomplete".into())
    }
}
pub(super) fn materialize(
    request: &Request,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    let mut request = request.clone();
    request.timeout_seconds = until
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .map(|d| d.as_secs())
        .filter(|s| *s >= 5)
        .ok_or("artifact workspace remaining allowance")?;
    let input = root.join("artifact-workspace-input.json");
    let output = root.join("artifact-workspace.json");
    admission::publish(&input, &request)?;
    let report = execution::run_process(
        &std::env::current_exe()?.canonicalize()?,
        vec![
            "automation".into(),
            "hf-certify".into(),
            "artifact-workspace-worker".into(),
            "--input".into(),
            input.to_str().ok_or("workspace input Unicode")?.into(),
            "--output".into(),
            output.to_str().ok_or("workspace output Unicode")?.into(),
        ],
        root,
        "artifact-workspace",
        until,
        cancel,
    )?;
    evidence["artifact_workspace_process"] = publication::process_observation(&report);
    let receipt: Value = serde_json::from_slice(&admission::read(&output, 1048576)?)?;
    evidence["artifact_workspace"] = receipt.clone();
    if !execution::clean(&report)
        || receipt["completed"] != true
        || receipt["request_sha256"] != admission::digest(&serde_json::to_vec(&request)?)
    {
        return Err("artifact workspace owned process/receipt refused".into());
    }
    guard(until, cancel)
}
#[cfg(test)]
#[path = "artifact_workspace/tests.rs"]
mod tests;
