use super::{process, source, workload};
use crate::{
    automation::canary_receipts::{Digest, RunAttempt},
    command::DynResult,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Context {
    pub(super) controller_root: PathBuf,
    pub(super) controller_revision: String,
    pub(super) selected_source: String,
    pub(super) run_id: String,
    pub(super) run_attempt: String,
}
impl Context {
    pub(super) fn validate(&self) -> DynResult<()> {
        source::revision(&self.controller_revision)?;
        if !self.selected_source.is_empty() {
            source::revision(&self.selected_source)?;
        }
        RunAttempt::try_from(self.run_attempt.clone())?;
        if self.run_id.is_empty() || !self.run_id.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("invalid package workflow run".into());
        }
        if !self.controller_root.is_absolute()
            || process::text(&self.controller_root, &["rev-parse", "HEAD"])?
                != self.controller_revision
        {
            return Err("package controller differs from frozen context".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub(super) context: Context,
    pub(super) root: PathBuf,
    pub(super) closure: PathBuf,
    pub(super) output: PathBuf,
}

#[derive(Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FileIdentity {
    File { sha256: Digest, executable: bool },
    Symlink { target: String },
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: u8,
    context: Context,
    root: PathBuf,
    closure: PathBuf,
    native_head: String,
    producer_sha256: Digest,
    files: BTreeMap<String, FileIdentity>,
}

fn inventory(root: &Path) -> DynResult<BTreeMap<String, FileIdentity>> {
    let canonical_root = root.canonicalize()?;
    let root = canonical_root.as_path();
    let names = process::git(
        root,
        &[
            "ls-files".into(),
            "--cached".into(),
            "--others".into(),
            "--exclude-standard".into(),
            "-z".into(),
        ],
        None,
    )?;
    let names = names
        .split(|byte| *byte == 0)
        .filter(|name| !name.is_empty())
        .collect::<BTreeSet<_>>();
    let mut files = BTreeMap::new();
    for name in names {
        process::check()?;
        let name = std::str::from_utf8(name)?;
        let relative = Path::new(name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err("unsafe source inventory path".into());
        }
        let path = root.join(relative);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let identity = if metadata.file_type().is_symlink() {
            FileIdentity::Symlink {
                target: fs::read_link(&path)?
                    .to_str()
                    .ok_or("non-UTF8 source link")?
                    .to_owned(),
            }
        } else if metadata.is_file() {
            #[cfg(unix)]
            let executable = {
                use std::os::unix::fs::PermissionsExt;
                metadata.permissions().mode() & 0o111 != 0
            };
            #[cfg(not(unix))]
            let executable = false;
            if !path.canonicalize()?.starts_with(root) {
                return Err("source inventory parent escapes selected checkout".into());
            }
            FileIdentity::File {
                sha256: Digest::of_file(&path)?,
                executable,
            }
        } else {
            return Err("unsupported source inventory member".into());
        };
        files.insert(name.to_owned(), identity);
    }
    Ok(files)
}

pub(super) fn write(input: &Input) -> DynResult<Digest> {
    input.context.validate()?;
    if !input.root.is_absolute() || !input.closure.is_absolute() {
        return Err("producer paths must be absolute".into());
    }
    let root = input.root.canonicalize()?;
    let closure = input.closure.canonicalize()?;
    let output = new_output(
        &input.output,
        &[&root, &input.context.controller_root.canonicalize()?],
    )?;
    let native = source::prepared(&root)?;
    let first = workload::verify_producer(
        &root,
        &closure,
        &output.with_extension("before.diff"),
        &native.head,
    )?;
    let files = inventory(&root)?;
    let second = workload::verify_producer(
        &root,
        &closure,
        &output.with_extension("after.diff"),
        &native.head,
    )?;
    if first.digest() != second.digest() {
        return Err("producer changed while sealing controller receipt".into());
    }
    let receipt = Receipt {
        schema: 1,
        context: input.context.clone(),
        root,
        closure,
        native_head: native.head,
        producer_sha256: first.digest().clone(),
        files,
    };
    let bytes = serde_json::to_vec(&receipt)?;
    create_file(&output, &bytes)?;
    Ok(Digest::of_bytes(&bytes))
}

/// The expected digest is a protected caller output, not a package/agent assertion.
pub(super) fn consume(
    path: &Path,
    expected: &Digest,
    context: &Context,
    root: &Path,
    closure: &Path,
    candidate: &str,
) -> DynResult<Vec<u8>> {
    context.validate()?;
    source::revision(candidate)?;
    let bytes = fs::read(path)?;
    if Digest::of_bytes(&bytes) != *expected {
        return Err("controller producer receipt differs from frozen dependency digest".into());
    }
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    if receipt.schema != 1
        || receipt.context != *context
        || receipt.root != root.canonicalize()?
        || receipt.closure != closure.canonicalize()?
    {
        return Err("producer receipt belongs to another controller/run/source/closure".into());
    }
    if receipt.files != inventory(root)? {
        return Err("source files differ from pre-staging producer inventory".into());
    }
    process::text(root, &["diff", "--cached", "--exit-code", candidate, "--"])?;
    process::text(root, &["diff", "--exit-code", candidate, "--"])?;
    if !process::text(root, &["ls-files", "--others", "--exclude-standard"])?.is_empty() {
        return Err("untracked source after candidate snapshot".into());
    }
    let prepared = source::prepared(root)?;
    if prepared.head != receipt.native_head {
        return Err("native prepared source differs from pre-snapshot producer".into());
    }
    workload::seal_verified_receipt(
        closure,
        &receipt.producer_sha256,
        &receipt.native_head,
        candidate,
    )
}

pub(super) fn new_output(path: &Path, roots: &[&Path]) -> DynResult<PathBuf> {
    if !path.is_absolute()
        || fs::symlink_metadata(path).is_ok()
        || path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err("output must be a new normalized absolute path".into());
    }
    let parent = path
        .parent()
        .ok_or("output has no parent")?
        .canonicalize()?;
    let resolved = parent.join(path.file_name().ok_or("output has no name")?);
    if roots.iter().any(|root| resolved.starts_with(root)) {
        return Err("controller receipt/package output must be outside checkouts".into());
    }
    Ok(resolved)
}

pub(super) fn create_file(path: &Path, bytes: &[u8]) -> DynResult<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(all(test, unix))]
#[path = "producer_inventory_tests.rs"]
mod inventory_tests;
