//! File I/O owner for a separately supervised preflight worker.
//! Parent supervision, not in-process polling, bounds blocking filesystem I/O.
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub binaries: [PathBuf; 2],
    pub model: PathBuf,
    pub minimum_context_tokens: u64,
}
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct FileIdentity {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct BinaryIdentity {
    pub file: FileIdentity,
    pub adjacent_runtime_root: PathBuf,
    pub runtime_validation: String,
}
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelMetadata {
    pub sha256: String,
    pub architecture: String,
    pub native_context_tokens: u64,
}
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct Evidence {
    pub schema_version: u64,
    pub binaries: [BinaryIdentity; 2],
    pub model: FileIdentity,
    pub model_metadata: ModelMetadata,
}
fn live(cancellation: &Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        Err("benchmark preflight interrupted".into())
    } else {
        Ok(())
    }
}
pub(super) fn regular(path: &Path) -> DynResult<PathBuf> {
    if !path.is_absolute() || path.as_os_str().len() > 16 * 1024 {
        return Err("preflight requires bounded absolute approved local paths".into());
    }
    let canonical = fs::canonicalize(path)?;
    let metadata = fs::metadata(&canonical)?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err("preflight requires a nonempty local regular file".into());
    }
    Ok(canonical)
}
fn fingerprint(path: &Path, cancellation: &Cancellation) -> DynResult<FileIdentity> {
    live(cancellation)?;
    let path = regular(path)?;
    let before = fs::metadata(&path)?;
    let sha256 = crate::product::digest::file_sha256(&path).map_err(|error| error.error)?;
    live(cancellation)?;
    let after = fs::metadata(&path)?;
    if before.len() != after.len() || before.modified()? != after.modified()? {
        return Err("preflight source changed while hashing".into());
    }
    Ok(FileIdentity {
        path,
        sha256,
        bytes: after.len(),
    })
}
pub(super) fn adjacent_runtime_root(binary: &Path) -> DynResult<PathBuf> {
    let root = binary
        .parent()
        .ok_or("binary has no parent")?
        .join("native-runtimes");
    let metadata = fs::symlink_metadata(&root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() || fs::canonicalize(&root)? != root {
        return Err(
            "native runtime root must be a local directory directly beside the admitted binary"
                .into(),
        );
    }
    Ok(root)
}
fn binary(path: &Path, cancellation: &Cancellation) -> DynResult<BinaryIdentity> {
    let file = fingerprint(path, cancellation)?;
    let adjacent_runtime_root = adjacent_runtime_root(&file.path)?;
    Ok(BinaryIdentity {
        file,
        adjacent_runtime_root,
        runtime_validation: "pending_owning_native_package_and_loader_policy".into(),
    })
}
/// Only invoke inside the parent's bounded owned worker, never its coordinator callback.
pub(super) fn execute(input: &Input, cancellation: &Cancellation) -> DynResult<Evidence> {
    if input.schema_version != 1 || input.minimum_context_tokens == 0 {
        return Err("invalid benchmark preflight schema or context requirement".into());
    }
    let binaries = [
        binary(&input.binaries[0], cancellation)?,
        binary(&input.binaries[1], cancellation)?,
    ];
    let model = fingerprint(&input.model, cancellation)?;
    live(cancellation)?;
    let verified = crate::automation::replay_matrix::model_preflight::verify(
        &model.path,
        &model.sha256,
        input.minimum_context_tokens,
    )?;
    live(cancellation)?;
    let model_metadata: ModelMetadata = serde_json::from_value(serde_json::to_value(verified)?)?;
    let final_hash =
        crate::product::digest::file_sha256(&model.path).map_err(|error| error.error)?;
    if final_hash != model.sha256 {
        return Err("GGUF source changed during metadata admission".into());
    }
    live(cancellation)?;
    Ok(Evidence {
        schema_version: 1,
        binaries,
        model,
        model_metadata,
    })
}
#[cfg(test)]
#[path = "identity_worker_tests.rs"]
mod tests;

/// Publish into a parent-owned fresh receipt path. Diagnostics may remain on stderr.
#[cfg(test)]
pub(super) fn write_receipt(
    input: &Input,
    output: &Path,
    cancellation: &Cancellation,
) -> DynResult<()> {
    use std::io::Write;
    let parent = output
        .parent()
        .ok_or("preflight receipt requires a parent")?;
    if !output.is_absolute() || fs::canonicalize(parent)? != parent || output.exists() {
        return Err("preflight receipt requires a fresh absolute parent-owned path".into());
    }
    let evidence = execute(input, cancellation)?;
    let mut bytes = serde_json::to_vec(&evidence)?;
    if bytes.len() > 512 * 1024 {
        return Err("preflight receipt exceeds 512KiB".into());
    }
    bytes.push(b'\n');
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&bytes)?;
    temporary.as_file().sync_all()?;
    live(cancellation)?;
    temporary.persist_noclobber(output)?;
    Ok(())
}
