//! The claimed GGUF an audit checks against: a local file, or a Hugging
//! Face file named `repo@revision/file` that the audit reads by range into
//! a sparse local copy, so the client never downloads the model.
//!
//! Who named the model decides how it opens:
//!
//! - The user ([`open`]): a file here, or a Hugging Face file at any
//!   revision, a branch included.
//! - A node, in a receipt ([`open_named_by_receipt`], [`open_remote`]):
//!   a Hugging Face file only, never a file here, and only at a commit. A
//!   branch can move, so a node could fail an audit and then publish other
//!   bytes under the same name, and nobody could check the failure again.
//!
//! The name is hostile text when a node wrote it ([`hf_file`]), and the
//! copy's path stays under the cache root whatever it says (`cache_path`).
//! The copy is pinned by the commit Hugging Face resolves the revision to,
//! which [`ClaimedFile::pinned`] returns. The token goes to the resolver
//! only (`hf_remote`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use skippy_receipts::shards::shard_files;
use skippy_receipts::sparse_copy::{RemoteIdentity, SparseCopy};
use skippy_receipts::weights::limits::CannotAudit;
use skippy_receipts::weights::{ByteSource, LocalFile, WeightFile};

mod cache_path;
pub mod hf_file;
mod hf_remote;

pub use hf_file::HfFile;
use hf_remote::HfRemote;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// The claimed GGUF, opened.
pub struct ClaimedFile {
    /// What the native loader reads: the file, or its sparse copy.
    pub path: PathBuf,
    pub weights: WeightFile,
    /// What names this file and no other: `repo@commit/file`, or a local
    /// file's path, size and modification time.
    pub identity: String,
    /// The Hugging Face file at the commit its revision resolved to, to
    /// keep with an audit so a recheck reads the same bytes. `None` for a
    /// local file.
    pub pinned: Option<HfFile>,
}

/// Where the sparse copies live, unless the caller names a place.
pub fn default_cache_root() -> PathBuf {
    skippy_model_hf::application_cache_dir()
        .join("receipts")
        .join("claimed")
}

/// Opens `claimed`, a model the user named: a path when it names a file
/// here, else a Hugging Face file, whose sparse copy lives under
/// `cache_root`. Never for a name out of a receipt:
/// [`open_named_by_receipt`] is for that.
pub fn open(claimed: &Path, cache_root: &Path) -> Result<ClaimedFile> {
    if claimed.exists() {
        let weights = open_local(claimed)?;
        return Ok(ClaimedFile {
            path: claimed.to_owned(),
            weights,
            identity: local_identity(claimed)?,
            pinned: None,
        });
    }
    let text = claimed.to_string_lossy();
    let file = HfFile::parse_checked(&text).with_context(|| {
        format!(
            "the claimed model {text} is neither a file here nor repo@revision/file on Hugging Face"
        )
    })?;
    open_hf(
        &file,
        &endpoint(),
        skippy_model_hf::hf_token_override(),
        cache_root,
    )
}

/// Opens the model a receipt names. The node wrote the name, so it is only
/// ever a Hugging Face file at a commit ([`open_remote`]): a name that is a
/// path here opens nothing.
pub fn open_named_by_receipt(model: &str, cache_root: &Path) -> Result<ClaimedFile> {
    let file = HfFile::parse_checked(model).with_context(|| {
        format!("the receipt names {model:?}, not repo@commit/file on Hugging Face")
    })?;
    open_remote(&file, cache_root)
}

/// Opens `file` on Hugging Face (or `HF_ENDPOINT`), never a file here: for
/// a claimed model a node named, which must not choose what this machine
/// reads. The revision must be a commit, which names the same bytes for
/// good.
pub fn open_remote(file: &HfFile, cache_root: &Path) -> Result<ClaimedFile> {
    require_pinned(file)?;
    open_hf(
        file,
        &endpoint(),
        skippy_model_hf::hf_token_override(),
        cache_root,
    )
}

/// Refuses a name a node wrote whose revision can move.
fn require_pinned(file: &HfFile) -> Result<()> {
    file.check()
        .with_context(|| format!("{file} names no file on Hugging Face"))?;
    if !file.is_pinned() {
        return Err(CannotAudit::new(format!(
            "the receipt names {file}, and the revision {:?} is not a commit: it can move to other bytes, so the audit could not be checked again. A receipt must name the 40-digit commit",
            file.revision
        ))
        .into());
    }
    Ok(())
}

/// The Hugging Face endpoint: `HF_ENDPOINT`, when set and not empty.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim().trim_end_matches('/').to_owned())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned())
}

/// Opens the sparse copy of `file` and reads its header; for the first
/// shard of a split GGUF, the copies of every shard, at the commit the
/// first resolves to.
pub fn open_hf(
    file: &HfFile,
    endpoint: &str,
    token: Option<String>,
    cache_root: &Path,
) -> Result<ClaimedFile> {
    // the shards first: a name of too many is refused before any request
    let shard_names = split_shards(&file.file)?;
    let (first, pinned) = open_hf_copy(file, endpoint, token.clone(), cache_root)?;
    let path = first.path().to_owned();
    let weights = match shard_names {
        None => WeightFile::open(Box::new(first)),
        Some(names) => {
            let mut shards: Vec<Box<dyn ByteSource>> = vec![Box::new(first)];
            for name in &names[1..] {
                let shard = pinned.sibling(name);
                shards.push(Box::new(
                    open_hf_copy(&shard, endpoint, token.clone(), cache_root)?.0,
                ));
            }
            WeightFile::open_shards(shards)
        }
    }
    .with_context(|| format!("read the GGUF header of {pinned}"))?;
    Ok(ClaimedFile {
        path,
        weights,
        identity: pinned.to_string(),
        pinned: Some(pinned),
    })
}

/// Opens the sparse copy of one file, and the file at the commit the copy
/// is pinned to.
fn open_hf_copy(
    file: &HfFile,
    endpoint: &str,
    token: Option<String>,
    cache_root: &Path,
) -> Result<(SparseCopy, HfFile)> {
    let remote = HfRemote::new(file, endpoint, token)?;
    let resolved = remote.resolve()?;
    let commit = match resolved.commit {
        Some(commit) if file.is_pinned() && commit != file.revision => bail!(
            "Hugging Face resolved {}@{} to commit {commit}",
            file.repo,
            file.revision
        ),
        Some(commit) => commit,
        None if file.is_pinned() => file.revision.clone(),
        None => bail!(
            "Hugging Face did not say which commit {}@{} is at, so its bytes cannot be pinned; name the commit",
            file.repo,
            file.revision
        ),
    };
    let size = match resolved.size {
        Some(size) => size,
        None => remote.probe_size(&resolved.url)?,
    };
    remote.set_url(resolved.url)?;
    let pinned = file.at(&commit);
    let path = cache_path::sparse_copy_path(cache_root, &pinned)?;
    let identity = RemoteIdentity {
        source: pinned.to_string(),
        size,
    };
    let copy = SparseCopy::open(&path, identity, Box::new(remote))?;
    Ok((copy, pinned))
}

/// Reads the header of the GGUF at `path`, a local file or a sparse copy,
/// with every shard of a split one, as strictly as an audit opens it: for
/// a caller about to hand the file to the native loader.
pub fn check_header(path: &Path) -> Result<()> {
    open_local(path).map(|_| ())
}

/// Reads a local GGUF, with every shard of a split one.
fn open_local(path: &Path) -> Result<WeightFile> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let Some(names) = split_shards(name)? else {
        return WeightFile::open(Box::new(LocalFile::open(path)?));
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let shards = names
        .iter()
        .map(|name| Ok(Box::new(LocalFile::open(&dir.join(name))?) as Box<dyn ByteSource>))
        .collect::<Result<Vec<_>>>()?;
    WeightFile::open_shards(shards)
}

/// The shard files of the split GGUF `file` names, or `None` for a whole
/// GGUF. A claim names a split by its first shard.
fn split_shards(file: &str) -> Result<Option<Vec<String>>> {
    let name = file.rsplit('/').next().unwrap_or(file);
    match skippy_model_ref::split_gguf_shard_info(name) {
        None => Ok(None),
        Some(shard) if shard.total == "00001" => Ok(None),
        Some(shard) if shard.part != "00001" => {
            bail!(
                "{file} is shard {} of a split GGUF: claim its first shard",
                shard.part
            )
        }
        Some(_) => shard_files(file),
    }
}

fn local_identity(path: &Path) -> Result<String> {
    let canonical = path.canonicalize()?;
    let meta = std::fs::metadata(&canonical)?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    Ok(format!("{}:{}:{modified}", canonical.display(), meta.len()))
}

#[cfg(test)]
mod tests;
