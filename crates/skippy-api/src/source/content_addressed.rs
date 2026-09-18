use std::{ffi::OsStr, path::Path};

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{SkippyPackageIdentity, SkippyPackageSourceFile, hex_lower, synthetic_gguf_package};

struct ManagedHfSnapshotBlob {
    blob_root: std::path::PathBuf,
    target: std::path::PathBuf,
}

pub(super) struct HuggingFaceSourceFiles {
    pub files: Vec<SkippyPackageSourceFile>,
    pub identity_sha256: String,
}

#[derive(Serialize)]
struct HuggingFaceGgufIdentity<'a> {
    schema_version: u32,
    source: &'static str,
    repository: &'a str,
    revision: String,
    files: &'a [HuggingFaceGgufIdentityFile],
}

#[derive(Clone, Serialize)]
struct HuggingFaceGgufIdentityFile {
    path: String,
    bytes: u64,
    sha256: String,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_huggingface_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Resolve the authoritative content digests already recorded by the
/// Hugging Face cache for a GGUF distribution.
///
/// Large Hub artifacts are stored under `blobs/<sha256>` and snapshot entries
/// point at those immutable blobs. Their names and sizes are authoritative
/// identity inputs, so startup does not need to scan the weight payload.
pub(super) fn huggingface_source_files(
    identity: &model_hf::store::local::HuggingFaceModelIdentity,
    snapshot_root: &Path,
    mut build_client: impl FnMut() -> Result<HFClientSync>,
) -> Result<HuggingFaceSourceFiles> {
    anyhow::ensure!(
        is_huggingface_commit(&identity.revision),
        "Hugging Face GGUF identity requires an immutable 40-character commit SHA, got {}",
        identity.revision
    );

    let snapshot_path = snapshot_root.join(&identity.file);
    anyhow::ensure!(
        snapshot_path.exists(),
        "Hugging Face GGUF snapshot entry is unavailable: {}",
        snapshot_path.display()
    );

    let snapshot_paths = snapshot_source_paths(&snapshot_path)?;
    let repo_dir = snapshot_root
        .parent()
        .and_then(Path::parent)
        .context("Hugging Face snapshot has no repository root")?;
    let blob_root = repo_dir
        .join("blobs")
        .canonicalize()
        .with_context(|| format!("canonicalize Hugging Face blob root {}", repo_dir.display()))?;
    let local_paths = match managed_hf_multipart_view_in_root(&snapshot_paths, &blob_root)? {
        Some(paths) => paths,
        None => snapshot_paths
            .iter()
            .map(|path| {
                path.canonicalize().with_context(|| {
                    format!("canonicalize Hugging Face GGUF source {}", path.display())
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };

    let mut api = None;
    let resolved = snapshot_paths
        .into_iter()
        .zip(local_paths)
        .map(|(snapshot_path, local_path)| {
            let relative_file = snapshot_path
                .strip_prefix(snapshot_root)
                .with_context(|| {
                    format!(
                        "derive Hugging Face repository path for {}",
                        snapshot_path.display()
                    )
                })?
                .to_string_lossy()
                .replace('\\', "/");
            let local_metadata = local_path
                .metadata()
                .with_context(|| format!("stat local GGUF source {}", local_path.display()))?;
            anyhow::ensure!(
                local_metadata.is_file(),
                "Hugging Face GGUF source is not a file: {}",
                local_path.display()
            );

            let (sha256, bytes) =
                match managed_hf_snapshot_blob_in_root(&snapshot_path, &blob_root)? {
                    Some(blob) => {
                        let sha256 = blob
                            .target
                            .file_name()
                            .and_then(|name| name.to_str())
                            .filter(|value| is_sha256(value))
                            .with_context(|| {
                                format!(
                                    "Hugging Face GGUF blob has no authoritative SHA-256 name: {}",
                                    blob.target.display()
                                )
                            })?
                            .to_string();
                        let metadata = blob.target.metadata().with_context(|| {
                            format!("stat Hugging Face GGUF blob {}", blob.target.display())
                        })?;
                        (sha256, metadata.len())
                    }
                    None => {
                        hub_file_identity(&mut api, identity, &relative_file, &mut build_client)?
                    }
                };
            anyhow::ensure!(
                local_metadata.len() == bytes,
                "Hugging Face GGUF metadata and local source size differ: {}",
                snapshot_path.display()
            );
            Ok((
                SkippyPackageSourceFile {
                    path: local_path,
                    bytes,
                    sha256: sha256.clone(),
                },
                HuggingFaceGgufIdentityFile {
                    path: relative_file,
                    bytes,
                    sha256,
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let (files, identity_files): (Vec<_>, Vec<_>) = resolved.into_iter().unzip();
    let identity_sha256 = huggingface_identity_sha256(identity, &identity_files)?;
    Ok(HuggingFaceSourceFiles {
        files,
        identity_sha256,
    })
}

fn huggingface_identity_sha256(
    identity: &model_hf::store::local::HuggingFaceModelIdentity,
    files: &[HuggingFaceGgufIdentityFile],
) -> Result<String> {
    let canonical = HuggingFaceGgufIdentity {
        schema_version: 1,
        source: "hugging-face-gguf",
        repository: &identity.repo_id,
        revision: identity.revision.to_ascii_lowercase(),
        files,
    };
    let bytes = serde_json::to_vec(&canonical).context("serialize Hugging Face GGUF identity")?;
    Ok(hex_lower(&Sha256::digest(bytes)))
}

fn hub_file_identity(
    api: &mut Option<HFClientSync>,
    identity: &model_hf::store::local::HuggingFaceModelIdentity,
    file: &str,
    build_client: &mut impl FnMut() -> Result<HFClientSync>,
) -> Result<(String, u64)> {
    let api = match api {
        Some(api) => api,
        None => api.insert(build_client()?),
    };
    let (owner, name) = identity
        .repo_id
        .split_once('/')
        .unwrap_or(("", identity.repo_id.as_str()));
    let metadata = api
        .model(owner, name)
        .get_file_metadata()
        .filepath(file.to_string())
        .revision(identity.revision.as_str())
        .send()
        .with_context(|| {
            format!(
                "fetch Hugging Face file metadata for {}@{}/{}",
                identity.repo_id, identity.revision, file
            )
        })?;
    anyhow::ensure!(
        metadata.commit_hash == identity.revision,
        "Hugging Face metadata resolved to commit {}, expected {}",
        metadata.commit_hash,
        identity.revision
    );
    let sha256 = is_sha256(&metadata.etag)
        .then_some(metadata.etag)
        .with_context(|| {
            format!(
                "Hugging Face ETag has no authoritative payload SHA-256 for {}@{}/{}",
                identity.repo_id, identity.revision, file
            )
        })?;
    Ok((sha256, metadata.file_size))
}

fn snapshot_source_paths(model_path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let Some(file_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        anyhow::bail!(
            "Hugging Face GGUF snapshot path has no UTF-8 filename: {}",
            model_path.display()
        );
    };
    let Some(shard) = model_ref::split_gguf_shard_info(file_name) else {
        return Ok(vec![model_path.to_path_buf()]);
    };
    anyhow::ensure!(
        shard.part == "00001",
        "split GGUF inputs must point at the first shard, got {}",
        model_path.display()
    );
    let total = shard
        .total
        .parse::<u32>()
        .with_context(|| format!("parse split GGUF shard total in {file_name}"))?;
    let parent = model_path
        .parent()
        .with_context(|| format!("split GGUF shard has no parent: {}", model_path.display()))?;
    (1..=total)
        .map(|index| {
            let path = parent.join(format!("{}-{index:05}-of-{:05}.gguf", shard.prefix, total));
            path.metadata().with_context(|| {
                format!(
                    "read split GGUF shard {index}/{total} for {}",
                    model_path.display()
                )
            })?;
            Ok(path)
        })
        .collect()
}

fn managed_hf_snapshot_blob(path: &Path) -> Result<Option<ManagedHfSnapshotBlob>> {
    let Some(repo_dir) = path.ancestors().find_map(|revision_dir| {
        let snapshots_dir = revision_dir.parent()?;
        (snapshots_dir.file_name() == Some(OsStr::new("snapshots")))
            .then(|| snapshots_dir.parent())
            .flatten()
    }) else {
        return Ok(None);
    };
    let Some(cache_root) = repo_dir.parent() else {
        return Ok(None);
    };
    if model_hf::huggingface_identity_for_path_in_cache(path, cache_root).is_none() {
        return Ok(None);
    }
    let blob_root = repo_dir
        .join("blobs")
        .canonicalize()
        .with_context(|| format!("canonicalize Hugging Face blob root {}", repo_dir.display()))?;
    managed_hf_snapshot_blob_in_root(path, &blob_root)
}

fn managed_hf_snapshot_blob_in_root(
    path: &Path,
    blob_root: &Path,
) -> Result<Option<ManagedHfSnapshotBlob>> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stat Hugging Face GGUF snapshot entry {}", path.display()))?;
    if !metadata.file_type().is_symlink() {
        return Ok(None);
    }
    let target = path
        .canonicalize()
        .with_context(|| format!("resolve Hugging Face GGUF snapshot link {}", path.display()))?;
    anyhow::ensure!(
        target.starts_with(blob_root),
        "Hugging Face GGUF snapshot link escapes its blob store: {}",
        path.display()
    );
    Ok(Some(ManagedHfSnapshotBlob {
        blob_root: blob_root.to_path_buf(),
        target,
    }))
}

/// Give native multipart GGUF discovery stable upstream filenames without
/// loading through mutable snapshot symlinks. The view consists only of hard
/// links to the already validated Hugging Face blobs, so it consumes no model
/// payload space and remains compatible with strict regular-file attestation.
pub(super) fn managed_hf_multipart_view(
    snapshot_paths: &[std::path::PathBuf],
) -> Result<Option<Vec<std::path::PathBuf>>> {
    if snapshot_paths.len() <= 1 {
        return Ok(None);
    }

    let Some(source) = managed_hf_snapshot_blob(&snapshot_paths[0])? else {
        return Ok(None);
    };
    managed_hf_multipart_view_in_root(snapshot_paths, &source.blob_root)
}

fn managed_hf_multipart_view_in_root(
    snapshot_paths: &[std::path::PathBuf],
    blob_root: &Path,
) -> Result<Option<Vec<std::path::PathBuf>>> {
    if snapshot_paths.len() <= 1 {
        return Ok(None);
    }
    let mut sources = Vec::with_capacity(snapshot_paths.len());
    for snapshot_path in snapshot_paths {
        let Some(source) = managed_hf_snapshot_blob_in_root(snapshot_path, blob_root)? else {
            return Ok(None);
        };
        sources.push(source);
    }

    let mut view_id = Sha256::new();
    view_id.update(b"mesh-llm-hf-multipart-view-v1\0");
    for (snapshot_path, source) in snapshot_paths.iter().zip(&sources) {
        let name = snapshot_path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| {
                format!(
                    "Hugging Face GGUF snapshot path has no UTF-8 filename: {}",
                    snapshot_path.display()
                )
            })?;
        view_id.update((name.len() as u64).to_le_bytes());
        view_id.update(name.as_bytes());
        let target = source.target.to_string_lossy();
        view_id.update((target.len() as u64).to_le_bytes());
        view_id.update(target.as_bytes());
    }
    let repo_root = blob_root.parent().with_context(|| {
        format!(
            "Hugging Face blob root has no parent: {}",
            blob_root.display()
        )
    })?;
    let view_dir = repo_root
        .join(".skippy")
        .join("multipart-gguf")
        .join(hex_lower(&view_id.finalize()));
    std::fs::create_dir_all(&view_dir)
        .with_context(|| format!("create multipart GGUF view {}", view_dir.display()))?;

    let mut view_paths = Vec::with_capacity(sources.len());
    for (snapshot_path, source) in snapshot_paths.iter().zip(&sources) {
        let destination = view_dir.join(
            snapshot_path
                .file_name()
                .context("multipart GGUF snapshot path has no filename")?,
        );
        match std::fs::hard_link(&source.target, &destination) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "create multipart GGUF view {} -> {}",
                        destination.display(),
                        source.target.display()
                    )
                });
            }
        }
        ensure_same_regular_file(&source.target, &destination)?;
        view_paths.push(destination);
    }
    Ok(Some(view_paths))
}

fn ensure_same_regular_file(source: &Path, destination: &Path) -> Result<()> {
    let source_metadata = std::fs::symlink_metadata(source)
        .with_context(|| format!("stat Hugging Face GGUF blob {}", source.display()))?;
    let destination_metadata = std::fs::symlink_metadata(destination)
        .with_context(|| format!("stat multipart GGUF view {}", destination.display()))?;
    anyhow::ensure!(
        source_metadata.is_file()
            && !source_metadata.file_type().is_symlink()
            && destination_metadata.is_file()
            && !destination_metadata.file_type().is_symlink(),
        "multipart GGUF view must contain regular files"
    );
    anyhow::ensure!(
        same_file::is_same_file(source, destination).with_context(|| {
            format!(
                "compare multipart GGUF view {} with Hugging Face blob {}",
                destination.display(),
                source.display()
            )
        })?,
        "multipart GGUF view does not reference its verified Hugging Face blob: {}",
        destination.display()
    );
    Ok(())
}

#[derive(Serialize)]
struct ContentAddressedGgufManifest<'a> {
    schema_version: u32,
    package_kind: &'a str,
    source_model_sha256: &'a str,
    source_model_bytes: u64,
    source_files: &'a [ContentAddressedGgufManifestFile],
    architecture: &'a str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
}

#[derive(Serialize)]
struct ContentAddressedGgufManifestFile {
    ordinal: u32,
    bytes: u64,
    sha256: String,
}

/// Build a path-independent identity for a GGUF that must already exist on
/// every split-serving participant.
///
/// The returned package and manifest identities contain no model alias,
/// filename, or absolute path. The path remains node-local and is registered
/// only for resolving a later inventory or load request on this process.
pub fn synthetic_content_addressed_gguf_package(
    model_id: &str,
    model_path: &Path,
) -> Result<SkippyPackageIdentity> {
    synthetic_gguf_package(model_id, model_path)
}

pub(super) fn validate_source_set(model_path: &Path) -> Result<()> {
    anyhow::ensure!(
        model_path.is_absolute(),
        "content-addressed GGUF source path must be absolute: {}",
        model_path.display()
    );
    anyhow::ensure!(
        model_path.to_str().is_some(),
        "content-addressed GGUF source path must be valid UTF-8: {}",
        model_path.display()
    );
    let validate_file = |path: &Path| -> Result<()> {
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("stat content-addressed GGUF source {}", path.display()))?;
        let managed_hf_snapshot_blob = if metadata.file_type().is_symlink() {
            managed_hf_snapshot_blob(path)?
        } else {
            None
        };
        anyhow::ensure!(
            metadata.is_file() || managed_hf_snapshot_blob.is_some(),
            "content-addressed GGUF source must be a non-symlink file: {}",
            path.display()
        );
        if let Some(source) = managed_hf_snapshot_blob {
            let target_metadata = std::fs::symlink_metadata(&source.target).with_context(|| {
                format!("stat Hugging Face GGUF blob {}", source.target.display())
            })?;
            anyhow::ensure!(
                target_metadata.is_file() && !target_metadata.file_type().is_symlink(),
                "Hugging Face GGUF snapshot link must resolve to a regular file: {}",
                path.display()
            );
        }
        anyhow::ensure!(
            path.to_str().is_some(),
            "content-addressed GGUF source path must be valid UTF-8: {}",
            path.display()
        );
        Ok(())
    };
    validate_file(model_path)?;
    let Some(file_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        anyhow::bail!(
            "content-addressed GGUF source has no UTF-8 filename: {}",
            model_path.display()
        );
    };
    let Some(shard) = model_ref::split_gguf_shard_info(file_name) else {
        return Ok(());
    };
    anyhow::ensure!(
        shard.part == "00001",
        "split GGUF inputs must point at the first shard, got {}",
        model_path.display()
    );
    let total = shard
        .total
        .parse::<u32>()
        .with_context(|| format!("parse split GGUF shard total in {file_name}"))?;
    let parent = model_path
        .parent()
        .with_context(|| format!("split GGUF shard has no parent: {}", model_path.display()))?;
    for index in 1..=total {
        validate_file(&parent.join(format!("{}-{index:05}-of-{:05}.gguf", shard.prefix, total)))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn manifest_sha256(
    source_model_sha256: &str,
    source_model_bytes: u64,
    source_files: &[SkippyPackageSourceFile],
    architecture: &str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
) -> Result<String> {
    let files = source_files
        .iter()
        .enumerate()
        .map(|(index, file)| ContentAddressedGgufManifestFile {
            ordinal: u32::try_from(index).unwrap_or(u32::MAX),
            bytes: file.bytes,
            sha256: file.sha256.clone(),
        })
        .collect::<Vec<_>>();
    let manifest = ContentAddressedGgufManifest {
        schema_version: 2,
        package_kind: "content-addressed-direct-gguf",
        source_model_sha256,
        source_model_bytes,
        source_files: &files,
        architecture,
        context_length,
        layer_count,
        activation_width,
        tensor_count,
    };
    let bytes =
        serde_json::to_vec(&manifest).context("serialize content-addressed GGUF manifest")?;
    Ok(hex_lower(&Sha256::digest(bytes)))
}

pub(super) fn ensure_fingerprint_unchanged(
    source_files: &[SkippyPackageSourceFile],
    expected: Option<&[crate::source_registry::VerifiedFileFingerprint]>,
) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let paths = source_files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        crate::source_registry::verified_path_fingerprint(&paths).as_deref() == Some(expected),
        "content-addressed GGUF source changed while its identity was being computed"
    );
    Ok(())
}

pub(super) fn aggregate_source_sha256(source_files: &[SkippyPackageSourceFile]) -> String {
    if source_files.len() == 1 {
        return source_files[0].sha256.clone();
    }
    let mut hasher = Sha256::new();
    hasher.update(b"mesh-llm-split-gguf-v1\0");
    hasher.update((source_files.len() as u64).to_le_bytes());
    for (index, file) in source_files.iter().enumerate() {
        hasher.update((index as u64).to_le_bytes());
        hasher.update(file.bytes.to_le_bytes());
        hasher.update(file.sha256.as_bytes());
    }
    hex_lower(&hasher.finalize())
}

#[cfg(test)]
mod tests;
