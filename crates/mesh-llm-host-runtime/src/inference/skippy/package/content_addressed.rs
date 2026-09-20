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
    identity: &crate::models::HuggingFaceModelIdentity,
) -> Result<HuggingFaceSourceFiles> {
    anyhow::ensure!(
        is_huggingface_commit(&identity.revision),
        "Hugging Face GGUF identity requires an immutable 40-character commit SHA, got {}",
        identity.revision
    );

    let snapshot_root = model_hf::store::local::huggingface_snapshot_path(
        &identity.repo_id,
        hf_hub::RepoTypeModel,
        &identity.revision,
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
                .strip_prefix(&snapshot_root)
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
                    None => hub_file_identity(&mut api, identity, &relative_file)?,
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
    identity: &crate::models::HuggingFaceModelIdentity,
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
    identity: &crate::models::HuggingFaceModelIdentity,
    file: &str,
) -> Result<(String, u64)> {
    let api = match api {
        Some(api) => api,
        None => api.insert(crate::models::build_hf_api(false)?),
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
    if crate::models::huggingface_identity_for_path(path).is_none() {
        return Ok(None);
    }
    let Some(repo_dir) = path.ancestors().find_map(|revision_dir| {
        let snapshots_dir = revision_dir.parent()?;
        (snapshots_dir.file_name() == Some(OsStr::new("snapshots")))
            .then(|| snapshots_dir.parent())
            .flatten()
    }) else {
        return Ok(None);
    };
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
        .join(".mesh-llm")
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
    expected: Option<&[super::super::local_source::VerifiedFileFingerprint]>,
) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let paths = source_files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        super::super::local_source::verified_path_fingerprint(&paths).as_deref() == Some(expected),
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
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn hf_identity(repo_id: &str, revision: &str) -> crate::models::HuggingFaceModelIdentity {
        crate::models::HuggingFaceModelIdentity {
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            file: "model-00001-of-00002.gguf".to_string(),
            canonical_ref: format!("{repo_id}@{revision}/model-00001-of-00002.gguf"),
            local_file_name: "model-00001-of-00002.gguf".to_string(),
        }
    }

    fn hf_identity_files() -> Vec<HuggingFaceGgufIdentityFile> {
        vec![
            HuggingFaceGgufIdentityFile {
                path: "model-00001-of-00002.gguf".to_string(),
                bytes: 100,
                sha256: "a".repeat(64),
            },
            HuggingFaceGgufIdentityFile {
                path: "model-00002-of-00002.gguf".to_string(),
                bytes: 200,
                sha256: "b".repeat(64),
            },
        ]
    }

    #[test]
    fn hugging_face_identity_is_stable_across_local_cache_paths() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let first_path = first_dir.path().join("model.gguf");
        let second_path = second_dir.path().join("model.gguf");
        write_test_metadata_gguf(&first_path, 4096);
        std::fs::copy(&first_path, &second_path).unwrap();
        let identity = hf_identity("owner/model", &"c".repeat(40));
        let files = hf_identity_files();
        let identity_sha256 = huggingface_identity_sha256(&identity, &files).unwrap();
        let bytes = first_path.metadata().unwrap().len();
        let source_file = |path| SkippyPackageSourceFile {
            path,
            bytes,
            sha256: "a".repeat(64),
        };
        let first = super::super::synthetic_gguf_package_from_source_files(
            vec![source_file(first_path)],
            None,
            Some(identity_sha256.clone()),
            true,
        )
        .unwrap();
        let second = super::super::synthetic_gguf_package_from_source_files(
            vec![source_file(second_path)],
            None,
            Some(identity_sha256),
            true,
        )
        .unwrap();

        assert_eq!(first.package_ref, second.package_ref);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.source_model_sha256, second.source_model_sha256);
        assert_ne!(first.source_model_path, second.source_model_path);
        assert_eq!(
            super::super::super::local_source::verify_registered_content_source(
                "owner/model",
                &first.package_ref,
                &first.manifest_sha256,
                &first.source_model_sha256,
            )
            .unwrap()
            .package_ref,
            first.package_ref
        );
    }

    #[test]
    fn hugging_face_identity_changes_with_commit_or_ordered_file_set() {
        let first = hf_identity("owner/model", &"c".repeat(40));
        let second = hf_identity("owner/model", &"d".repeat(40));
        let files = hf_identity_files();
        let first_digest = huggingface_identity_sha256(&first, &files).unwrap();

        assert_ne!(
            first_digest,
            huggingface_identity_sha256(&second, &files).unwrap()
        );

        let mut reordered = files.clone();
        reordered.reverse();
        assert_ne!(
            first_digest,
            huggingface_identity_sha256(&first, &reordered).unwrap()
        );

        let mut renamed = files;
        renamed[0].path = "other-00001-of-00002.gguf".to_string();
        assert_ne!(
            first_digest,
            huggingface_identity_sha256(&first, &renamed).unwrap()
        );
    }

    fn push_test_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as i64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_test_metadata_gguf(path: &Path, context_length: u32) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&4i64.to_le_bytes());
        push_test_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        push_test_gguf_string(&mut bytes, "llama");
        for (key, value) in [
            ("llama.block_count", 2),
            ("llama.embedding_length", 128),
            ("llama.context_length", context_length),
        ] {
            push_test_gguf_string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn identity_ignores_alias_and_absolute_path() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let first_path = first_dir.path().join("first-name.gguf");
        let second_path = second_dir.path().join("other-name.gguf");
        write_test_metadata_gguf(&first_path, 4096);
        std::fs::copy(&first_path, &second_path).unwrap();

        let first = synthetic_content_addressed_gguf_package("alias-a", &first_path).unwrap();
        let second = synthetic_content_addressed_gguf_package("alias-b", &second_path).unwrap();

        assert_eq!(first.package_ref, second.package_ref);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.source_model_sha256, second.source_model_sha256);
        assert_ne!(first.source_model_path, second.source_model_path);
        assert!(
            super::super::super::local_source::is_content_addressed_gguf_ref(&first.package_ref)
        );
    }

    #[test]
    fn identity_changes_with_source_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.gguf");
        let second_path = dir.path().join("second.gguf");
        write_test_metadata_gguf(&first_path, 4096);
        write_test_metadata_gguf(&second_path, 8192);

        let first = synthetic_content_addressed_gguf_package("same-alias", &first_path).unwrap();
        let second = synthetic_content_addressed_gguf_package("same-alias", &second_path).unwrap();

        assert_ne!(first.package_ref, second.package_ref);
        assert_ne!(first.manifest_sha256, second.manifest_sha256);
        assert_ne!(first.source_model_sha256, second.source_model_sha256);
    }

    #[test]
    fn split_digest_is_path_independent_and_order_sensitive() {
        let first = vec![
            SkippyPackageSourceFile {
                path: PathBuf::from("/node-a/model-00001-of-00002.gguf"),
                bytes: 10,
                sha256: "a".repeat(64),
            },
            SkippyPackageSourceFile {
                path: PathBuf::from("/node-a/model-00002-of-00002.gguf"),
                bytes: 20,
                sha256: "b".repeat(64),
            },
        ];
        let relocated = vec![
            SkippyPackageSourceFile {
                path: PathBuf::from("/other/first.gguf"),
                ..first[0].clone()
            },
            SkippyPackageSourceFile {
                path: PathBuf::from("/other/second.gguf"),
                ..first[1].clone()
            },
        ];
        let reversed = vec![first[1].clone(), first[0].clone()];

        assert_eq!(
            aggregate_source_sha256(&first),
            aggregate_source_sha256(&relocated)
        );
        assert_ne!(
            aggregate_source_sha256(&first),
            aggregate_source_sha256(&reversed)
        );
    }

    #[test]
    fn split_identity_matches_across_paths_and_filename_prefixes() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let first_primary = first_dir.path().join("alpha-00001-of-00002.gguf");
        let first_secondary = first_dir.path().join("alpha-00002-of-00002.gguf");
        let second_primary = second_dir.path().join("beta-00001-of-00002.gguf");
        let second_secondary = second_dir.path().join("beta-00002-of-00002.gguf");
        write_test_metadata_gguf(&first_primary, 4096);
        write_test_metadata_gguf(&first_secondary, 4096);
        std::fs::copy(&first_primary, &second_primary).unwrap();
        std::fs::copy(&first_secondary, &second_secondary).unwrap();

        let first =
            synthetic_content_addressed_gguf_package("logical/model", &first_primary).unwrap();
        let second =
            synthetic_content_addressed_gguf_package("logical/model", &second_primary).unwrap();

        assert_eq!(first.source_files.len(), 2);
        assert_eq!(second.source_files.len(), 2);
        assert_eq!(first.package_ref, second.package_ref);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.source_model_sha256, second.source_model_sha256);
        assert_ne!(first.source_model_path, second.source_model_path);
    }

    #[test]
    fn identity_requires_an_absolute_path() {
        let error = synthetic_content_addressed_gguf_package(
            "logical/model",
            Path::new("relative-model.gguf"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("must be absolute"));
    }

    #[cfg(unix)]
    #[test]
    fn identity_rejects_user_supplied_symlink() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model-target.gguf");
        let link = dir.path().join("model.gguf");
        write_test_metadata_gguf(&target, 4096);
        symlink(&target, &link).unwrap();

        let error = synthetic_content_addressed_gguf_package("logical/model", &link)
            .unwrap_err()
            .to_string();

        assert!(error.contains("non-symlink file"));
        assert!(error.contains("model.gguf"));
    }

    #[cfg(unix)]
    #[test]
    fn split_rejects_symlinked_secondary_shard() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let primary = dir.path().join("model-00001-of-00002.gguf");
        let secondary = dir.path().join("model-00002-of-00002.gguf");
        let target = dir.path().join("secondary-target.gguf");
        write_test_metadata_gguf(&primary, 4096);
        write_test_metadata_gguf(&target, 4096);
        symlink(&target, &secondary).unwrap();

        let error = synthetic_content_addressed_gguf_package("logical/model", &primary)
            .unwrap_err()
            .to_string();

        assert!(error.contains("non-symlink file"));
        assert!(error.contains("00002-of-00002"));
    }

    #[cfg(unix)]
    #[test]
    fn split_accepts_hugging_face_snapshot_links_and_builds_regular_named_view() {
        use std::os::unix::fs::symlink;

        let cache = tempfile::tempdir().unwrap();
        let repo = cache.path().join("models--example--model");
        let blobs = repo.join("blobs");
        let snapshot = repo.join("snapshots").join("revision");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();

        let first_blob = blobs.join("first-blob");
        let second_blob = blobs.join("second-blob");
        write_test_metadata_gguf(&first_blob, 4096);
        write_test_metadata_gguf(&second_blob, 4096);

        let first = snapshot.join("model-00001-of-00002.gguf");
        let second = snapshot.join("model-00002-of-00002.gguf");
        symlink(Path::new("../../blobs/first-blob"), &first).unwrap();
        symlink(Path::new("../../blobs/second-blob"), &second).unwrap();

        validate_source_set(&first).unwrap();
        let source_paths = super::super::direct_gguf_source_paths(&first).unwrap();

        assert_eq!(
            source_paths
                .iter()
                .filter_map(|path| path.file_name())
                .collect::<Vec<_>>(),
            vec![first.file_name().unwrap(), second.file_name().unwrap()]
        );
        assert!(source_paths.iter().all(|path| {
            let metadata = std::fs::symlink_metadata(path).unwrap();
            metadata.is_file() && !metadata.file_type().is_symlink()
        }));
        assert!(same_file::is_same_file(&first_blob, &source_paths[0]).unwrap());
        assert!(same_file::is_same_file(&second_blob, &source_paths[1]).unwrap());

        let reused = super::super::direct_gguf_source_paths(&first).unwrap();
        assert_eq!(reused, source_paths);

        std::fs::remove_file(&source_paths[1]).unwrap();
        std::fs::write(&source_paths[1], b"tampered").unwrap();
        let error = super::super::direct_gguf_source_paths(&first)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not reference its verified Hugging Face blob"));
    }

    #[cfg(unix)]
    #[test]
    fn hugging_face_snapshot_link_rejects_blob_store_escape() {
        use std::os::unix::fs::symlink;

        let cache = tempfile::tempdir().unwrap();
        let repo = cache.path().join("models--example--model");
        let snapshot = repo.join("snapshots").join("revision");
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();

        let outside = cache.path().join("outside.gguf");
        let model = snapshot.join("model.gguf");
        write_test_metadata_gguf(&outside, 4096);
        symlink(&outside, &model).unwrap();

        let error = validate_source_set(&model).unwrap_err().to_string();

        assert!(error.contains("escapes its blob store"), "{error}");
    }

    // APFS rejects invalid UTF-8 path bytes at creation time; Linux permits
    // them and therefore exercises the canonical-parent edge directly.
    #[cfg(target_os = "linux")]
    #[test]
    fn identity_rejects_non_utf8_canonical_parent() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target_dir = dir.path().join(OsString::from_vec(vec![b'm', 0xff]));
        std::fs::create_dir(&target_dir).unwrap();
        let target_model = target_dir.join("model.gguf");
        write_test_metadata_gguf(&target_model, 4096);
        let utf8_parent = dir.path().join("models");
        symlink(&target_dir, &utf8_parent).unwrap();

        let error = synthetic_content_addressed_gguf_package(
            "logical/model",
            &utf8_parent.join("model.gguf"),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("canonical content-addressed GGUF path"));
        assert!(error.contains("valid UTF-8"));
    }
}
