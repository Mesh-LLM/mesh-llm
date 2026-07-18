use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_runtime::package::PackageGenerationInfo;

use super::hash_cache::{self, SidecarDigestCache};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkippyPackageIdentity {
    pub package_ref: String,
    pub manifest_sha256: String,
    pub source_model_path: PathBuf,
    pub source_model_sha256: String,
    pub source_model_bytes: u64,
    pub source_files: Vec<SkippyPackageSourceFile>,
    pub layer_weight_bytes: Vec<u64>,
    pub layer_count: u32,
    pub activation_width: u32,
    pub tensor_count: u64,
    pub generation: Option<PackageGenerationInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SkippyPackageSourceFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Serialize)]
struct SyntheticGgufManifest<'a> {
    schema_version: u32,
    package_kind: &'a str,
    model_id: &'a str,
    package_ref: &'a str,
    source_model_path: &'a str,
    source_model_sha256: &'a str,
    source_model_bytes: u64,
    source_files: &'a [SyntheticGgufManifestFile],
    architecture: &'a str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
}

#[derive(Serialize)]
struct SyntheticGgufManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

pub fn synthetic_direct_gguf_package(
    model_id: &str,
    model_path: &Path,
) -> Result<SkippyPackageIdentity> {
    let digest_cache = SidecarDigestCache::open_default();
    let source_files = direct_gguf_source_files(model_path, digest_cache.as_ref())?;

    let source_model_path = source_files
        .first()
        .map(|file| file.path.clone())
        .context("direct GGUF source file list is empty")?;

    let compact = crate::models::gguf::scan_gguf_compact_meta(&source_model_path)
        .with_context(|| format!("read GGUF metadata {}", source_model_path.display()))?;

    let tensor_count = gguf_tensor_count(&source_model_path)
        .with_context(|| format!("read GGUF tensor count {}", source_model_path.display()))?;

    anyhow::ensure!(
        compact.layer_count > 0,
        "GGUF metadata for {} does not contain a positive layer count",
        source_model_path.display()
    );
    anyhow::ensure!(
        compact.embedding_size > 0,
        "GGUF metadata for {} does not contain a positive embedding size",
        source_model_path.display()
    );
    let source_model_bytes = source_files.iter().map(|file| file.bytes).sum();

    let source_model_sha256 = aggregate_source_sha256(&source_files);

    let package_ref = format!("gguf://{}", source_model_path.display());

    let manifest_sha256 = synthetic_manifest_sha256(SyntheticManifestInput {
        model_id,
        package_ref: &package_ref,
        source_model_path: &source_model_path.to_string_lossy(),
        source_model_sha256: &source_model_sha256,
        source_model_bytes,
        source_files: &source_files,
        architecture: &compact.architecture,
        context_length: compact.context_length,
        layer_count: compact.layer_count,
        activation_width: compact.embedding_size,
        tensor_count,
    })?;

    Ok(SkippyPackageIdentity {
        package_ref,
        manifest_sha256,
        source_model_path,
        source_model_sha256,
        source_model_bytes,
        source_files,
        layer_weight_bytes: Vec::new(),
        layer_count: compact.layer_count,
        activation_width: compact.embedding_size,
        tensor_count,
        generation: None,
    })
}

struct SyntheticManifestInput<'a> {
    model_id: &'a str,
    package_ref: &'a str,
    source_model_path: &'a str,
    source_model_sha256: &'a str,
    source_model_bytes: u64,
    source_files: &'a [SkippyPackageSourceFile],
    architecture: &'a str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
}

fn synthetic_manifest_sha256(input: SyntheticManifestInput<'_>) -> Result<String> {
    let files = input
        .source_files
        .iter()
        .map(|file| SyntheticGgufManifestFile {
            path: file.path.to_string_lossy().to_string(),
            bytes: file.bytes,
            sha256: file.sha256.clone(),
        })
        .collect::<Vec<_>>();
    let manifest = SyntheticGgufManifest {
        schema_version: 1,
        package_kind: "direct-gguf",
        model_id: input.model_id,
        package_ref: input.package_ref,
        source_model_path: input.source_model_path,
        source_model_sha256: input.source_model_sha256,
        source_model_bytes: input.source_model_bytes,
        source_files: &files,
        architecture: input.architecture,
        context_length: input.context_length,
        layer_count: input.layer_count,
        activation_width: input.activation_width,
        tensor_count: input.tensor_count,
    };
    let bytes = serde_json::to_vec(&manifest).context("serialize synthetic GGUF manifest")?;
    Ok(hex_lower(&Sha256::digest(bytes)))
}

fn direct_gguf_source_files(
    model_path: &Path,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<Vec<SkippyPackageSourceFile>> {
    let canonical = model_path
        .canonicalize()
        .with_context(|| format!("canonicalize GGUF path {}", model_path.display()))?;
    let Some(file_name) = canonical.file_name().and_then(|name| name.to_str()) else {
        anyhow::bail!("GGUF path has no UTF-8 filename: {}", canonical.display());
    };
    let Some(shard) = model_ref::split_gguf_shard_info(file_name) else {
        let file = source_file(&canonical, digest_cache)?;
        return Ok(vec![file]);
    };
    anyhow::ensure!(
        shard.part == "00001",
        "split GGUF inputs must point at the first shard, got {}",
        canonical.display()
    );
    let total = shard
        .total
        .parse::<u32>()
        .with_context(|| format!("parse split GGUF shard total in {file_name}"))?;
    anyhow::ensure!(
        total > 0,
        "split GGUF shard total must be greater than zero"
    );
    let parent = canonical
        .parent()
        .with_context(|| format!("split GGUF shard has no parent: {}", canonical.display()))?;
    let mut files = Vec::with_capacity(total as usize);
    for index in 1..=total {
        let shard_name = format!("{}-{index:05}-of-{:05}.gguf", shard.prefix, total);
        let path = parent.join(shard_name);
        files.push(source_file(&path, digest_cache).with_context(|| {
            format!(
                "read split GGUF shard {index}/{total} for {}",
                canonical.display()
            )
        })?);
    }
    Ok(files)
}

fn source_file(
    path: &Path,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<SkippyPackageSourceFile> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize GGUF source {}", path.display()))?;
    let metadata = canonical
        .metadata()
        .with_context(|| format!("stat GGUF source {}", canonical.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "GGUF source is not a file: {}",
        canonical.display()
    );
    let sha256 = source_file_sha256(&canonical, &metadata, digest_cache)?;
    Ok(SkippyPackageSourceFile {
        path: canonical.clone(),
        bytes: metadata.len(),
        sha256,
    })
}

/// SHA-256 of a source file, served from the sidecar cache when the file's
/// `(size, mtime, ctime)` is unchanged and recomputed (then cached) otherwise.
fn source_file_sha256(
    path: &Path,
    metadata: &std::fs::Metadata,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<String> {
    let mtime_nanos = hash_cache::file_mtime_nanos(metadata);
    let ctime_nanos = hash_cache::file_ctime_nanos(metadata);
    if let (Some(cache), Some(mtime_nanos)) = (digest_cache, mtime_nanos)
        && let Some(sha256) = cache.lookup(path, metadata.len(), mtime_nanos, ctime_nanos)
    {
        tracing::debug!(path = %path.display(), "GGUF source sha256 cache hit");
        return Ok(sha256);
    }
    let started = Instant::now();
    let sha256 = file_sha256(path)?;
    tracing::debug!(
        path = %path.display(),
        bytes = metadata.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "computed GGUF source sha256"
    );
    if let (Some(cache), Some(mtime_nanos)) = (digest_cache, mtime_nanos) {
        cache.store(path, metadata.len(), mtime_nanos, ctime_nanos, &sha256);
    }
    Ok(sha256)
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("open GGUF source {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash GGUF source {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn aggregate_source_sha256(source_files: &[SkippyPackageSourceFile]) -> String {
    if source_files.len() == 1 {
        return source_files[0].sha256.clone();
    }
    let mut hasher = Sha256::new();
    for file in source_files {
        hasher.update(file.path.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(file.bytes.to_le_bytes());
        hasher.update([0]);
        hasher.update(file.sha256.as_bytes());
        hasher.update([0]);
    }
    hex_lower(&hasher.finalize())
}

fn gguf_tensor_count(path: &Path) -> Result<u64> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("open GGUF {}", path.display()))?);
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .with_context(|| format!("read GGUF magic {}", path.display()))?;
    anyhow::ensure!(&magic == b"GGUF", "not a GGUF file: {}", path.display());
    let version = read_u32_le(&mut reader)?;
    anyhow::ensure!(
        version >= 2,
        "unsupported GGUF version {version} in {}",
        path.display()
    );
    read_gguf_count(&mut reader, version)
}

fn read_u32_le(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i64_le(reader: &mut impl Read) -> Result<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read i64")?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_gguf_count(reader: &mut impl Read, _version: u32) -> Result<u64> {
    let value = read_i64_le(reader)?;
    u64::try_from(value).context("GGUF count is negative")
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Build a `SkippyPackageIdentity` from a remote HF layer package.
///
/// Resolves the package into the local HF cache for inspection, downloading
/// the manifest and shared metadata that the resolver requires, but not layer
/// files. Layer artifacts are fetched later by the node that materializes or
/// loads its assigned stage.
pub fn identity_from_layer_package(package_ref: &str) -> Result<SkippyPackageIdentity> {
    // Resolve hf:// to a local package dir for lightweight package inspection.
    let local_ref =
        super::materialization::resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let info = skippy_runtime::package::inspect_layer_package(&local_ref)
        .with_context(|| format!("inspect layer package {package_ref}"))?;

    let activation_width =
        required_layer_package_activation_width(package_ref, info.activation_width)?;
    let source_model_bytes = info
        .source_model_bytes
        .unwrap_or_else(|| info.layers.iter().map(|l| l.artifact_bytes).sum::<u64>());
    let layer_weight_bytes = layer_weight_bytes_from_info(&info);

    // For local paths inside an HF cache, convert to an exact hf:// ref so all
    // nodes resolve the same snapshot independently. HF cache dirs look like:
    // .../models--owner--name/snapshots/<hash>/
    let canonical_package_ref = canonical_layer_package_ref(package_ref, &local_ref);

    Ok(SkippyPackageIdentity {
        package_ref: canonical_package_ref,
        manifest_sha256: info.manifest_sha256,
        source_model_path: PathBuf::from(&info.source_model_path),
        source_model_sha256: info.source_model_sha256,
        source_model_bytes,
        source_files: Vec::new(),
        layer_weight_bytes,
        layer_count: info.layer_count,
        activation_width,
        tensor_count: info.layers.iter().map(|l| l.tensor_count as u64).sum(),
        generation: info.generation,
    })
}

fn layer_weight_bytes_from_info(info: &skippy_runtime::package::LayerPackageInfo) -> Vec<u64> {
    let mut layers = info.layers.clone();
    layers.sort_by_key(|layer| layer.layer_index);
    if layers.len() != info.layer_count as usize
        || layers
            .iter()
            .enumerate()
            .any(|(index, layer)| layer.layer_index as usize != index)
    {
        return Vec::new();
    }
    let mut weights = layers
        .into_iter()
        .map(|layer| layer.tensor_bytes.max(layer.artifact_bytes))
        .collect::<Vec<_>>();
    let accounted = weights.iter().copied().sum::<u64>();
    let unaccounted = info
        .source_model_bytes
        .unwrap_or_default()
        .saturating_sub(accounted);
    if let Some((first, rest)) = weights.split_first_mut() {
        *first = first.saturating_add(unaccounted.div_ceil(2));
        if let Some(last) = rest.last_mut() {
            *last = last.saturating_add(unaccounted / 2);
        } else {
            *first = first.saturating_add(unaccounted / 2);
        }
    }
    weights
}

/// Detect if a local path is inside an HF cache directory and convert to `hf://` ref.
///
/// HF cache paths look like:
///   `.../hub/models--owner--name/snapshots/<hash>/`
///
/// Returns `Some("hf://owner/name@hash")` if detected, `None` otherwise.
fn hf_ref_from_cache_path(path: &str) -> Option<String> {
    // Walk path components looking for "models--*" followed by "snapshots"
    let path = std::path::Path::new(path);
    let components: Vec<&std::ffi::OsStr> = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s),
            _ => None,
        })
        .collect();
    for (i, comp) in components.iter().enumerate() {
        let s = comp.to_str()?;
        if let Some(repo_part) = s.strip_prefix("models--") {
            // Verify next component is "snapshots" and preserve the exact
            // snapshot revision/hash so peers fetch identical package content.
            if components.get(i + 1).and_then(|c| c.to_str()) == Some("snapshots") {
                let revision = components.get(i + 2)?.to_str()?;
                // repo_part is "owner--name", convert to "owner/name"
                let repo = repo_part.replacen("--", "/", 1);
                if repo.contains('/') {
                    return Some(format!("hf://{repo}@{revision}"));
                }
            }
        }
    }
    None
}

fn canonical_layer_package_ref(package_ref: &str, local_ref: &str) -> String {
    hf_ref_from_cache_path(local_ref)
        .or_else(|| hf_ref_from_cache_path(package_ref))
        .unwrap_or_else(|| package_ref.to_string())
}

fn required_layer_package_activation_width(
    package_ref: &str,
    activation_width: Option<u32>,
) -> Result<u32> {
    activation_width.with_context(|| {
        format!(
            "layer package {package_ref} is missing activation_width; rebuild the package manifest"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_manifest_identity_is_stable_and_metadata_sensitive() {
        let source_files = vec![SkippyPackageSourceFile {
            path: PathBuf::from("/models/model.gguf"),
            bytes: 12,
            sha256: "abc123".to_string(),
        }];
        let first = synthetic_manifest_sha256(SyntheticManifestInput {
            model_id: "model-a",
            package_ref: "gguf:///models/model.gguf",
            source_model_path: "/models/model.gguf",
            source_model_sha256: "abc123",
            source_model_bytes: 12,
            source_files: &source_files,
            architecture: "llama",
            context_length: 4096,
            layer_count: 32,
            activation_width: 4096,
            tensor_count: 100,
        })
        .unwrap();
        let second = synthetic_manifest_sha256(SyntheticManifestInput {
            model_id: "model-a",
            package_ref: "gguf:///models/model.gguf",
            source_model_path: "/models/model.gguf",
            source_model_sha256: "abc123",
            source_model_bytes: 12,
            source_files: &source_files,
            architecture: "llama",
            context_length: 4096,
            layer_count: 32,
            activation_width: 4096,
            tensor_count: 100,
        })
        .unwrap();
        let changed = synthetic_manifest_sha256(SyntheticManifestInput {
            model_id: "model-a",
            package_ref: "gguf:///models/model.gguf",
            source_model_path: "/models/model.gguf",
            source_model_sha256: "abc123",
            source_model_bytes: 12,
            source_files: &source_files,
            architecture: "llama",
            context_length: 4096,
            layer_count: 33,
            activation_width: 4096,
            tensor_count: 100,
        })
        .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, changed);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn direct_gguf_source_files_expand_split_shards() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("Model-Q4_K_M-00001-of-00003.gguf");
        std::fs::write(&first, b"one").unwrap();
        std::fs::write(dir.path().join("Model-Q4_K_M-00002-of-00003.gguf"), b"two").unwrap();
        std::fs::write(
            dir.path().join("Model-Q4_K_M-00003-of-00003.gguf"),
            b"three",
        )
        .unwrap();

        let files = direct_gguf_source_files(&first, None).unwrap();

        assert_eq!(files.len(), 3);
        assert_eq!(
            files.iter().map(|file| file.bytes).collect::<Vec<_>>(),
            vec![3, 3, 5]
        );
        assert!(files[0].path.ends_with("Model-Q4_K_M-00001-of-00003.gguf"));
        assert!(files[2].path.ends_with("Model-Q4_K_M-00003-of-00003.gguf"));
    }

    #[test]
    fn direct_gguf_source_files_report_missing_split_shard() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("Model-Q4_K_M-00001-of-00002.gguf");
        std::fs::write(&first, b"one").unwrap();

        let error = direct_gguf_source_files(&first, None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("split GGUF shard 2/2"));
    }

    #[test]
    fn direct_gguf_source_files_reject_non_primary_split_shard() {
        let dir = tempfile::tempdir().unwrap();
        let second = dir.path().join("Model-Q4_K_M-00002-of-00002.gguf");
        std::fs::write(&second, b"two").unwrap();

        let error = direct_gguf_source_files(&second, None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("first shard"));
    }

    #[test]
    fn source_file_sha256_is_stable_and_content_sensitive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content-a").unwrap();

        let first = source_file(&path, None).unwrap();
        let second = source_file(&path, None).unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.sha256.len(), 64);
        assert!(first.sha256.chars().all(|c| c.is_ascii_hexdigit()));

        std::fs::write(&path, b"content-b-longer").unwrap();
        let changed = source_file(&path, None).unwrap();
        assert_ne!(first.sha256, changed.sha256);
    }

    #[test]
    fn source_file_reuses_cached_sha256_while_metadata_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content").unwrap();
        let cache = SidecarDigestCache::open_in(dir.path().join("hashes"));

        let computed = source_file(&path, Some(&cache)).unwrap();

        // A matching (size, mtime, ctime) record now exists; prove the second
        // call serves it by making the cached value observably distinct while
        // still shaped like a real SHA-256.
        let distinct_sha256 = "f".repeat(64);
        let canonical = path.canonicalize().unwrap();
        let metadata = canonical.metadata().unwrap();
        let mtime_nanos = hash_cache::file_mtime_nanos(&metadata).unwrap();
        let ctime_nanos = hash_cache::file_ctime_nanos(&metadata);
        cache.store(
            &canonical,
            metadata.len(),
            mtime_nanos,
            ctime_nanos,
            &distinct_sha256,
        );

        let cached = source_file(&path, Some(&cache)).unwrap();
        assert_eq!(cached.sha256, distinct_sha256);
        assert_ne!(computed.sha256, cached.sha256);
    }

    /// Regression test for the review concern on the sidecar cache: a GGUF
    /// replaced with different same-size content while tooling restores its
    /// mtime must not reuse the stale hash. The inode ctime moves on any
    /// rewrite and cannot be restored from userspace, so the cache misses.
    #[cfg(unix)]
    #[test]
    fn source_file_recomputes_when_same_size_content_replaced_with_restored_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content-a").unwrap();
        let cache = SidecarDigestCache::open_in(dir.path().join("hashes"));

        let first = source_file(&path, Some(&cache)).unwrap();
        let original_mtime = path.metadata().unwrap().modified().unwrap();

        // Inode timestamps use a coarse clock; wait long enough that the
        // rewrite lands on a later ctime tick than the original write.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Replace with different content of identical size, then restore the
        // original mtime the way rsync/tar/cp --preserve style tooling does.
        std::fs::write(&path, b"content-b").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(original_mtime)
            .unwrap();
        let metadata = path.metadata().unwrap();
        assert_eq!(metadata.len(), first.bytes);
        assert_eq!(metadata.modified().unwrap(), original_mtime);

        let replaced = source_file(&path, Some(&cache)).unwrap();
        assert_ne!(first.sha256, replaced.sha256);
        assert_eq!(replaced.sha256, file_sha256(&path).unwrap());
    }

    #[test]
    fn source_file_recomputes_when_size_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content-a").unwrap();
        let cache = SidecarDigestCache::open_in(dir.path().join("hashes"));

        let first = source_file(&path, Some(&cache)).unwrap();

        std::fs::write(&path, b"content-b-longer").unwrap();
        let changed = source_file(&path, Some(&cache)).unwrap();
        assert_ne!(first.sha256, changed.sha256);
        assert_eq!(changed.sha256, file_sha256(&path).unwrap());
    }

    #[test]
    fn hf_ref_from_cache_path_preserves_snapshot_revision() {
        let package_ref =
            "/cache/hub/models--meshllm--Qwen3-layers/snapshots/abc123/model-package.json";

        assert_eq!(
            hf_ref_from_cache_path(package_ref),
            Some("hf://meshllm/Qwen3-layers@abc123".to_string())
        );
    }

    #[test]
    fn canonical_layer_package_ref_prefers_resolved_snapshot() {
        let local_ref = "/cache/hub/models--meshllm--Qwen3-layers/snapshots/abc123";

        assert_eq!(
            canonical_layer_package_ref("hf://meshllm/Qwen3-layers@main", local_ref),
            "hf://meshllm/Qwen3-layers@abc123"
        );
    }

    #[test]
    fn layer_package_activation_width_is_required() {
        let error =
            required_layer_package_activation_width("hf://meshllm/Qwen3-layers@abc123", None)
                .unwrap_err()
                .to_string();

        assert!(error.contains("missing activation_width"));
        assert!(error.contains("rebuild the package manifest"));
    }

    #[test]
    fn package_layer_weights_include_shared_model_bytes_at_endpoints() {
        let info = skippy_runtime::package::LayerPackageInfo {
            package_dir: PathBuf::from("/models/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "org/model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(120),
            layer_count: 2,
            activation_width: Some(1024),
            generation: None,
            projectors: Vec::new(),
            layers: vec![
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 0,
                    tensor_count: 1,
                    tensor_bytes: 30,
                    artifact_bytes: 30,
                },
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 1,
                    tensor_count: 1,
                    tensor_bytes: 40,
                    artifact_bytes: 40,
                },
            ],
        };

        assert_eq!(layer_weight_bytes_from_info(&info), vec![55, 65]);
    }

    #[test]
    fn package_layer_weights_require_contiguous_indices() {
        let mut info = skippy_runtime::package::LayerPackageInfo {
            package_dir: PathBuf::from("/models/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "org/model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(70),
            layer_count: 2,
            activation_width: Some(1024),
            generation: None,
            projectors: Vec::new(),
            layers: vec![
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 0,
                    tensor_count: 1,
                    tensor_bytes: 30,
                    artifact_bytes: 30,
                },
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 2,
                    tensor_count: 1,
                    tensor_bytes: 40,
                    artifact_bytes: 40,
                },
            ],
        };

        assert!(layer_weight_bytes_from_info(&info).is_empty());
        info.layers[1].layer_index = 1;
        assert_eq!(layer_weight_bytes_from_info(&info), vec![30, 40]);
    }
}
