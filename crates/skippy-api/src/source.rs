//! Source identity and preparation shared by standalone Skippy and Mesh.
use crate::hash_cache::SidecarDigestCache;
use crate::package::{SkippyPackageIdentity, SkippyPackageSourceFile, source_file};
#[cfg(test)]
use crate::{hash_cache, package::file_sha256};
use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_ffi::TensorRole;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};
mod content_addressed;
mod legacy_identity;
pub use content_addressed::synthetic_content_addressed_gguf_package;
pub mod planning;
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
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<SkippyPackageIdentity> {
    if let Some(root) = safetensors_checkpoint_root(model_path) {
        return synthetic_safetensors_package(model_id, &root, digest_cache);
    }
    synthetic_gguf_package(model_id, model_path)
}

/// Build a content-addressed GGUF package from the immutable provenance and
/// object digests already recorded by Hugging Face.
///
/// The canonical identity includes the repository, resolved commit, and exact
/// ordered file set, but avoids reading the model weights at startup.
pub fn synthetic_huggingface_gguf_package(
    _model_id: &str,
    identity: &model_hf::store::local::HuggingFaceModelIdentity,
    snapshot_root: &Path,
    build_client: impl FnMut() -> Result<hf_hub::HFClientSync>,
) -> Result<SkippyPackageIdentity> {
    let source =
        content_addressed::huggingface_source_files(identity, snapshot_root, build_client)?;
    let source_paths = source
        .files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    let verified_fingerprint = crate::source_registry::verified_path_fingerprint(&source_paths);
    synthetic_gguf_package_from_source_files(
        source.files,
        verified_fingerprint,
        Some(source.identity_sha256),
        true,
    )
}

fn safetensors_checkpoint_root(model_path: &Path) -> Option<PathBuf> {
    let root = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent()?
    };
    (root.join("config.json").is_file()
        && (root.join("model.safetensors").is_file()
            || root.join("model.safetensors.index.json").is_file()))
    .then(|| root.to_path_buf())
}

fn synthetic_safetensors_package(
    model_id: &str,
    checkpoint_root: &Path,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<SkippyPackageIdentity> {
    let checkpoint_root = checkpoint_root
        .canonicalize()
        .with_context(|| format!("canonicalize checkpoint {}", checkpoint_root.display()))?;
    let config_path = checkpoint_root.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .with_context(|| format!("read checkpoint config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse checkpoint config {}", config_path.display()))?;
    let config_u32 = |key: &str| -> Result<u32> {
        let value = config
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("checkpoint config missing integer {key}"))?;
        u32::try_from(value).with_context(|| format!("checkpoint config {key} exceeds u32"))
    };
    let layer_count = config_u32("num_hidden_layers")?;
    let activation_width = config_u32("hidden_size")?;
    anyhow::ensure!(layer_count > 0, "checkpoint layer count must be positive");
    anyhow::ensure!(
        activation_width > 0,
        "checkpoint hidden size must be positive"
    );
    let architecture = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("safetensors");
    let context_length = config
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or_else(|| {
            if architecture == "granitemoehybrid" {
                1 << 20
            } else {
                0
            }
        });
    let plan = skippy_model::hf_checkpoint::inspect_hf_checkpoint(&checkpoint_root, None, 1.0)?;
    let tensor_count = u64::try_from(plan.tensor_count).context("tensor count exceeds u64")?;

    let mut source_paths = skippy_model::hf_checkpoint::discover_safetensors(&checkpoint_root)?;
    for name in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "chat_template.jinja",
    ] {
        let path = checkpoint_root.join(name);
        if path.is_file() {
            source_paths.push(path);
        }
    }
    source_paths.sort();
    source_paths.dedup();
    let source_files = direct_gguf_source_files_from_paths(source_paths, digest_cache)?;
    let source_model_bytes = source_files.iter().map(|file| file.bytes).sum();
    let source_model_sha256 = legacy_identity::aggregate_source_sha256(&source_files);
    let package_ref = format!("safetensors://{}", checkpoint_root.display());
    let manifest_sha256 = synthetic_manifest_sha256(SyntheticManifestInput {
        model_id,
        package_kind: "direct-safetensors",
        package_ref: &package_ref,
        source_model_path: &checkpoint_root.to_string_lossy(),
        source_model_sha256: &source_model_sha256,
        source_model_bytes,
        source_files: &source_files,
        architecture,
        context_length,
        layer_count,
        activation_width,
        tensor_count,
    })?;
    Ok(SkippyPackageIdentity {
        package_ref,
        manifest_sha256,
        source_model_path: checkpoint_root,
        source_model_sha256,
        source_model_bytes,
        source_files,
        layer_weight_bytes: Vec::new(),
        layer_count,
        activation_width,
        tensor_count,
        generation: None,
    })
}

fn synthetic_gguf_package(_model_id: &str, model_path: &Path) -> Result<SkippyPackageIdentity> {
    // Direct local GGUF identity crosses the mesh and is therefore always
    // content addressed. Absolute paths and filenames are node-local locators;
    // they must never participate in model, package, or admission identity.
    content_addressed::validate_source_set(model_path)?;
    let source_paths = direct_gguf_source_paths(model_path)?;
    for path in &source_paths {
        anyhow::ensure!(
            path.to_str().is_some(),
            "canonical content-addressed GGUF path must be valid UTF-8: {}",
            path.display()
        );
    }
    let verified_fingerprint = crate::source_registry::verified_path_fingerprint(&source_paths);
    let source_files = direct_gguf_source_files_from_paths(source_paths, None)?;
    synthetic_gguf_package_from_source_files(source_files, verified_fingerprint, None, false)
}

fn synthetic_gguf_package_from_source_files(
    source_files: Vec<SkippyPackageSourceFile>,
    verified_fingerprint: Option<Vec<crate::source_registry::VerifiedFileFingerprint>>,
    source_identity_sha256: Option<String>,
    allow_immutable_source_metadata: bool,
) -> Result<SkippyPackageIdentity> {
    content_addressed::ensure_fingerprint_unchanged(
        &source_files,
        verified_fingerprint.as_deref(),
    )?;

    let source_model_path = source_files
        .first()
        .map(|file| file.path.clone())
        .context("direct GGUF source file list is empty")?;

    let compact = model_artifact::gguf::scan_gguf_compact_meta(&source_model_path)
        .with_context(|| format!("read GGUF metadata {}", source_model_path.display()))?;

    let tensor_count = gguf_tensor_count(&source_model_path)
        .with_context(|| format!("read GGUF tensor count {}", source_model_path.display()))?;

    let layer_count = compact.executable_layer_count().with_context(|| {
        format!(
            "GGUF metadata for {} does not contain a positive executable layer count (block_count={}, nextn_predict_layers={})",
            source_model_path.display(),
            compact.layer_count,
            compact.nextn_predict_layers
        )
    })?;
    anyhow::ensure!(
        compact.embedding_size > 0,
        "GGUF metadata for {} does not contain a positive embedding size",
        source_model_path.display()
    );
    let source_model_bytes = source_files.iter().map(|file| file.bytes).sum();
    let layer_weight_bytes = direct_gguf_layer_weight_bytes(&source_files, layer_count)
        .with_context(|| {
            format!(
                "inspect GGUF tensor weights {}",
                source_model_path.display()
            )
        })?;
    content_addressed::ensure_fingerprint_unchanged(
        &source_files,
        verified_fingerprint.as_deref(),
    )?;

    let source_model_sha256 = source_identity_sha256
        .unwrap_or_else(|| content_addressed::aggregate_source_sha256(&source_files));
    let package_ref = crate::source_registry::content_addressed_package_ref(&source_model_sha256)?;
    let manifest_sha256 = content_addressed::manifest_sha256(
        &source_model_sha256,
        source_model_bytes,
        &source_files,
        &compact.architecture,
        compact.context_length,
        layer_count,
        compact.embedding_size,
        tensor_count,
    )?;

    let identity = SkippyPackageIdentity {
        package_ref,
        manifest_sha256,
        source_model_path,
        source_model_sha256,
        source_model_bytes,
        source_files,
        layer_weight_bytes,
        layer_count,
        activation_width: compact.embedding_size,
        tensor_count,
        generation: None,
    };
    crate::source_registry::register_content_addressed_identity(
        &identity,
        verified_fingerprint,
        allow_immutable_source_metadata,
    );
    Ok(identity)
}

struct SyntheticManifestInput<'a> {
    model_id: &'a str,
    package_kind: &'a str,
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
        package_kind: input.package_kind,
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

pub fn direct_gguf_source_paths(model_path: &Path) -> Result<Vec<PathBuf>> {
    // Parse multipart names before canonicalizing. Hugging Face snapshots keep
    // those names on symlinks whose blob targets are content-addressed hashes;
    // canonicalizing the primary first would erase the shard-set information.
    let Some(file_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        anyhow::bail!("GGUF path has no UTF-8 filename: {}", model_path.display());
    };
    let Some(shard) = model_ref::split_gguf_shard_info(file_name) else {
        return Ok(vec![model_path.canonicalize().with_context(|| {
            format!("canonicalize GGUF path {}", model_path.display())
        })?]);
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
    anyhow::ensure!(
        total > 0,
        "split GGUF shard total must be greater than zero"
    );
    let parent = model_path
        .parent()
        .with_context(|| format!("split GGUF shard has no parent: {}", model_path.display()))?;
    let mut snapshot_paths = Vec::with_capacity(total as usize);
    for index in 1..=total {
        let shard_name = format!("{}-{index:05}-of-{:05}.gguf", shard.prefix, total);
        let path = parent.join(shard_name);
        path.metadata().with_context(|| {
            format!(
                "read split GGUF shard {index}/{total} for {}",
                model_path.display()
            )
        })?;
        snapshot_paths.push(path);
    }
    if let Some(view) = content_addressed::managed_hf_multipart_view(&snapshot_paths)? {
        return Ok(view);
    }
    snapshot_paths
        .into_iter()
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("canonicalize split GGUF shard {}", path.display()))
        })
        .collect()
}

#[cfg(test)]
fn direct_gguf_source_files(
    model_path: &Path,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<Vec<SkippyPackageSourceFile>> {
    direct_gguf_source_files_from_paths(direct_gguf_source_paths(model_path)?, digest_cache)
}

fn direct_gguf_source_files_from_paths(
    source_paths: Vec<PathBuf>,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<Vec<SkippyPackageSourceFile>> {
    source_paths
        .into_iter()
        .map(|path| source_file(&path, digest_cache))
        .collect()
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

fn direct_gguf_layer_weight_bytes(
    source_files: &[SkippyPackageSourceFile],
    layer_count: u32,
) -> Result<Vec<u64>> {
    if !skippy_runtime::native_runtime_loaded() {
        tracing::debug!(
            "GGUF tensor layout unavailable because the native runtime is not loaded; \
             using capacity-based split planning"
        );
        return Ok(Vec::new());
    }

    let mut tensors = Vec::new();
    for source_file in source_files {
        let info = match skippy_runtime::ModelInfo::open(&source_file.path) {
            Ok(info) => info,
            Err(error) => {
                tracing::debug!(
                    path = %source_file.path.display(),
                    error = %error,
                    "GGUF tensor layout unavailable; using capacity-based split planning"
                );
                return Ok(Vec::new());
            }
        };
        tensors.extend(
            info.tensors()
                .with_context(|| format!("read GGUF tensors {}", source_file.path.display()))?,
        );
    }
    Ok(layer_weight_bytes_from_tensors(&tensors, layer_count))
}

fn layer_weight_bytes_from_tensors(
    tensors: &[skippy_runtime::TensorInfo],
    layer_count: u32,
) -> Vec<u64> {
    let Ok(layer_count) = usize::try_from(layer_count) else {
        return Vec::new();
    };
    if layer_count == 0 {
        return Vec::new();
    }

    let mut weights = vec![0_u64; layer_count];
    let mut shared_bytes = 0_u64;
    let mut seen = std::collections::BTreeSet::new();

    for tensor in tensors {
        if !seen.insert(tensor.name.as_str()) {
            continue;
        }
        let bytes = tensor.byte_size;
        match tensor.layer_index {
            Some(layer) if (layer as usize) < layer_count => {
                weights[layer as usize] = weights[layer as usize].saturating_add(bytes);
            }
            // Native MTP blocks are appended after the trunk's declared layer
            // count and must stay with the final stage that owns logits.
            Some(_) => {
                let last = weights.len() - 1;
                weights[last] = weights[last].saturating_add(bytes);
            }
            None => match tensor.role {
                TensorRole::Embedding => {
                    weights[0] = weights[0].saturating_add(bytes);
                }
                TensorRole::FinalNorm | TensorRole::Output => {
                    let last = weights.len() - 1;
                    weights[last] = weights[last].saturating_add(bytes);
                }
                TensorRole::Unknown
                | TensorRole::Metadata
                | TensorRole::Tokenizer
                | TensorRole::Layer => {
                    shared_bytes = shared_bytes.saturating_add(bytes);
                }
            },
        }
    }

    // Metadata is loaded at every stage but is normally tiny. Split it between
    // endpoints so total model weight stays conserved without biasing a middle
    // stage in multi-node plans.
    weights[0] = weights[0].saturating_add(shared_bytes.div_ceil(2));
    let last = weights.len() - 1;
    weights[last] = weights[last].saturating_add(shared_bytes / 2);
    weights
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

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_runtime::TensorInfo;
    #[test]
    fn synthetic_direct_identity_accepts_safetensors_checkpoint_directory() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            r#"{
              "model_type": "qwen2",
              "num_hidden_layers": 1,
              "hidden_size": 4,
              "max_position_embeddings": 128
            }"#,
        )
        .unwrap();
        let header = serde_json::json!({
            "model.layers.0.input_layernorm.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        })
        .to_string();
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(header.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(header.as_bytes());
        safetensors.extend_from_slice(&1.0_f32.to_le_bytes());
        std::fs::write(root.path().join("model.safetensors"), safetensors).unwrap();

        let identity = synthetic_direct_gguf_package("test", root.path(), None).unwrap();

        assert!(identity.package_ref.starts_with("safetensors://"));
        assert_eq!(
            identity.source_model_path,
            root.path().canonicalize().unwrap()
        );
        assert_eq!(identity.layer_count, 1);
        assert_eq!(identity.activation_width, 4);
        assert_eq!(identity.tensor_count, 1);
        assert_eq!(identity.source_files.len(), 2);
    }

    #[test]
    fn synthetic_manifest_identity_is_stable_and_metadata_sensitive() {
        let source_files = vec![SkippyPackageSourceFile {
            path: PathBuf::from("/models/model.gguf"),
            bytes: 12,
            sha256: "abc123".to_string(),
        }];
        let first = synthetic_manifest_sha256(SyntheticManifestInput {
            model_id: "model-a",
            package_kind: "direct-gguf",
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
            package_kind: "direct-gguf",
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
            package_kind: "direct-gguf",
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
    fn direct_gguf_weights_charge_native_mtp_block_to_final_stage() {
        let tensors = vec![
            tensor("token_embd.weight", None, TensorRole::Embedding, 5),
            tensor("blk.0.attn_norm.weight", Some(0), TensorRole::Layer, 10),
            tensor("blk.1.attn_norm.weight", Some(1), TensorRole::Layer, 10),
            tensor("blk.2.nextn.eh_proj.weight", Some(2), TensorRole::Layer, 7),
            tensor("output_norm.weight", None, TensorRole::FinalNorm, 1),
            tensor("output.weight", None, TensorRole::Output, 9),
            tensor("general.alignment", None, TensorRole::Metadata, 3),
        ];

        assert_eq!(layer_weight_bytes_from_tensors(&tensors, 2), vec![17, 28]);
    }

    #[cfg(feature = "dynamic-native-runtime")]
    #[test]
    fn direct_gguf_weights_fall_back_when_dynamic_runtime_is_unloaded() {
        assert!(!skippy_runtime::native_runtime_loaded());
        let source_files = vec![SkippyPackageSourceFile {
            path: PathBuf::from("/models/not-opened.gguf"),
            bytes: 12,
            sha256: "abc123".to_string(),
        }];

        assert!(
            direct_gguf_layer_weight_bytes(&source_files, 2)
                .unwrap()
                .is_empty()
        );
    }

    fn tensor(
        name: &str,
        layer_index: Option<u32>,
        role: TensorRole,
        byte_size: u64,
    ) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            layer_index,
            role,
            ggml_type: 0,
            byte_size,
            element_count: byte_size,
        }
    }
}
