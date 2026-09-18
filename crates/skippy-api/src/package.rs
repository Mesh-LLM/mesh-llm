//! Verified local package identity and source-file digests.
//! Callers choose an advisory digest cache explicitly; None always hashes source bytes.
use crate::hash_cache::{self, SidecarDigestCache};
use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_package_format::{
    PackageManifest as PackageManifestV2, ProposerKind, StrategyKind, TensorStorage,
};
use skippy_runtime::package::PackageGenerationInfo;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};
const PACKAGE_V2_MANIFEST: &str = "model-package.json";
fn hex_lower(bytes: &[u8]) -> String {
    hex::encode(bytes)
}
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

/// Resolve a validated package-v2 directory into a shared planning identity. The content-derived package ID remains in the v2 manifest and is
/// carried into split control by the generation-10 admission descriptor.
pub fn identity_from_package_v2(
    package_dir: &Path,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<SkippyPackageIdentity> {
    let package_dir = package_dir.canonicalize().with_context(|| {
        format!(
            "canonicalize package-v2 directory {}",
            package_dir.display()
        )
    })?;
    anyhow::ensure!(
        package_dir.is_dir(),
        "package-v2 path is not a directory: {}",
        package_dir.display()
    );
    let manifest_path = package_dir.join(PACKAGE_V2_MANIFEST);
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?;
    let schema_version = serde_json::from_slice::<serde_json::Value>(&manifest_bytes)
        .context("parse package manifest envelope")?
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .context("package manifest is missing integer schema_version")?;
    anyhow::ensure!(
        schema_version == u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION),
        "split serving requires package schema {}; found schema {schema_version}",
        skippy_package_format::PACKAGE_SCHEMA_VERSION
    );
    let manifest: PackageManifestV2 =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, &package_dir)
            .context("resolve package-v2 metadata carrier")?;
    let computed_package_id = manifest
        .computed_package_id()
        .context("compute package-v2 identity")?;
    anyhow::ensure!(
        manifest.package_id == computed_package_id,
        "package-v2 manifest package_id does not match its content"
    );
    let required_native_abi = format!(
        "{}.{}.{}",
        skippy_ffi::ABI_VERSION_MAJOR,
        skippy_ffi::ABI_VERSION_MINOR,
        skippy_ffi::ABI_VERSION_PATCH
    );
    anyhow::ensure!(
        manifest.native_abi_version == required_native_abi,
        "package-v2 native ABI {} differs from runtime ABI {required_native_abi}",
        manifest.native_abi_version
    );

    let metadata_artifact = manifest
        .artifact_catalog
        .entries
        .iter()
        .find(|artifact| artifact.id == manifest.source_model.metadata_artifact_id)
        .context("package-v2 metadata artifact is absent")?;
    let source_model_path = safe_package_v2_artifact_path(&package_dir, &metadata_artifact.path)?;
    let source_metadata = source_model_path
        .metadata()
        .with_context(|| format!("stat package-v2 source {}", source_model_path.display()))?;
    anyhow::ensure!(
        source_metadata.is_file(),
        "package-v2 metadata artifact is not a file: {}",
        source_model_path.display()
    );
    anyhow::ensure!(
        source_metadata.len() == metadata_artifact.byte_size,
        "package-v2 metadata artifact size {} differs from manifest {}",
        source_metadata.len(),
        metadata_artifact.byte_size
    );
    let source_sha256 = source_file_sha256(&source_model_path, &source_metadata, digest_cache)?;
    anyhow::ensure!(
        source_sha256 == metadata_artifact.sha256,
        "package-v2 metadata artifact SHA-256 differs from manifest"
    );
    let architecture = manifest
        .model_metadata
        .get("general.architecture")
        .and_then(serde_json::Value::as_str)
        .context("package-v2 model metadata is missing general.architecture")?;
    let activation_width_key = format!("{architecture}.embedding_length");
    let activation_width = manifest
        .model_metadata
        .get(&activation_width_key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .with_context(|| {
            format!("package-v2 model metadata is missing positive {activation_width_key}")
        })?;

    let source_model_bytes = manifest
        .source_model
        .files
        .iter()
        .try_fold(0_u64, |total, file| total.checked_add(file.byte_size))
        .context("package-v2 source byte count overflow")?;
    anyhow::ensure!(
        source_model_bytes > 0,
        "package-v2 source model byte count must be positive"
    );
    let mut source_files = package_v2_source_files(&package_dir, &manifest, digest_cache)?;
    let layer_weight_bytes = package_v2_layer_weight_bytes(&manifest)?;
    let tensor_count = u64::try_from(manifest.tensor_catalog.entries.len())
        .context("package-v2 tensor count exceeds u64")?;
    let manifest_sha256 = hex_lower(&Sha256::digest(&manifest_bytes));
    source_files.push(SkippyPackageSourceFile {
        path: manifest_path,
        bytes: u64::try_from(manifest_bytes.len())
            .context("package-v2 manifest byte count exceeds u64")?,
        sha256: manifest_sha256.clone(),
    });
    let generation = manifest.generation.as_ref().map(package_v2_generation_info);

    Ok(SkippyPackageIdentity {
        package_ref: package_dir.to_string_lossy().into_owned(),
        manifest_sha256,
        source_model_path,
        source_model_sha256: manifest.source_model.sha256,
        source_model_bytes,
        source_files,
        layer_weight_bytes,
        layer_count: manifest.layer_count,
        activation_width,
        tensor_count,
        generation,
    })
}

fn safe_package_v2_artifact_path(package_dir: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    anyhow::ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_))),
        "package-v2 artifact path is not a safe relative path: {relative:?}"
    );
    let resolved = package_dir
        .join(relative)
        .canonicalize()
        .with_context(|| format!("canonicalize package-v2 artifact {relative:?}"))?;
    anyhow::ensure!(
        resolved.starts_with(package_dir),
        "package-v2 artifact escapes its package directory: {relative:?}"
    );
    Ok(resolved)
}

fn package_v2_source_files(
    package_dir: &Path,
    manifest: &PackageManifestV2,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<Vec<SkippyPackageSourceFile>> {
    let referenced_artifact_ids = manifest
        .tensor_catalog
        .entries
        .iter()
        .filter_map(|tensor| match &tensor.storage {
            TensorStorage::Owned { artifact_id, .. } => Some(artifact_id.as_str()),
            TensorStorage::Alias { .. } => None,
        })
        .chain(std::iter::once(
            manifest.source_model.metadata_artifact_id.as_str(),
        ))
        .collect::<std::collections::BTreeSet<_>>();
    let mut artifacts = manifest
        .artifact_catalog
        .entries
        .iter()
        .filter(|artifact| referenced_artifact_ids.contains(artifact.id.as_str()))
        .collect::<Vec<_>>();
    artifacts.sort_by(|left, right| left.id.cmp(&right.id));
    artifacts
        .into_iter()
        .map(|artifact| {
            let path = safe_package_v2_artifact_path(package_dir, &artifact.path)?;
            let source = source_file(&path, digest_cache)?;
            anyhow::ensure!(
                source.bytes == artifact.byte_size,
                "package-v2 artifact {:?} size differs from manifest",
                artifact.id
            );
            anyhow::ensure!(
                source.sha256 == artifact.sha256,
                "package-v2 artifact {:?} SHA-256 differs from manifest",
                artifact.id
            );
            Ok(source)
        })
        .collect()
}

pub fn package_v2_layer_weight_bytes(manifest: &PackageManifestV2) -> Result<Vec<u64>> {
    if !manifest
        .tensor_catalog
        .entries
        .iter()
        .any(|tensor| tensor.layer_ordinal.is_some())
    {
        return Ok(Vec::new());
    }
    let mut layer_bytes = vec![0_u64; manifest.layer_count as usize];
    for tensor in &manifest.tensor_catalog.entries {
        let Some(layer) = tensor.layer_ordinal else {
            continue;
        };
        let TensorStorage::Owned { stored_length, .. } = tensor.storage else {
            continue;
        };
        let slot = layer_bytes
            .get_mut(layer as usize)
            .with_context(|| format!("package-v2 tensor layer {layer} exceeds layer count"))?;
        *slot = slot
            .checked_add(stored_length)
            .context("package-v2 layer byte count overflow")?;
    }
    if layer_bytes.contains(&0) {
        return Ok(Vec::new());
    }
    Ok(layer_bytes)
}

pub fn package_v2_generation_info(
    generation: &skippy_package_format::Generation,
) -> PackageGenerationInfo {
    PackageGenerationInfo {
        speculative_decoding: generation.speculative_decoding.as_ref().map(|speculative| {
            skippy_runtime::package::PackageSpeculativeDecodingInfo {
                default: speculative.default.clone(),
                proposers: speculative
                    .proposers
                    .iter()
                    .map(|(name, proposer)| {
                        let info = match &proposer.kind {
                            ProposerKind::NativeMtp {
                                prediction_depth,
                                layer_indices,
                            } => skippy_runtime::package::PackageSpeculativeProposerInfo {
                                proposer_type: "native-mtp".to_string(),
                                prediction_depth: Some(*prediction_depth),
                                layer_indices: layer_indices.clone(),
                                ngram_min: None,
                                ngram_max: None,
                                max_proposal_tokens: None,
                                history_scope: None,
                            },
                            ProposerKind::NgramCache {
                                ngram_min,
                                ngram_max,
                                max_proposal_tokens,
                                history_scope,
                            }
                            | ProposerKind::NgramSuffix {
                                ngram_min,
                                ngram_max,
                                max_proposal_tokens,
                                history_scope,
                            } => skippy_runtime::package::PackageSpeculativeProposerInfo {
                                proposer_type: match &proposer.kind {
                                    ProposerKind::NgramCache { .. } => "ngram-cache",
                                    ProposerKind::NgramSuffix { .. } => "ngram-suffix",
                                    ProposerKind::NativeMtp { .. } => unreachable!(),
                                }
                                .to_string(),
                                prediction_depth: None,
                                layer_indices: Vec::new(),
                                ngram_min: Some(*ngram_min),
                                ngram_max: Some(*ngram_max),
                                max_proposal_tokens: Some(*max_proposal_tokens),
                                history_scope: Some(history_scope.clone()),
                            },
                        };
                        (name.clone(), info)
                    })
                    .collect(),
                strategies: speculative
                    .strategies
                    .iter()
                    .map(|(name, strategy)| {
                        let info = match &strategy.kind {
                            StrategyKind::NativeMtp {
                                proposer,
                                prediction_depth,
                                layer_indices,
                                window_policy,
                            } => skippy_runtime::package::PackageSpeculativeStrategyInfo {
                                strategy_type: "native-mtp".to_string(),
                                prediction_depth: *prediction_depth,
                                layer_indices: layer_indices.clone(),
                                window_policy: window_policy
                                    .as_ref()
                                    .map(package_v2_window_policy_info),
                                proposer: proposer.clone(),
                                primary: None,
                                extender: None,
                                extension_policy: None,
                            },
                            StrategyKind::NgramCache {
                                proposer,
                                window_policy,
                            }
                            | StrategyKind::NgramSuffix {
                                proposer,
                                window_policy,
                            } => skippy_runtime::package::PackageSpeculativeStrategyInfo {
                                strategy_type: match &strategy.kind {
                                    StrategyKind::NgramCache { .. } => "ngram-cache",
                                    StrategyKind::NgramSuffix { .. } => "ngram-suffix",
                                    StrategyKind::NativeMtp { .. }
                                    | StrategyKind::Composite { .. } => unreachable!(),
                                }
                                .to_string(),
                                prediction_depth: None,
                                layer_indices: Vec::new(),
                                window_policy: window_policy
                                    .as_ref()
                                    .map(package_v2_window_policy_info),
                                proposer: Some(proposer.clone()),
                                primary: None,
                                extender: None,
                                extension_policy: None,
                            },
                            StrategyKind::Composite {
                                primary,
                                extender,
                                extension_policy,
                                window_policy,
                            } => skippy_runtime::package::PackageSpeculativeStrategyInfo {
                                strategy_type: "composite".to_string(),
                                prediction_depth: None,
                                layer_indices: Vec::new(),
                                window_policy: window_policy
                                    .as_ref()
                                    .map(package_v2_window_policy_info),
                                proposer: None,
                                primary: Some(primary.clone()),
                                extender: Some(extender.clone()),
                                extension_policy: Some(
                                    skippy_runtime::package::PackageExtensionPolicyInfo {
                                        max_tokens: extension_policy.max_tokens,
                                    },
                                ),
                            },
                        };
                        (name.clone(), info)
                    })
                    .collect(),
            }
        }),
    }
}

fn package_v2_window_policy_info(
    policy: &skippy_package_format::WindowPolicy,
) -> skippy_runtime::package::PackageWindowPolicyInfo {
    skippy_runtime::package::PackageWindowPolicyInfo {
        default: policy.default.clone(),
        initial_window: policy.initial_window,
        min_window: policy.min_window,
        max_window: policy.max_window,
        pipeline_depth: policy.pipeline_depth,
    }
}

pub fn source_file(
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
pub fn source_file_sha256(
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

pub fn file_sha256(path: &Path) -> Result<String> {
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

impl From<SkippyPackageIdentity> for crate::StageSourceIdentity {
    fn from(identity: SkippyPackageIdentity) -> Self {
        Self {
            package_ref: identity.package_ref,
            manifest_sha256: identity.manifest_sha256,
            source_model_path: identity.source_model_path,
            source_model_sha256: identity.source_model_sha256,
            source_model_bytes: identity.source_model_bytes,
            layer_count: identity.layer_count,
        }
    }
}
