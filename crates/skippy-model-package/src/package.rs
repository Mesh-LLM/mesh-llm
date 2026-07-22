use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use model_artifact::{ModelArtifactFile, ResolvedModelArtifact};
use model_hf::HfModelRepository;
use model_ref::{format_canonical_ref, normalize_gguf_distribution_id, parse_model_ref};
use serde::{Deserialize, Serialize};
use skippy_runtime::{ModelInfo, TensorInfo};

use crate::gguf_header::activation_width;
use crate::hash::file_sha256;
use crate::plan::{layer_count, stage_plan_from_tensors};
use crate::progress::{PackageProgress, format_bytes};
use crate::write::{ModelSource, local_artifact_files, write_json_file, write_stage_artifact};

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageManifest {
    pub(crate) schema_version: u32,
    pub(crate) model_id: String,
    pub(crate) source_model: PackageSourceModel,
    pub(crate) format: String,
    pub(crate) layer_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) activation_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) generation: Option<PackageGeneration>,
    pub(crate) shared: PackageShared,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) projectors: Vec<PackageProjector>,
    pub(crate) layers: Vec<PackageLayer>,
    pub(crate) skippy_abi_version: String,
    pub(crate) created_at_unix_secs: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSourceModel {
    pub(crate) path: String,
    pub(crate) sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) primary_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) canonical_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) distribution_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) files: Vec<ModelArtifactFile>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageShared {
    pub(crate) metadata: PackageArtifact,
    pub(crate) embeddings: PackageArtifact,
    pub(crate) output: PackageArtifact,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageGeneration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) speculative_decoding: Option<PackageSpeculativeDecoding>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSpeculativeDecoding {
    pub(crate) default: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) proposers: BTreeMap<String, PackageSpeculativeProposer>,
    pub(crate) strategies: BTreeMap<String, PackageSpeculativeStrategy>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSpeculativeProposer {
    #[serde(rename = "type")]
    pub(crate) proposer_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prediction_depth: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) layer_indices: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ngram_min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ngram_max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_proposal_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) history_scope: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSpeculativeStrategy {
    #[serde(rename = "type")]
    pub(crate) strategy_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prediction_depth: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) layer_indices: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) window_policy: Option<PackageWindowPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) proposer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) primary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) extender: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) extension_policy: Option<PackageExtensionPolicy>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageExtensionPolicy {
    pub(crate) max_tokens: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageWindowPolicy {
    pub(crate) default: String,
    pub(crate) initial_window: u32,
    pub(crate) min_window: u32,
    pub(crate) max_window: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageLayer {
    pub(crate) layer_index: u32,
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageArtifact {
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageProjector {
    pub(crate) kind: String,
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone)]
pub(crate) struct PackageArtifactSpec {
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
    relative_path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ArtifactHook {
    pub(crate) command: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub(crate) struct ExplicitSourceIdentity {
    pub(crate) model_id: Option<String>,
    pub(crate) source_repo: Option<String>,
    pub(crate) source_revision: Option<String>,
    pub(crate) source_file: Option<String>,
}

#[derive(Debug)]
pub(crate) struct PackageInput {
    pub(crate) model_path: PathBuf,
    pub(crate) model_id: String,
    pub(crate) source_identity: PackageSourceIdentity,
}

#[derive(Debug)]
pub(crate) struct PackageSourceIdentity {
    pub(crate) repo: Option<String>,
    pub(crate) revision: Option<String>,
    pub(crate) primary_file: Option<String>,
    pub(crate) canonical_ref: Option<String>,
    pub(crate) distribution_id: Option<String>,
    pub(crate) files: Vec<ModelArtifactFile>,
}

pub(crate) fn write_package(
    model: String,
    out_dir: PathBuf,
    projectors: Vec<PathBuf>,
    artifact_hook: ArtifactHook,
    artifact_transform: ArtifactHook,
    explicit: ExplicitSourceIdentity,
    resume_existing_artifacts: bool,
) -> Result<()> {
    let input = resolve_package_input(model, explicit)?;
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output directory {}", out_dir.display()))?;
    fs::create_dir_all(out_dir.join("shared"))
        .with_context(|| format!("create shared directory {}", out_dir.display()))?;
    fs::create_dir_all(out_dir.join("layers"))
        .with_context(|| format!("create layers directory {}", out_dir.display()))?;
    if !projectors.is_empty() {
        fs::create_dir_all(out_dir.join("projectors"))
            .with_context(|| format!("create projectors directory {}", out_dir.display()))?;
    }

    let source = ModelSource::open(&input.model_path)?;
    let tensors = &source.tensors;
    let layer_count = layer_count(tensors)?;
    let activation_width = activation_width(&input.model_path)?;
    let source_sha256 = file_sha256(&input.model_path)?;
    let mut progress = PackageProgress::new(3 + layer_count as usize + projectors.len() + 1);

    progress.start_step("shared/metadata.gguf")?;
    let metadata = write_package_artifact(
        &source,
        tensors,
        PackageArtifactSpec {
            stage_index: 0,
            layer_start: 0,
            layer_end: 0,
            includes_embeddings: false,
            includes_output: false,
            relative_path: PathBuf::from("shared/metadata.gguf"),
        },
        &out_dir,
        &artifact_hook,
        &artifact_transform,
        resume_existing_artifacts,
    )?;
    progress.finish_step(&artifact_progress_detail(&metadata))?;
    progress.start_step("shared/embeddings.gguf")?;
    let embeddings = write_package_artifact(
        &source,
        tensors,
        PackageArtifactSpec {
            stage_index: 1,
            layer_start: 0,
            layer_end: 0,
            includes_embeddings: true,
            includes_output: false,
            relative_path: PathBuf::from("shared/embeddings.gguf"),
        },
        &out_dir,
        &artifact_hook,
        &artifact_transform,
        resume_existing_artifacts,
    )?;
    progress.finish_step(&artifact_progress_detail(&embeddings))?;
    progress.start_step("shared/output.gguf")?;
    let output = write_package_artifact(
        &source,
        tensors,
        PackageArtifactSpec {
            stage_index: 2,
            layer_start: layer_count,
            layer_end: layer_count,
            includes_embeddings: false,
            includes_output: true,
            relative_path: PathBuf::from("shared/output.gguf"),
        },
        &out_dir,
        &artifact_hook,
        &artifact_transform,
        resume_existing_artifacts,
    )?;
    progress.finish_step(&artifact_progress_detail(&output))?;

    let mut layers = Vec::new();
    for layer_index in 0..layer_count {
        let relative = PathBuf::from(format!("layers/layer-{layer_index:03}.gguf"));
        progress.start_step(&relative.display().to_string())?;
        let artifact = write_package_artifact(
            &source,
            tensors,
            PackageArtifactSpec {
                stage_index: 1000 + layer_index,
                layer_start: layer_index,
                layer_end: layer_index + 1,
                includes_embeddings: false,
                includes_output: false,
                relative_path: relative,
            },
            &out_dir,
            &artifact_hook,
            &artifact_transform,
            resume_existing_artifacts,
        )?;
        progress.finish_step(&artifact_progress_detail(&artifact))?;
        layers.push(PackageLayer {
            layer_index,
            path: artifact.path,
            tensor_count: artifact.tensor_count,
            tensor_bytes: artifact.tensor_bytes,
            artifact_bytes: artifact.artifact_bytes,
            sha256: artifact.sha256,
        });
    }

    let mut package_projectors = Vec::new();
    for (index, projector) in projectors.iter().enumerate() {
        progress.start_step(&projector.display().to_string())?;
        let package_projector =
            copy_projector_artifact(projector, index, &out_dir, &artifact_hook)?;
        progress.finish_step(&projector_progress_detail(&package_projector))?;
        package_projectors.push(package_projector);
    }

    let manifest = PackageManifest {
        schema_version: 1,
        model_id: input.model_id,
        source_model: PackageSourceModel {
            path: input.model_path.display().to_string(),
            sha256: source_sha256,
            repo: input.source_identity.repo,
            revision: input.source_identity.revision,
            primary_file: input.source_identity.primary_file,
            canonical_ref: input.source_identity.canonical_ref,
            distribution_id: input.source_identity.distribution_id,
            files: input.source_identity.files,
        },
        format: "layer-package".to_string(),
        layer_count,
        activation_width: Some(activation_width),
        generation: package_generation(tensors),
        shared: PackageShared {
            metadata,
            embeddings,
            output,
        },
        projectors: package_projectors,
        layers,
        skippy_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        created_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_secs(),
    };

    let manifest_path = out_dir.join("model-package.json");
    progress.start_step("model-package.json")?;
    write_json_file(&manifest_path, &manifest)?;
    let manifest_bytes = fs::metadata(&manifest_path)
        .with_context(|| format!("read manifest metadata {}", manifest_path.display()))?
        .len();
    progress.finish_step(&format!(
        "model-package.json {}",
        format_bytes(manifest_bytes)
    ))?;
    progress.finish()?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn artifact_progress_detail(artifact: &PackageArtifact) -> String {
    format!(
        "{} {}",
        artifact.path,
        format_bytes(artifact.artifact_bytes)
    )
}

fn projector_progress_detail(projector: &PackageProjector) -> String {
    format!(
        "{} {}",
        projector.path,
        format_bytes(projector.artifact_bytes)
    )
}

fn resolve_package_input(model: String, explicit: ExplicitSourceIdentity) -> Result<PackageInput> {
    let path = PathBuf::from(&model);
    if path.exists() {
        return resolve_local_package_input(path, explicit);
    }

    if explicit.model_id.is_some()
        || explicit.source_repo.is_some()
        || explicit.source_revision.is_some()
        || explicit.source_file.is_some()
    {
        bail!(
            "explicit source identity flags are only valid when write-package input is a local path"
        );
    }

    parse_model_ref(&model).with_context(|| {
        format!(
            "write-package input must be a model coordinate like org/repo:Q4_K_M, not {model:?}"
        )
    })?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build async runtime for Hugging Face model resolution")?;

    runtime.block_on(async {
        let repository = HfModelRepository::from_env()?;
        let artifact = model_artifact::resolve_model_artifact_ref(&model, &repository).await?;
        let paths = repository.download_artifact_files(&artifact).await?;
        let primary_index = artifact
            .files
            .iter()
            .position(|file| file.path == artifact.primary_file)
            .context("resolved artifact file list did not include primary file")?;
        let model_path = paths
            .get(primary_index)
            .cloned()
            .context("downloaded artifact path list did not include primary file")?;
        Ok(package_input_from_resolved_artifact(model_path, artifact))
    })
}

pub(crate) fn resolve_local_package_input(
    model_path: PathBuf,
    explicit: ExplicitSourceIdentity,
) -> Result<PackageInput> {
    let model_id = explicit.model_id.context(
        "local write-package input requires --model-id; prefer passing a coordinate like org/repo:Q4_K_M",
    )?;
    let parsed_model_id = parse_model_ref(&model_id)
        .with_context(|| format!("--model-id must be a model coordinate, got {model_id:?}"))?;
    let cache_identity = if explicit.source_revision.is_none() || explicit.source_file.is_none() {
        HfModelRepository::from_env()
            .ok()
            .and_then(|repository| repository.identity_for_path(&model_path))
    } else {
        None
    };

    let repo = explicit
        .source_repo
        .or_else(|| {
            cache_identity
                .as_ref()
                .map(|identity| identity.repo_id.clone())
        })
        .unwrap_or_else(|| parsed_model_id.repo.clone());
    let revision = explicit
        .source_revision
        .or_else(|| cache_identity.as_ref().map(|identity| identity.revision.clone()))
        .context("local write-package input requires --source-revision for paths outside the Hugging Face cache")?;
    let primary_file = explicit
        .source_file
        .or_else(|| cache_identity.as_ref().map(|identity| identity.file.clone()))
        .context("local write-package input requires --source-file for paths outside the Hugging Face cache")?;
    let canonical_ref = format_canonical_ref(&repo, &revision, &primary_file);
    let distribution_id = normalize_gguf_distribution_id(&primary_file);
    let files = local_artifact_files(&model_path, &primary_file)?;

    Ok(PackageInput {
        model_path,
        model_id: parsed_model_id.display_id(),
        source_identity: PackageSourceIdentity {
            repo: Some(repo),
            revision: Some(revision),
            primary_file: Some(primary_file.clone()),
            canonical_ref: Some(canonical_ref),
            distribution_id,
            files,
        },
    })
}

fn package_input_from_resolved_artifact(
    model_path: PathBuf,
    artifact: ResolvedModelArtifact,
) -> PackageInput {
    PackageInput {
        model_path,
        model_id: artifact.model_id,
        source_identity: PackageSourceIdentity {
            repo: Some(artifact.source_repo),
            revision: Some(artifact.source_revision),
            primary_file: Some(artifact.primary_file),
            canonical_ref: Some(artifact.canonical_ref),
            distribution_id: Some(artifact.distribution_id),
            files: artifact.files,
        },
    }
}

#[cfg(test)]
pub(crate) fn model_distribution_id(model: &Path) -> Option<String> {
    model
        .to_str()
        .and_then(normalize_gguf_distribution_id)
        .or_else(|| {
            model
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(normalize_gguf_distribution_id)
        })
}

pub(crate) fn write_package_artifact(
    source: &ModelSource,
    tensors: &[TensorInfo],
    spec: PackageArtifactSpec,
    out_dir: &Path,
    artifact_hook: &ArtifactHook,
    artifact_transform: &ArtifactHook,
    resume_existing_artifacts: bool,
) -> Result<PackageArtifact> {
    let stage = stage_plan_from_tensors(
        spec.stage_index as usize,
        spec.layer_start,
        spec.layer_end,
        spec.includes_embeddings,
        spec.includes_output,
        tensors,
    );
    let path = out_dir.join(&spec.relative_path);
    if !should_resume_package_artifact(&path, resume_existing_artifacts) {
        write_stage_artifact(source, &stage, &path)?;
    }
    let relative_path = spec.relative_path.display().to_string();
    run_artifact_hook(artifact_transform, &path, &relative_path)?;
    let artifact = read_package_artifact(&path, &spec.relative_path)?;
    run_artifact_hook(artifact_hook, &path, &artifact.path)?;
    Ok(artifact)
}

pub(crate) fn should_resume_package_artifact(path: &Path, resume_existing_artifacts: bool) -> bool {
    resume_existing_artifacts && path.is_file()
}

fn read_package_artifact(path: &Path, relative_path: &Path) -> Result<PackageArtifact> {
    let artifact_info = ModelInfo::open(path)
        .with_context(|| format!("open package artifact {}", path.display()))?;
    let artifact_tensors = artifact_info
        .tensors()
        .with_context(|| format!("read package artifact tensors {}", path.display()))?;
    let metadata =
        fs::metadata(path).with_context(|| format!("read artifact metadata {}", path.display()))?;
    Ok(PackageArtifact {
        path: relative_path.display().to_string(),
        tensor_count: artifact_tensors.len(),
        tensor_bytes: artifact_tensors.iter().map(|tensor| tensor.byte_size).sum(),
        artifact_bytes: metadata.len(),
        sha256: file_sha256(path)?,
    })
}

fn copy_projector_artifact(
    projector: &Path,
    index: usize,
    out_dir: &Path,
    artifact_hook: &ArtifactHook,
) -> Result<PackageProjector> {
    if !projector.is_file() {
        bail!("projector is not a file: {}", projector.display());
    }
    let info = ModelInfo::open(projector)
        .with_context(|| format!("open multimodal projector GGUF {}", projector.display()))?;
    let tensors = info
        .tensors()
        .with_context(|| format!("read multimodal projector tensors {}", projector.display()))?;
    let file_name = projector
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("mmproj-{index:03}.gguf"));
    let relative_path = PathBuf::from("projectors").join(file_name);
    let output_path = out_dir.join(&relative_path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create projector directory {}", parent.display()))?;
    }
    fs::copy(projector, &output_path).with_context(|| {
        format!(
            "copy multimodal projector {} to {}",
            projector.display(),
            output_path.display()
        )
    })?;
    let metadata = fs::metadata(&output_path)
        .with_context(|| format!("read projector metadata {}", output_path.display()))?;

    let package_projector = PackageProjector {
        kind: "mmproj".to_string(),
        path: relative_path.to_string_lossy().replace('\\', "/"),
        tensor_count: tensors.len(),
        tensor_bytes: tensors.iter().map(|tensor| tensor.byte_size).sum(),
        artifact_bytes: metadata.len(),
        sha256: file_sha256(&output_path)?,
    };
    run_artifact_hook(artifact_hook, &output_path, &package_projector.path)?;
    Ok(package_projector)
}

pub(crate) fn run_artifact_hook(
    artifact_hook: &ArtifactHook,
    absolute_path: &Path,
    relative_path: &str,
) -> Result<()> {
    let Some(command) = &artifact_hook.command else {
        return Ok(());
    };
    let status = ProcessCommand::new(command)
        .env("SKIPPY_PACKAGE_ARTIFACT_PATH", absolute_path)
        .env("SKIPPY_PACKAGE_ARTIFACT_RELATIVE_PATH", relative_path)
        .status()
        .with_context(|| format!("run artifact hook {}", command.display()))?;
    if !status.success() {
        bail!(
            "artifact hook {} failed for {} with status {status}",
            command.display(),
            relative_path
        );
    }
    Ok(())
}

pub(crate) fn package_generation(tensors: &[TensorInfo]) -> Option<PackageGeneration> {
    let mtp_layers = native_mtp_layer_indices(tensors);
    if mtp_layers.is_empty() {
        return None;
    }

    let strategy_id = "mtp".to_string();
    let mut strategies = BTreeMap::new();
    let mut proposers = BTreeMap::new();
    proposers.insert(
        strategy_id.clone(),
        PackageSpeculativeProposer {
            proposer_type: "native-mtp".to_string(),
            prediction_depth: Some(1),
            layer_indices: mtp_layers.clone(),
            ngram_min: None,
            ngram_max: None,
            max_proposal_tokens: None,
            history_scope: None,
        },
    );
    strategies.insert(
        strategy_id.clone(),
        PackageSpeculativeStrategy {
            strategy_type: "native-mtp".to_string(),
            prediction_depth: Some(1),
            layer_indices: mtp_layers,
            window_policy: Some(PackageWindowPolicy {
                default: "fixed".to_string(),
                initial_window: 1,
                min_window: 1,
                max_window: 1,
            }),
            proposer: Some(strategy_id.clone()),
            primary: None,
            extender: None,
            extension_policy: None,
        },
    );

    Some(PackageGeneration {
        speculative_decoding: Some(PackageSpeculativeDecoding {
            default: strategy_id,
            proposers,
            strategies,
        }),
    })
}

pub(crate) fn native_mtp_layer_indices(tensors: &[TensorInfo]) -> Vec<u32> {
    tensors
        .iter()
        .filter(|tensor| is_native_mtp_tensor_name(&tensor.name))
        .filter_map(|tensor| tensor.layer_index)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn is_native_mtp_tensor_name(name: &str) -> bool {
    name.contains(".nextn.")
}
