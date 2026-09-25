use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{Context, Result, bail};
use model_artifact::{ModelArtifactFile, ResolvedModelArtifact};
use model_hf::HfModelRepository;
use model_ref::{format_canonical_ref, normalize_gguf_distribution_id, parse_model_ref};

use crate::write::local_artifact_files;

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

pub(crate) fn resolve_package_input(
    model: String,
    explicit: ExplicitSourceIdentity,
) -> Result<PackageInput> {
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
