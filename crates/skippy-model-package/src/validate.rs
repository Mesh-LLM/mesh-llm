use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_ffi::TensorRole;
use skippy_runtime::{ModelInfo, TensorInfo};

use crate::gguf_header::activation_width;
use crate::hash::file_sha256;
use crate::package::{PackageArtifact, PackageManifest, PackageProjector};
use crate::plan::layer_count;
use crate::preflight;
use crate::write::ModelSource;

#[derive(Debug, Serialize)]
pub(crate) struct ValidateOutput {
    pub(crate) valid: bool,
    pub(crate) full_tensor_count: usize,
    pub(crate) required_owned_tensor_count: usize,
    pub(crate) missing_owned_tensors: Vec<String>,
    pub(crate) duplicate_owned_tensors: Vec<String>,
    pub(crate) slices: Vec<ValidateSlice>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ValidateSlice {
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) owned_tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) missing_from_full: Vec<String>,
    pub(crate) sha256: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackageValidateOutput {
    pub(crate) valid: bool,
    pub(crate) full_tensor_count: usize,
    pub(crate) layer_count: u32,
    pub(crate) manifest_layer_count_matches_model: bool,
    pub(crate) activation_width_matches_model: bool,
    pub(crate) expected_activation_width: u32,
    pub(crate) manifest_activation_width: Option<u32>,
    pub(crate) source_sha256_matches_manifest: bool,
    pub(crate) required_owned_tensor_count: usize,
    pub(crate) missing_owned_tensors: Vec<String>,
    pub(crate) duplicate_owned_tensors: Vec<String>,
    pub(crate) checked_artifact_count: usize,
    pub(crate) artifacts: Vec<PackageValidateArtifact>,
    pub(crate) checked_projector_count: usize,
    pub(crate) projectors: Vec<PackageValidateProjector>,
    pub(crate) missing_layers: Vec<u32>,
    pub(crate) duplicate_layers: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackageValidateArtifact {
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) owned_tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256_matches_manifest: bool,
    pub(crate) tensor_count_matches_manifest: bool,
    pub(crate) tensor_bytes_matches_manifest: bool,
    pub(crate) artifact_bytes_matches_manifest: bool,
    pub(crate) missing_from_full: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackageValidateProjector {
    pub(crate) path: String,
    pub(crate) kind: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256_matches_manifest: bool,
    pub(crate) tensor_count_matches_manifest: bool,
    pub(crate) tensor_bytes_matches_manifest: bool,
    pub(crate) artifact_bytes_matches_manifest: bool,
}

pub(crate) fn validate(full: PathBuf, slices: Vec<PathBuf>) -> Result<()> {
    if slices.is_empty() {
        bail!("at least one slice path is required");
    }

    let full_source = ModelSource::open(&full)?;
    let full_tensors = full_source.tensors;
    let full_names: BTreeSet<_> = full_tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let required_owned_tensors = full_tensors
        .iter()
        .filter(|tensor| is_owned_slice_tensor(tensor))
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let mut owned_occurrences = BTreeMap::<String, usize>::new();
    let mut output = ValidateOutput {
        valid: true,
        full_tensor_count: full_tensors.len(),
        required_owned_tensor_count: required_owned_tensors.len(),
        missing_owned_tensors: Vec::new(),
        duplicate_owned_tensors: Vec::new(),
        slices: Vec::new(),
    };

    for path in slices {
        let source = ModelSource::open(&path)?;
        let tensors = source.tensors;
        let missing_from_full = tensors
            .iter()
            .filter(|tensor| !full_names.contains(tensor.name.as_str()))
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let owned_tensor_count = tensors
            .iter()
            .filter(|tensor| is_owned_slice_tensor(tensor))
            .inspect(|tensor| {
                *owned_occurrences.entry(tensor.name.clone()).or_default() += 1;
            })
            .count();
        if tensors.is_empty() || !missing_from_full.is_empty() {
            output.valid = false;
        }
        output.slices.push(ValidateSlice {
            path: path.display().to_string(),
            tensor_count: tensors.len(),
            owned_tensor_count,
            tensor_bytes: tensors.iter().map(|tensor| tensor.byte_size).sum(),
            missing_from_full,
            sha256: file_sha256(&path)?,
        });
    }

    output.missing_owned_tensors = required_owned_tensors
        .iter()
        .filter(|name| !owned_occurrences.contains_key(*name))
        .cloned()
        .collect();
    output.duplicate_owned_tensors = owned_occurrences
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect();
    if !output.missing_owned_tensors.is_empty() || !output.duplicate_owned_tensors.is_empty() {
        output.valid = false;
    }

    println!("{}", serde_json::to_string_pretty(&output)?);
    if !output.valid {
        bail!("slice validation failed");
    }
    Ok(())
}

fn is_owned_slice_tensor(tensor: &TensorInfo) -> bool {
    matches!(
        tensor.role,
        TensorRole::Embedding | TensorRole::Layer | TensorRole::FinalNorm | TensorRole::Output
    )
}

pub(crate) fn validate_package(full: PathBuf, package: PathBuf) -> Result<()> {
    let full_source = ModelSource::open(&full)?;
    let full_tensors = full_source.tensors;
    let full_names: BTreeSet<_> = full_tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let required_owned_tensors = full_tensors
        .iter()
        .filter(|tensor| is_owned_slice_tensor(tensor))
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let mut owned_occurrences = BTreeMap::<String, usize>::new();
    let full_layer_count = layer_count(&full_tensors)?;
    let manifest_path = package.join("model-package.json");
    let manifest: PackageManifest = serde_json::from_str(
        &fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse {}", manifest_path.display()))?;
    let source_sha256_matches_manifest = file_sha256(&full)? == manifest.source_model.sha256;
    let manifest_layer_count_matches_model = manifest.layer_count == full_layer_count;
    let expected_activation_width = activation_width(&full)?;
    let manifest_activation_width = manifest.activation_width;
    let activation_width_matches_model =
        manifest_activation_width == Some(expected_activation_width);

    let expected_layers = (0..manifest.layer_count).collect::<BTreeSet<_>>();
    let mut layer_occurrences = BTreeMap::<u32, usize>::new();
    for layer in &manifest.layers {
        *layer_occurrences.entry(layer.layer_index).or_default() += 1;
    }
    let actual_layers = layer_occurrences.keys().copied().collect::<BTreeSet<_>>();
    let missing_layers = expected_layers
        .difference(&actual_layers)
        .copied()
        .collect::<Vec<_>>();
    let duplicate_layers = layer_occurrences
        .into_iter()
        .filter_map(|(layer, count)| (count > 1).then_some(layer))
        .collect::<Vec<_>>();

    let mut artifacts = Vec::new();
    artifacts.push(validate_package_artifact(
        &package,
        &manifest.shared.metadata,
        &full_names,
        &mut owned_occurrences,
    )?);
    artifacts.push(validate_package_artifact(
        &package,
        &manifest.shared.embeddings,
        &full_names,
        &mut owned_occurrences,
    )?);
    artifacts.push(validate_package_artifact(
        &package,
        &manifest.shared.output,
        &full_names,
        &mut owned_occurrences,
    )?);
    for layer in &manifest.layers {
        artifacts.push(validate_package_artifact(
            &package,
            &PackageArtifact {
                path: layer.path.clone(),
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
                sha256: layer.sha256.clone(),
            },
            &full_names,
            &mut owned_occurrences,
        )?);
    }
    let projectors = manifest
        .projectors
        .iter()
        .map(|projector| validate_package_projector(&package, projector))
        .collect::<Result<Vec<_>>>()?;

    let missing_owned_tensors = required_owned_tensors
        .iter()
        .filter(|name| !owned_occurrences.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    let duplicate_owned_tensors = owned_occurrences
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect::<Vec<_>>();
    let valid = source_sha256_matches_manifest
        && manifest_layer_count_matches_model
        && activation_width_matches_model
        && missing_layers.is_empty()
        && duplicate_layers.is_empty()
        && missing_owned_tensors.is_empty()
        && duplicate_owned_tensors.is_empty()
        && artifacts.iter().all(|artifact| {
            artifact.sha256_matches_manifest
                && artifact.tensor_count_matches_manifest
                && artifact.tensor_bytes_matches_manifest
                && artifact.artifact_bytes_matches_manifest
                && artifact.missing_from_full.is_empty()
        })
        && projectors.iter().all(|projector| {
            projector.sha256_matches_manifest
                && projector.tensor_count_matches_manifest
                && projector.tensor_bytes_matches_manifest
                && projector.artifact_bytes_matches_manifest
        });
    let output = PackageValidateOutput {
        valid,
        full_tensor_count: full_tensors.len(),
        layer_count: manifest.layer_count,
        manifest_layer_count_matches_model,
        activation_width_matches_model,
        expected_activation_width,
        manifest_activation_width,
        source_sha256_matches_manifest,
        required_owned_tensor_count: required_owned_tensors.len(),
        missing_owned_tensors,
        duplicate_owned_tensors,
        checked_artifact_count: artifacts.len(),
        artifacts,
        checked_projector_count: projectors.len(),
        projectors,
        missing_layers,
        duplicate_layers,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    if !valid {
        bail!("package validation failed");
    }
    Ok(())
}

pub(crate) fn run_preflight(
    package: PathBuf,
    stages: Option<usize>,
    verify_sha256: bool,
) -> Result<()> {
    let report = preflight::preflight_package(
        &package,
        &preflight::PackagePreflightOptions {
            stages,
            verify_sha256,
        },
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.valid {
        bail!("package preflight failed");
    }
    Ok(())
}

pub(crate) fn validate_package_artifact(
    package: &Path,
    artifact: &PackageArtifact,
    full_names: &BTreeSet<&str>,
    owned_occurrences: &mut BTreeMap<String, usize>,
) -> Result<PackageValidateArtifact> {
    let path = package.join(&artifact.path);
    let info = ModelInfo::open(&path)?;
    let tensors = info.tensors()?;
    let missing_from_full = tensors
        .iter()
        .filter(|tensor| !full_names.contains(tensor.name.as_str()))
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    let owned_tensor_count = tensors
        .iter()
        .filter(|tensor| is_owned_slice_tensor(tensor))
        .inspect(|tensor| {
            *owned_occurrences.entry(tensor.name.clone()).or_default() += 1;
        })
        .count();
    let tensor_bytes = tensors.iter().map(|tensor| tensor.byte_size).sum();
    let artifact_bytes = fs::metadata(&path)
        .with_context(|| format!("read artifact metadata {}", path.display()))?
        .len();
    Ok(PackageValidateArtifact {
        path: artifact.path.clone(),
        tensor_count: tensors.len(),
        owned_tensor_count,
        tensor_bytes,
        artifact_bytes,
        sha256_matches_manifest: file_sha256(&path)? == artifact.sha256,
        tensor_count_matches_manifest: tensors.len() == artifact.tensor_count,
        tensor_bytes_matches_manifest: tensor_bytes == artifact.tensor_bytes,
        artifact_bytes_matches_manifest: artifact_bytes == artifact.artifact_bytes,
        missing_from_full,
    })
}

pub(crate) fn validate_package_projector(
    package: &Path,
    projector: &PackageProjector,
) -> Result<PackageValidateProjector> {
    let path = package.join(&projector.path);
    let info = ModelInfo::open(&path)
        .with_context(|| format!("open package projector {}", path.display()))?;
    let tensors = info
        .tensors()
        .with_context(|| format!("read package projector tensors {}", path.display()))?;
    let tensor_bytes = tensors.iter().map(|tensor| tensor.byte_size).sum();
    let artifact_bytes = fs::metadata(&path)
        .with_context(|| format!("read projector metadata {}", path.display()))?
        .len();
    Ok(PackageValidateProjector {
        path: projector.path.clone(),
        kind: projector.kind.clone(),
        tensor_count: tensors.len(),
        tensor_bytes,
        artifact_bytes,
        sha256_matches_manifest: file_sha256(&path)? == projector.sha256,
        tensor_count_matches_manifest: tensors.len() == projector.tensor_count,
        tensor_bytes_matches_manifest: tensor_bytes == projector.tensor_bytes,
        artifact_bytes_matches_manifest: artifact_bytes == projector.artifact_bytes,
    })
}
