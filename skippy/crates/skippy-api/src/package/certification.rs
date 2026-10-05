//! Package materialization and OpenAI smoke certification with explicit caller policy.
use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use skippy_package_format::{
    PackageManifest as PackageManifestV2, TensorStorage, stage_admission::StageAdmissionDescriptor,
};
use skippy_runtime::package::{self, PackageIntegrityOptions, PackageStageRequest};

use super::acquisition::remote::PackageAcquisition;
use super::inspection::{StagePackageInfo, inspect_local_stage_package};
use crate::hash_cache::SidecarDigestCache;

mod runtime_smoke;
use runtime_smoke::runtime_smoke_gates;

#[derive(Clone, Debug)]
pub struct SkippyCertificationRequest {
    pub model_ref: String,
    pub package_only: bool,
    pub api_base: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CertificationGateStatus {
    Passed,
    Failed,
    Incomplete,
    NotRequired,
}

#[derive(Debug, Serialize)]
pub struct SkippyCertificationReport {
    pub schema_version: u32,
    pub status: CertificationGateStatus,
    pub input: String,
    pub resolved_package_ref: String,
    pub local_package_dir: String,
    pub model_id: String,
    pub manifest_sha256: String,
    pub source_model_path: String,
    pub source_model_sha256: String,
    pub source_model_bytes: Option<u64>,
    pub layer_count: u32,
    pub package_gate: CertificationGate,
    pub materialized_stages: Vec<CertifiedStage>,
    pub runtime_gates: Vec<CertificationGate>,
}

#[derive(Debug, Serialize)]
pub struct CertificationGate {
    pub name: String,
    pub status: CertificationGateStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CertifiedStage {
    pub stage_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub selected_part_count: usize,
    pub verified_artifacts: usize,
    pub cached_artifacts: usize,
    pub materialized_path: String,
    pub materialized_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CertificationStageRange {
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
}

pub async fn certify_layer_package(
    request: SkippyCertificationRequest,
    resolved_package_ref: String,
    acquisition: PackageAcquisition,
    digest_cache: Option<SidecarDigestCache>,
) -> Result<SkippyCertificationReport> {
    let local_ref =
        acquisition.resolve_hf_package_to_local(&resolved_package_ref, 0, 0, false, false)?;
    let info =
        inspect_local_stage_package(&resolved_package_ref, &local_ref, digest_cache.as_ref())?;
    let layer_count = info.layer_count;
    let materialized_stages = tokio::task::spawn_blocking({
        let package_ref = resolved_package_ref.clone();
        let model_id = info.model_id.clone();
        move || certify_package_stages(&acquisition, &package_ref, &model_id, layer_count)
    })
    .await
    .map_err(anyhow::Error::from)??;
    let runtime_gates = runtime_smoke_gates(&request, &info).await;
    let package_gate = CertificationGate {
        name: "package_materialization".to_string(),
        status: CertificationGateStatus::Passed,
        details: Some("manifest, selected artifacts, and two local stage ranges verified".into()),
    };
    let status = aggregate_certification_status(
        std::iter::once(package_gate.status).chain(runtime_gates.iter().map(|gate| gate.status)),
    );

    Ok(SkippyCertificationReport {
        schema_version: 1,
        status,
        input: request.model_ref,
        resolved_package_ref,
        local_package_dir: info.package_dir.display().to_string(),
        model_id: info.model_id,
        manifest_sha256: info.manifest_sha256,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
        layer_count: info.layer_count,
        package_gate,
        materialized_stages,
        runtime_gates,
    })
}

fn certify_package_stages(
    acquisition: &PackageAcquisition,
    package_ref: &str,
    model_id: &str,
    layer_count: u32,
) -> Result<Vec<CertifiedStage>> {
    let ranges = certification_stage_ranges(layer_count)?;
    let metadata_ref = acquisition.resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    if super::metadata::is_package_v2_ref(&metadata_ref) {
        return materialize_package_v2_certification_stages(acquisition, package_ref, &ranges);
    }
    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let local_ref = acquisition.resolve_hf_package_to_local(
                package_ref,
                range.layer_start,
                range.layer_end,
                range.include_embeddings,
                range.include_output,
            )?;
            let stage_id = format!("cert-stage-{index}");
            let request = PackageStageRequest {
                model_id: model_id.to_string(),
                topology_id: "skippy-certification".to_string(),
                package_ref: local_ref,
                stage_id: stage_id.clone(),
                layer_start: range.layer_start,
                layer_end: range.layer_end,
                source_stage: range.include_embeddings,
                terminal_stage: range.include_output,
            };
            let integrity_options = match acquisition.integrity_cache.as_deref() {
                Some(path) => PackageIntegrityOptions::verify_with_cache(path),
                None => PackageIntegrityOptions::verify_without_cache(),
            };
            let selected =
                package::select_layer_package_parts_with_integrity(&request, &integrity_options)?;
            let materialized = package::materialize_layer_package_details(&request)?;
            let materialized_bytes = fs::metadata(&materialized.output_path)
                .with_context(|| {
                    format!(
                        "read materialized certification stage {}",
                        materialized.output_path.display()
                    )
                })?
                .len();
            Ok(CertifiedStage {
                stage_id,
                layer_start: range.layer_start,
                layer_end: range.layer_end,
                include_embeddings: range.include_embeddings,
                include_output: range.include_output,
                selected_part_count: materialized.selected_parts.len(),
                verified_artifacts: selected.integrity.verified_artifacts,
                cached_artifacts: selected.integrity.cached_artifacts,
                materialized_path: materialized.output_path.display().to_string(),
                materialized_bytes,
            })
        })
        .collect()
}

fn materialize_package_v2_certification_stages(
    acquisition: &PackageAcquisition,
    package_ref: &str,
    ranges: &[CertificationStageRange],
) -> Result<Vec<CertifiedStage>> {
    let local_ref = acquisition.resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let package_dir = PathBuf::from(&local_ref);
    let manifest_path = package_dir.join("model-package.json");
    let manifest: PackageManifestV2 = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, &package_dir)
            .context("resolve package-v2 metadata carrier")?;

    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let resident_tensor_ids = manifest
                .tensor_catalog
                .entries
                .iter()
                .map(|tensor| -> Result<Option<String>> {
                    let TensorStorage::Owned { artifact_id, .. } = &tensor.storage else {
                        return Ok(None);
                    };
                    let selected = match tensor.layer_ordinal {
                        Some(layer) => layer >= range.layer_start && layer < range.layer_end,
                        None => {
                            let artifact = manifest
                                .artifact_catalog
                                .entries
                                .iter()
                                .find(|artifact| artifact.id == *artifact_id)
                                .with_context(|| {
                                    format!(
                                        "package-v2 tensor {:?} references missing artifact {:?}",
                                        tensor.id, artifact_id
                                    )
                                })?;
                            match artifact.path.as_str() {
                                "shared/common.gguf" => true,
                                "shared/embeddings.gguf" => range.include_embeddings,
                                "shared/output.gguf" => range.include_output,
                                path => bail!(
                                    "package-v2 non-layer tensor {:?} references unsupported artifact path {:?}",
                                    tensor.id,
                                    path
                                ),
                            }
                        }
                    };
                    Ok(selected.then(|| tensor.id.clone()))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            let descriptor = StageAdmissionDescriptor {
                package_id: manifest.package_id.clone(),
                resident_tensor_ids,
                sidecars: if range.include_embeddings {
                    manifest.sidecars.clone()
                } else {
                    Vec::new()
                },
            };
            let (_stage_ref, model_parts, projector_path) =
                acquisition.resolve_package_v2_stage_to_local(package_ref, &descriptor)?;
            let materialized_path = model_parts
                .first()
                .context("package-v2 certification stage has no primary model artifact")?
                .display()
                .to_string();
            let selected_part_count = model_parts.len() + usize::from(projector_path.is_some());
            let materialized_bytes = model_parts.iter().chain(projector_path.iter()).try_fold(
                0_u64,
                |total, path| {
                    let bytes = fs::metadata(path)
                        .with_context(|| format!("stat package-v2 artifact {}", path.display()))?
                        .len();
                    total
                        .checked_add(bytes)
                        .context("package-v2 certification byte count overflow")
                },
            )?;
            Ok(CertifiedStage {
                stage_id: format!("cert-stage-{index}"),
                layer_start: range.layer_start,
                layer_end: range.layer_end,
                include_embeddings: range.include_embeddings,
                include_output: range.include_output,
                selected_part_count,
                verified_artifacts: selected_part_count,
                cached_artifacts: 0,
                materialized_path,
                materialized_bytes,
            })
        })
        .collect()
}

fn certification_stage_ranges(layer_count: u32) -> Result<Vec<CertificationStageRange>> {
    if layer_count < 2 {
        bail!("layer package certification requires at least two transformer layers");
    }
    let split = layer_count / 2;
    Ok(vec![
        CertificationStageRange {
            layer_start: 0,
            layer_end: split,
            include_embeddings: true,
            include_output: false,
        },
        CertificationStageRange {
            layer_start: split,
            layer_end: layer_count,
            include_embeddings: false,
            include_output: true,
        },
    ])
}

fn aggregate_certification_status(
    statuses: impl IntoIterator<Item = CertificationGateStatus>,
) -> CertificationGateStatus {
    let mut saw_incomplete = false;
    for status in statuses {
        match status {
            CertificationGateStatus::Failed => return CertificationGateStatus::Failed,
            CertificationGateStatus::Incomplete => saw_incomplete = true,
            CertificationGateStatus::Passed | CertificationGateStatus::NotRequired => {}
        }
    }
    if saw_incomplete {
        CertificationGateStatus::Incomplete
    } else {
        CertificationGateStatus::Passed
    }
}

#[cfg(test)]
mod tests;
