//! Local package-v2 admission materialization. Network acquisition is caller-owned.
use crate::stage_load::ResolvedStagePackage;
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use skippy_package_format::{
    Artifact as PackageV2Artifact, PackageManifest as PackageManifestV2,
    stage_admission::StageAdmissionDescriptor as PackageV2StageAdmissionDescriptor,
};
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub fn ensure_package_manifest_sha(package_ref: &str, expected_sha256: &str) -> Result<()> {
    if expected_sha256.trim().is_empty() {
        return Ok(());
    }
    anyhow::ensure!(
        expected_sha256.len() == 64 && expected_sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
        "package manifest sha256 must be a hex SHA-256 digest"
    );
    let manifest_path = Path::new(package_ref).join("model-package.json");
    let manifest_contents = fs::read(&manifest_path).context("read package manifest")?;
    let actual_sha = hex::encode(Sha256::digest(&manifest_contents));
    anyhow::ensure!(
        actual_sha.eq_ignore_ascii_case(expected_sha256),
        "package manifest sha256 mismatch"
    );
    Ok(())
}

pub fn verify_package_v2_artifact(package_dir: &Path, artifact: &PackageV2Artifact) -> Result<()> {
    let relative = safe_manifest_file_path(&artifact.path)?;
    let path = package_dir.join(&relative);
    let metadata = fs::metadata(&path)
        .with_context(|| format!("stat package-v2 artifact {}", relative.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "package-v2 artifact is not a file: {}",
        relative.display()
    );
    anyhow::ensure!(
        metadata.len() == artifact.byte_size,
        "package-v2 artifact {} size differs from manifest",
        relative.display()
    );
    let mut file = fs::File::open(&path)
        .with_context(|| format!("open package-v2 artifact {}", relative.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash package-v2 artifact {}", relative.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = hex::encode(hasher.finalize());
    anyhow::ensure!(
        actual.eq_ignore_ascii_case(&artifact.sha256),
        "package-v2 artifact {} SHA-256 differs from manifest",
        relative.display()
    );
    Ok(())
}

pub fn safe_manifest_file_path(path: &str) -> Result<PathBuf> {
    anyhow::ensure!(!path.is_empty(), "manifest file path is empty");
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("manifest file path is empty");
    };
    anyhow::ensure!(
        matches!(first, Component::Normal(_))
            && components.all(|component| matches!(component, Component::Normal(_))),
        "manifest file path must be a safe relative path: {}",
        path.display()
    );
    Ok(path.to_path_buf())
}

pub fn resolve_local_package_stage(
    package_dir: &Path,
    admission: &PackageV2StageAdmissionDescriptor,
) -> Result<(String, Vec<PathBuf>, Option<PathBuf>)> {
    let local_ref = package_dir.to_string_lossy().into_owned();
    let manifest_path = package_dir.join("model-package.json");
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?;
    let manifest: PackageManifestV2 =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, package_dir)
            .context("resolve package-v2 metadata carrier")?;
    let resolved = manifest
        .resolve_stage_admission(admission)
        .context("resolve exact package-v2 stage admission")?;
    let required = resolved
        .required_artifacts
        .iter()
        .map(|artifact| (*artifact).clone())
        .collect::<Vec<_>>();

    for artifact in &required {
        verify_package_v2_artifact(package_dir, artifact)?;
    }

    let sidecar_ids = resolved
        .sidecars
        .iter()
        .map(|sidecar| sidecar.artifact.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut model_artifacts = required
        .iter()
        .filter(|artifact| !sidecar_ids.contains(artifact.id.as_str()))
        .collect::<Vec<_>>();
    model_artifacts.sort_by(|left, right| {
        let left_primary = left.id == manifest.source_model.metadata_artifact_id;
        let right_primary = right.id == manifest.source_model.metadata_artifact_id;
        right_primary
            .cmp(&left_primary)
            .then_with(|| left.id.cmp(&right.id))
    });
    let model_parts = model_artifacts
        .into_iter()
        .map(|artifact| package_dir.join(&artifact.path))
        .collect::<Vec<_>>();
    let projector = resolved
        .sidecars
        .iter()
        .find(|sidecar| sidecar.kind == skippy_package_format::SidecarKind::Mmproj)
        .map(|sidecar| package_dir.join(&sidecar.artifact.path));
    Ok((local_ref, model_parts, projector))
}

pub fn package_admission_descriptor(
    admission: &skippy_protocol::StageAdmissionDescriptor,
) -> PackageV2StageAdmissionDescriptor {
    skippy_package_format::stage_admission::StageAdmissionDescriptor {
        package_id: admission.package_id.clone(),
        resident_tensor_ids: admission.resident_tensor_ids.clone(),
        sidecars: admission
            .sidecars
            .iter()
            .map(|sidecar| skippy_package_format::Sidecar {
                kind: match sidecar.kind {
                    skippy_protocol::StageAdmissionSidecarKind::Mmproj => {
                        skippy_package_format::SidecarKind::Mmproj
                    }
                },
                artifact_id: sidecar.artifact_id.clone(),
                name: sidecar.name.clone(),
            })
            .collect(),
    }
}
pub fn resolve_admitted_stage_package(
    package_dir: &Path,
    expected_manifest_sha256: &str,
    admission: &skippy_protocol::StageAdmissionDescriptor,
) -> Result<crate::stage_load::ResolvedStagePackage> {
    let descriptor = package_admission_descriptor(admission);
    let (local_ref, model_part_paths, projector_path) =
        resolve_local_package_stage(package_dir, &descriptor)?;
    stage_package_from_verified_parts(
        local_ref,
        expected_manifest_sha256,
        model_part_paths,
        projector_path,
    )
}

/// Attach source facts to artifact paths already verified by `resolve_local_package_stage`.
/// Acquisition adapters use this after fetching and verifying required artifacts.
pub fn stage_package_from_verified_parts(
    local_ref: String,
    expected_manifest_sha256: &str,
    model_part_paths: Vec<PathBuf>,
    projector_path: Option<PathBuf>,
) -> Result<ResolvedStagePackage> {
    ensure_package_manifest_sha(&local_ref, expected_manifest_sha256)?;
    let manifest_path = Path::new(&local_ref).join("model-package.json");
    let manifest: skippy_package_format::PackageManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
    let source_model_bytes = manifest
        .source_model
        .files
        .iter()
        .try_fold(0_u64, |total, file| total.checked_add(file.byte_size))
        .context("package-v2 source byte count overflow")?;
    let source_model_path = model_part_paths
        .first()
        .context("package-v2 admission selected no model artifacts")?
        .to_string_lossy()
        .into_owned();
    Ok(ResolvedStagePackage {
        local_ref,
        source_model_path,
        source_model_sha256: manifest.source_model.sha256,
        source_model_bytes: Some(source_model_bytes),
        model_part_paths,
        projector_path,
    })
}
