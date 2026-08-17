use super::*;

pub(super) fn collect_artifacts(manifest: &PackageManifest) -> Vec<ArtifactSpec> {
    let mut artifacts = vec![
        artifact_spec("metadata", None, &manifest.shared.metadata),
        artifact_spec("embeddings", None, &manifest.shared.embeddings),
        artifact_spec("output", None, &manifest.shared.output),
    ];
    artifacts.extend(
        manifest
            .layers
            .iter()
            .map(|layer| layer_artifact_spec(layer.layer_index, layer)),
    );
    artifacts.extend(manifest.projectors.iter().map(projector_artifact_spec));
    artifacts
}

fn artifact_spec(
    role: &'static str,
    layer_index: Option<u32>,
    artifact: &PackageArtifact,
) -> ArtifactSpec {
    ArtifactSpec {
        role,
        layer_index,
        path: artifact.path.clone(),
        tensor_count: artifact.tensor_count,
        tensor_bytes: artifact.tensor_bytes,
        artifact_bytes: artifact.artifact_bytes,
        sha256: artifact.sha256.clone(),
    }
}

fn layer_artifact_spec(layer_index: u32, layer: &PackageLayer) -> ArtifactSpec {
    ArtifactSpec {
        role: "layer",
        layer_index: Some(layer_index),
        path: layer.path.clone(),
        tensor_count: layer.tensor_count,
        tensor_bytes: layer.tensor_bytes,
        artifact_bytes: layer.artifact_bytes,
        sha256: layer.sha256.clone(),
    }
}

fn projector_artifact_spec(projector: &PackageProjector) -> ArtifactSpec {
    ArtifactSpec {
        role: "projector",
        layer_index: None,
        path: projector.path.clone(),
        tensor_count: projector.tensor_count,
        tensor_bytes: projector.tensor_bytes,
        artifact_bytes: projector.artifact_bytes,
        sha256: projector.sha256.clone(),
    }
}

pub(super) fn validate_layer_coverage(
    manifest: &PackageManifest,
    report: &mut PackagePreflightReport,
) {
    let mut counts = BTreeMap::<u32, usize>::new();
    for layer in &manifest.layers {
        *counts.entry(layer.layer_index).or_default() += 1;
        if layer.layer_index >= manifest.layer_count {
            report.error(
                "layer_index_out_of_range",
                format!(
                    "package layer index {} exceeds layer_count {}",
                    layer.layer_index, manifest.layer_count
                ),
                Some(layer.path.clone()),
                "rebuild the package so layer indexes are contiguous and in range",
            );
        }
    }
    for layer_index in 0..manifest.layer_count {
        if !counts.contains_key(&layer_index) {
            report.error(
                "missing_layer",
                format!("package manifest is missing layer {layer_index}"),
                Some("model-package.json".to_string()),
                "rebuild the package so every transformer layer has one artifact",
            );
        }
    }
    for (layer_index, count) in counts {
        if count > 1 {
            report.error(
                "duplicate_layer",
                format!("package manifest contains layer {layer_index} {count} times"),
                Some("model-package.json".to_string()),
                "rebuild the package so each layer appears exactly once",
            );
        }
    }
}

pub(super) fn validate_artifacts(
    package: &Path,
    artifacts: &[ArtifactSpec],
    verify_sha256: bool,
    report: &mut PackagePreflightReport,
) {
    for artifact in artifacts {
        report.artifacts.push(preflight_artifact(
            package,
            artifact,
            verify_sha256,
            &mut report.issues,
        ));
    }
}

fn preflight_artifact(
    package: &Path,
    artifact: &ArtifactSpec,
    verify_sha256: bool,
    issues: &mut Vec<PreflightIssue>,
) -> PreflightArtifact {
    let path = match safe_relative_path(&artifact.path) {
        Ok(path) => path,
        Err(message) => {
            push_error(
                issues,
                "unsafe_artifact_path",
                format!(
                    "package {} artifact path is unsafe: {message}",
                    artifact.role
                ),
                Some(artifact.path.clone()),
                "rebuild the package so artifact paths stay inside the package directory",
            );
            return artifact_output(artifact, false, None, None, None);
        }
    };
    validate_artifact_manifest(artifact, issues);
    let absolute = package.join(&path);
    let metadata = match fs::metadata(&absolute) {
        Ok(metadata) if metadata.is_file() => metadata,
        Ok(_) => {
            push_error(
                issues,
                "artifact_not_file",
                format!("package artifact {} is not a file", artifact.path),
                Some(artifact.path.clone()),
                "replace the artifact path with a regular GGUF file",
            );
            return artifact_output(artifact, false, None, None, None);
        }
        Err(error) => {
            push_error(
                issues,
                "missing_artifact",
                format!("package artifact {} is missing: {error}", artifact.path),
                Some(artifact.path.clone()),
                "download or rebuild the package artifact before starting split serving",
            );
            return artifact_output(artifact, false, None, None, None);
        }
    };
    let actual_len = metadata.len();
    let size_matches = actual_len == artifact.artifact_bytes;
    if !size_matches {
        push_error(
            issues,
            "artifact_size_mismatch",
            format!(
                "package artifact {} has {} bytes, manifest expects {}",
                artifact.path, actual_len, artifact.artifact_bytes
            ),
            Some(artifact.path.clone()),
            "redownload or rebuild the package artifact so manifest sizes match",
        );
    }
    let sha_matches = if verify_sha256 {
        Some(validate_artifact_sha(&absolute, artifact, issues))
    } else {
        None
    };
    artifact_output(
        artifact,
        true,
        Some(actual_len),
        Some(size_matches),
        sha_matches,
    )
}

fn validate_artifact_manifest(artifact: &ArtifactSpec, issues: &mut Vec<PreflightIssue>) {
    if artifact.artifact_bytes == 0 {
        push_error(
            issues,
            "empty_artifact",
            format!("package {} artifact declares zero bytes", artifact.role),
            Some(artifact.path.clone()),
            "rebuild the package; split artifacts must be non-empty files",
        );
    }
    if artifact.tensor_count == 0 && artifact.tensor_bytes > 0 {
        push_error(
            issues,
            "invalid_tensor_bytes",
            format!(
                "package {} artifact declares tensor_bytes without tensors",
                artifact.role
            ),
            Some(artifact.path.clone()),
            "rebuild the package manifest so tensor counts and bytes agree",
        );
    }
    if artifact.tensor_count > 0 && artifact.tensor_bytes == 0 {
        push_error(
            issues,
            "invalid_tensor_bytes",
            format!(
                "package {} artifact declares tensors but zero tensor_bytes",
                artifact.role
            ),
            Some(artifact.path.clone()),
            "rebuild the package manifest so tensor counts and bytes agree",
        );
    }
    if !is_sha256(&artifact.sha256) {
        push_error(
            issues,
            "invalid_artifact_sha256",
            format!(
                "package {} artifact sha256 is not a hex digest",
                artifact.role
            ),
            Some(artifact.path.clone()),
            "rebuild the package manifest so artifact checksums are valid",
        );
    }
}

pub(super) fn validate_artifact_sha(
    path: &Path,
    artifact: &ArtifactSpec,
    issues: &mut Vec<PreflightIssue>,
) -> bool {
    match file_sha256(path) {
        Ok(actual) if actual == artifact.sha256.to_ascii_lowercase() => true,
        Ok(actual) => {
            push_error(
                issues,
                "artifact_sha256_mismatch",
                format!(
                    "package artifact {} checksum mismatch: expected {}, got {}",
                    artifact.path, artifact.sha256, actual
                ),
                Some(artifact.path.clone()),
                "redownload or rebuild the package artifact so checksums match",
            );
            false
        }
        Err(error) => {
            push_error(
                issues,
                "artifact_sha256_unreadable",
                format!("cannot hash package artifact {}: {error}", artifact.path),
                Some(artifact.path.clone()),
                "ensure the artifact is readable before enabling checksum verification",
            );
            false
        }
    }
}

fn artifact_output(
    artifact: &ArtifactSpec,
    present: bool,
    actual_artifact_bytes: Option<u64>,
    size_matches_manifest: Option<bool>,
    sha256_matches_manifest: Option<bool>,
) -> PreflightArtifact {
    PreflightArtifact {
        role: artifact.role.to_string(),
        layer_index: artifact.layer_index,
        path: artifact.path.clone(),
        present,
        declared_artifact_bytes: artifact.artifact_bytes,
        actual_artifact_bytes,
        size_matches_manifest,
        sha256_matches_manifest,
    }
}

pub(super) fn build_stage_reports(
    manifest: &PackageManifest,
    stages: Option<usize>,
    report: &mut PackagePreflightReport,
) {
    let Some(stage_count) = stages else {
        return;
    };
    if stage_count == 0 {
        report.error(
            "invalid_stage_count",
            "--stages must be greater than zero",
            Some("model-package.json".to_string()),
            "choose a positive stage count for split preflight",
        );
        return;
    }
    if stage_count as u32 > manifest.layer_count {
        report.error(
            "stage_count_exceeds_layer_count",
            format!(
                "--stages {stage_count} exceeds package layer_count {}",
                manifest.layer_count
            ),
            Some("model-package.json".to_string()),
            "use at most one split stage per transformer layer",
        );
        return;
    }
    let artifact_map = stage_artifacts(report);
    for (stage_index, (layer_start, layer_end)) in
        partition_layers(manifest.layer_count, stage_count)
            .into_iter()
            .enumerate()
    {
        report.stages.push(stage_report(
            stage_index,
            layer_start,
            layer_end,
            stage_count,
            &artifact_map,
        ));
    }
}

fn stage_report(
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    stage_count: usize,
    artifact_map: &BTreeMap<String, StageArtifact>,
) -> PreflightStage {
    let includes_embeddings = stage_index == 0;
    let includes_output = stage_index + 1 == stage_count;
    let mut parts = vec!["metadata".to_string()];
    if includes_embeddings {
        parts.push("embeddings".to_string());
    }
    for layer_index in layer_start..layer_end {
        parts.push(format!("layer:{layer_index}"));
    }
    if includes_output {
        parts.push("output".to_string());
    }
    let artifact_bytes = parts
        .iter()
        .filter_map(|part| artifact_map.get(part))
        .filter(|artifact| artifact.present)
        .map(|artifact| artifact.bytes)
        .sum();
    let missing_parts = parts
        .iter()
        .filter(|part| {
            !artifact_map
                .get(*part)
                .is_some_and(|artifact| artifact.present)
        })
        .cloned()
        .collect::<Vec<_>>();
    PreflightStage {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        part_count: parts.len(),
        artifact_bytes,
        parts,
        missing_parts,
    }
}

#[derive(Clone, Copy)]
struct StageArtifact {
    present: bool,
    bytes: u64,
}

fn stage_artifacts(report: &PackagePreflightReport) -> BTreeMap<String, StageArtifact> {
    report
        .artifacts
        .iter()
        .map(|artifact| {
            (
                stage_part_key(artifact),
                StageArtifact {
                    present: artifact.present,
                    bytes: artifact
                        .actual_artifact_bytes
                        .unwrap_or(artifact.declared_artifact_bytes),
                },
            )
        })
        .collect()
}

fn stage_part_key(artifact: &PreflightArtifact) -> String {
    match (artifact.role.as_str(), artifact.layer_index) {
        ("layer", Some(layer)) => format!("layer:{layer}"),
        (role, _) => role.to_string(),
    }
}

fn partition_layers(layer_count: u32, stages: usize) -> Vec<(u32, u32)> {
    let base = layer_count / stages as u32;
    let extra = layer_count % stages as u32;
    let mut start = 0;
    (0..stages)
        .map(|stage_index| {
            let width = base + u32::from((stage_index as u32) < extra);
            let end = start + width;
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}
