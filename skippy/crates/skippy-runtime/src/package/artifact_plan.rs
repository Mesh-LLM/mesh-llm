//! Declaration-only legacy package stage planning. This never opens artifacts.
use super::{
    PackageArtifact, PackageManifest, PackagePart, load_manifest, safe_relative_manifest_path,
    validate_layer_manifest, validate_manifest_identity,
};
use anyhow::{Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::OpenOptions,
    io::Read as _,
    path::Path,
};

pub const MAX_MANIFEST_BYTES: usize = 8 * 1024 * 1024;

/// Read a bounded regular manifest through its descriptor. Normal cache blob
/// symlinks are allowed, while a FIFO cannot block opening before admission.
pub fn read_manifest_declaration(path: &Path) -> Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_MANIFEST_BYTES as u64,
        "manifest must be a regular file no larger than 8 MiB"
    );
    let mut contents = Vec::new();
    file.take(MAX_MANIFEST_BYTES as u64 + 1)
        .read_to_end(&mut contents)?;
    ensure!(
        contents.len() <= MAX_MANIFEST_BYTES,
        "package manifest exceeds 8 MiB"
    );
    Ok(contents)
}

/// Balanced nonempty half-open range, with remainder layers assigned first.
pub fn even_stage_range(index: u32, count: u32, layers: u32) -> Result<(u32, u32)> {
    ensure!(
        count > 0 && layers > 0 && count <= layers && index < count,
        "stage index/count must select a nonempty stage within layer count"
    );
    let base = layers / count;
    let remainder = layers % count;
    let start = base
        .checked_mul(index)
        .and_then(|value| value.checked_add(index.min(remainder)))
        .ok_or_else(|| anyhow::anyhow!("stage range overflow"))?;
    let end = start
        .checked_add(base + u32::from(index < remainder))
        .ok_or_else(|| anyhow::anyhow!("stage range overflow"))?;
    Ok((start, end))
}

/// Geometry and model identity admitted through the existing package schema.
/// Payload files are not opened during this declaration-only phase.
pub fn declared_geometry(contents: &[u8]) -> Result<(String, u32, u32)> {
    ensure!(
        contents.len() <= MAX_MANIFEST_BYTES,
        "package manifest exceeds 8 MiB"
    );
    let manifest = load_manifest(Path::new("model-package.json"), contents)?;
    validate_manifest_identity(&manifest)?;
    validate_layer_manifest(&manifest)?;
    validate_declared_paths(&manifest)?;
    let width = manifest
        .activation_width
        .filter(|width| *width > 0)
        .ok_or_else(|| anyhow::anyhow!("package manifest has no positive activation_width"))?;
    Ok((manifest.model_id, manifest.layer_count, width))
}

/// Plan declared path/size custody before downloads; this is not byte verification.
pub fn declared_stage_parts(
    contents: &[u8],
    index: u32,
    count: u32,
    start: u32,
    end: u32,
) -> Result<Vec<PackagePart>> {
    ensure!(
        contents.len() <= MAX_MANIFEST_BYTES,
        "package manifest exceeds 8 MiB"
    );
    let manifest = load_manifest(Path::new("model-package.json"), contents)?;
    validate_manifest_identity(&manifest)?;
    let layers = validate_layer_manifest(&manifest)?;
    let (expected_start, expected_end) = even_stage_range(index, count, manifest.layer_count)?;
    ensure!(
        (start, end) == (expected_start, expected_end),
        "stage layer range {start}..{end} does not match even stage range {expected_start}..{expected_end}"
    );
    validate_layer_range(&manifest, &layers, start, end)?;
    validate_declared_paths(&manifest)?;
    selected_declarations(&manifest, index, count, start, end)
}

pub(super) fn validate_layer_range(
    manifest: &PackageManifest,
    layers: &BTreeMap<u32, usize>,
    start: u32,
    end: u32,
) -> Result<()> {
    ensure!(start < end, "stage layer_start must be less than layer_end");
    ensure!(
        end <= manifest.layer_count,
        "stage layer_end {} exceeds package layer_count {}",
        end,
        manifest.layer_count
    );
    for index in start..end {
        ensure!(
            layers.contains_key(&index),
            "package is missing layer {index}"
        );
    }
    Ok(())
}

fn validate_declared_paths(manifest: &PackageManifest) -> Result<()> {
    let mut seen = BTreeSet::new();
    let shared = [
        &manifest.shared.metadata.path,
        &manifest.shared.embeddings.path,
        &manifest.shared.output.path,
    ];
    let all_paths = shared
        .into_iter()
        .chain(manifest.layers.iter().map(|layer| &layer.path))
        .chain(manifest.projectors.iter().map(|projector| &projector.path));
    for path in all_paths {
        // Paths are portable relative artifact names, and each emitted row is
        // exactly one tab-delimited line. No normalization can alias entries.
        ensure!(
            !path
                .chars()
                .any(|ch| ch.is_control() || matches!(ch, '\\' | ':'))
                && path
                    .split('/')
                    .all(|part| !part.is_empty() && part != "." && part != ".."),
            "artifact path must be a portable line-safe relative path"
        );
        safe_relative_manifest_path(path)?;
        ensure!(
            seen.insert(path),
            "package manifest contains duplicate artifact paths"
        );
    }
    Ok(())
}

fn declaration(
    role: &str,
    layer_index: Option<u32>,
    artifact: &PackageArtifact,
) -> Result<PackagePart> {
    Ok(PackagePart {
        role: role.to_owned(),
        layer_index,
        path: safe_relative_manifest_path(&artifact.path)?,
        sha256: artifact.sha256.to_ascii_lowercase(),
        artifact_bytes: artifact.artifact_bytes,
    })
}

fn selected_declarations(
    manifest: &PackageManifest,
    index: u32,
    count: u32,
    start: u32,
    end: u32,
) -> Result<Vec<PackagePart>> {
    let mut parts = vec![declaration("metadata", None, &manifest.shared.metadata)?];
    if index == 0 {
        parts.push(declaration(
            "embeddings",
            None,
            &manifest.shared.embeddings,
        )?);
    }
    if index + 1 == count {
        parts.push(declaration("output", None, &manifest.shared.output)?);
    }
    let layers: BTreeMap<_, _> = manifest
        .layers
        .iter()
        .map(|layer| (layer.layer_index, layer))
        .collect();
    for layer_index in start..end {
        let layer = layers
            .get(&layer_index)
            .ok_or_else(|| anyhow::anyhow!("missing layer {layer_index}"))?;
        parts.push(PackagePart {
            role: "layer".into(),
            layer_index: Some(layer_index),
            path: safe_relative_manifest_path(&layer.path)?,
            sha256: layer.sha256.to_ascii_lowercase(),
            artifact_bytes: layer.artifact_bytes,
        });
    }
    for projector in &manifest.projectors {
        parts.push(PackagePart {
            role: "projector".into(),
            layer_index: None,
            path: safe_relative_manifest_path(&projector.path)?,
            sha256: projector.sha256.to_ascii_lowercase(),
            artifact_bytes: projector.artifact_bytes,
        });
    }
    Ok(parts)
}

#[cfg(test)]
#[path = "artifact_plan_tests.rs"]
mod tests;
