use std::path::{Path, PathBuf};

use crate::inference::skippy::materialization::is_layer_package_ref;

use super::{LayerRange, SourceModelKind, StageInventoryRequest, StageSourceFetch};

#[derive(Clone, Debug)]
pub(super) struct InventorySource {
    pub(super) path: PathBuf,
    pub(super) bytes: Option<u64>,
    pub(super) layer_count: u32,
    pub(super) kind: SourceModelKind,
    pub(super) sha256: Option<String>,
}

pub(super) fn resolve_inventory_source(
    request: &StageInventoryRequest,
    source_fetch: StageSourceFetch,
) -> Option<InventorySource> {
    let local_source_required = crate::inference::skippy::effective_local_source_required(
        &request.model_id,
        request.runtime_profile.as_deref(),
        request.local_source_required,
    );
    if local_source_required
        && !crate::inference::skippy::is_content_addressed_gguf_ref(&request.package_ref)
    {
        return None;
    }
    if crate::inference::skippy::is_content_addressed_gguf_ref(&request.package_ref) {
        let expected_sha256 = request.expected_source_model_sha256.as_deref()?;
        let identity = crate::inference::skippy::verify_registered_content_source(
            &request.model_id,
            &request.package_ref,
            &request.manifest_sha256,
            expected_sha256,
        )
        .map_err(|error| {
            tracing::debug!(
                package_ref = request.package_ref,
                error = %error,
                "content-addressed GGUF inventory verification failed"
            );
            error
        })
        .ok()?;
        let kind = if is_split_gguf_path(&identity.source_model_path) {
            SourceModelKind::SplitGguf
        } else {
            SourceModelKind::PlainGguf
        };
        return Some(InventorySource {
            path: identity.source_model_path,
            bytes: Some(identity.source_model_bytes),
            layer_count: identity.layer_count,
            kind,
            sha256: Some(identity.source_model_sha256),
        });
    }
    if is_layer_package_ref(&request.package_ref) {
        // Inspecting an uncached package downloads its metadata.
        if !source_fetch.allows_fetch()
            && !crate::inference::skippy::package_metadata_is_local(&request.package_ref)
        {
            return None;
        }
        let identity = crate::inference::skippy::identity_from_layer_package(&request.package_ref)
            .map_err(|error| {
                tracing::debug!(
                    package_ref = request.package_ref,
                    error = %error,
                    "package inventory verification failed"
                );
                error
            })
            .ok()?;
        let kind = if is_split_gguf_path(&identity.source_model_path) {
            SourceModelKind::SplitGguf
        } else {
            SourceModelKind::PlainGguf
        };
        return Some(InventorySource {
            path: identity.source_model_path,
            bytes: Some(identity.source_model_bytes),
            layer_count: identity.layer_count,
            kind,
            sha256: Some(identity.source_model_sha256),
        });
    }

    for candidate in inventory_source_candidates(request) {
        if let Some(source) = resolve_direct_gguf_inventory_source(&candidate) {
            return Some(source);
        }
    }
    None
}

fn resolve_direct_gguf_inventory_source(candidate: &Path) -> Option<InventorySource> {
    let source_paths = match crate::inference::skippy::direct_gguf_source_paths(candidate) {
        Ok(paths) => paths,
        Err(error) => {
            tracing::debug!(
                path = %candidate.display(),
                "direct GGUF inventory source is unavailable: {error:#}"
            );
            return None;
        }
    };
    let source_path = source_paths.first()?.clone();
    let layer_count = crate::models::gguf::scan_gguf_compact_meta(&source_path)
        .and_then(|meta| meta.executable_layer_count())
        .or_else(|| crate::inference::skippy::infer_layer_count(&source_path).ok())?;
    let bytes = source_paths
        .iter()
        .filter_map(|path| path.metadata().ok().map(|metadata| metadata.len()))
        .sum();
    let kind = if is_split_gguf_path(&source_path) {
        SourceModelKind::SplitGguf
    } else {
        SourceModelKind::PlainGguf
    };
    Some(InventorySource {
        path: source_path,
        bytes: Some(bytes),
        layer_count,
        kind,
        sha256: None,
    })
}

pub(super) fn inventory_source_candidates(request: &StageInventoryRequest) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = request.package_ref.strip_prefix("gguf://")
        && !path.is_empty()
    {
        candidates.push(PathBuf::from(path));
    }
    if !request.model_id.is_empty() {
        candidates.push(crate::models::find_model_path(&request.model_id));
    }
    candidates
}

fn is_split_gguf_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(skippy_model_ref::split_gguf_shard_info)
        .is_some()
}

/// The layers a peer may load without fetching, when it may not fetch.
///
/// A cached package manifest does not mean the layer files are cached. Under
/// `LocalOnly`, a layer package reports only the layers whose artifacts are
/// on disk, so a coordinator sees the bytes this node would have to fetch and
/// leaves it out of the split. `None` when fetching is allowed or the source
/// is not a layer package: every layer of a resolved source is available.
pub(super) fn local_only_package_layers(
    request: &StageInventoryRequest,
    source_fetch: StageSourceFetch,
) -> Option<Vec<bool>> {
    if source_fetch.allows_fetch() || !is_layer_package_ref(&request.package_ref) {
        return None;
    }
    Some(
        crate::inference::skippy::materialization::package_v2_local_layers(&request.package_ref)
            .unwrap_or_else(|error| {
                tracing::debug!(
                    package_ref = request.package_ref,
                    error = %error,
                    "cannot list locally cached package layers"
                );
                Vec::new()
            }),
    )
}

/// The contiguous runs of layers whose flag equals `wanted`.
pub(super) fn layer_ranges(layers: &[bool], wanted: bool) -> Vec<LayerRange> {
    let mut ranges: Vec<LayerRange> = Vec::new();
    for (index, &flag) in (0_u32..).zip(layers) {
        if flag != wanted {
            continue;
        }
        match ranges.last_mut() {
            Some(range) if range.layer_end == index => range.layer_end = index + 1,
            _ => ranges.push(LayerRange {
                layer_start: index,
                layer_end: index + 1,
            }),
        }
    }
    ranges
}
