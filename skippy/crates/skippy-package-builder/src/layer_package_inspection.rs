//! Full local legacy package declaration/size inspection, without acquisition or native calls.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use skippy_runtime::package::{
    PackageIntegrityOptions, PackageStageRequest, inspect_layer_package,
    select_layer_package_parts_with_integrity,
};
use std::path::Path;
#[cfg(test)]
#[path = "layer_package_inspection/tests.rs"]
mod tests;

pub(crate) fn inspect(
    package: &Path,
    expected_layers: Option<u32>,
    expected_width: Option<u32>,
) -> Result<Value> {
    let root = package.canonicalize()?;
    ensure!(root.is_dir(), "local layer package must be a directory");
    let reference = root
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("local package path is not UTF-8"))?;
    let info = inspect_layer_package(reference)?;
    let width = info
        .activation_width
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow::anyhow!("legacy package requires a positive activation_width"))?;
    ensure!(
        info.layer_count > 0,
        "legacy package requires a positive layer_count"
    );
    ensure!(
        expected_layers.is_none_or(|value| value == info.layer_count),
        "legacy package layer_count differs from expected value"
    );
    ensure!(
        expected_width.is_none_or(|value| value == width),
        "legacy package activation_width differs from expected value"
    );
    let selection = select_layer_package_parts_with_integrity(
        &PackageStageRequest {
            model_id: info.model_id.clone(),
            topology_id: "wan-local-cache-inspection".into(),
            package_ref: reference.into(),
            stage_id: "complete-local-cache".into(),
            layer_start: 0,
            layer_end: info.layer_count,
            source_stage: true,
            terminal_stage: true,
        },
        &PackageIntegrityOptions::manifest_only(),
    )?;
    ensure!(
        selection.manifest_sha256 == info.manifest_sha256,
        "legacy manifest changed during local inspection"
    );
    Ok(json!({
        "model_id":info.model_id,"layer_count":info.layer_count,"activation_width":width,
        "artifact_count":selection.selected_parts.len()+selection.projector_paths.len(),
        "manifest_sha256":selection.manifest_sha256,
        "scope":"complete_local_legacy_manifest_and_declared_file_sizes",
        "artifact_sha256_verified":false,"independent_source_verified":false,"inference_qualified":false,
    }))
}
