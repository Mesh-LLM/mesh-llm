//! Local artifact custody through the existing native source/package owners.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use skippy_runtime::package::{
    PackageIntegrityOptions, PackageStageRequest, inspect_layer_package,
    select_layer_package_parts_with_integrity,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};
fn regular(path: &Path) -> Result<PathBuf> {
    ensure!(path.is_absolute(), "artifact path must be absolute");
    let path = path.canonicalize().context("canonical artifact")?;
    ensure!(
        std::fs::symlink_metadata(&path)?.is_file(),
        "artifact must be canonical regular file"
    );
    Ok(path)
}
fn dimensions(metadata: &BTreeMap<String, Value>, minimum: u64) -> Result<Value> {
    let arch = metadata
        .get("general.architecture")
        .and_then(Value::as_str)
        .context("architecture absent")?;
    let read = |suffix| {
        metadata
            .get(&format!("{arch}.{suffix}"))
            .and_then(Value::as_u64)
            .context("positive model dimension absent")
    };
    let layers = read("block_count")?;
    let width = read("embedding_length")?;
    let context = read("context_length")?;
    ensure!(
        layers > 0 && width > 0 && minimum > 0 && context >= minimum,
        "model dimensions/context refused"
    );
    Ok(
        json!({"architecture":arch,"layer_count":layers,"activation_width":width,"native_context_tokens":context}),
    )
}
fn pins(values: &[String]) -> Result<BTreeMap<String, String>> {
    ensure!(
        (1..=128).contains(&values.len()),
        "bounded nonempty shard pin roster required"
    );
    let mut out = BTreeMap::new();
    for value in values {
        let (name, sha) = value
            .split_once('=')
            .context("pin requires filename=SHA256")?;
        ensure!(
            !name.is_empty()
                && Path::new(name).components().count() == 1
                && !matches!(name, "." | "..")
                && !name.contains(['/', '\\'])
                && sha.len() == 64
                && sha
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid shard pin"
        );
        ensure!(
            out.insert(name.to_owned(), sha.to_owned()).is_none(),
            "duplicate shard pin"
        );
    }
    Ok(out)
}
pub(crate) fn source(model: &Path, expected: &[String], minimum: u64) -> Result<Value> {
    ensure!(model.is_absolute(), "source primary must be absolute");
    let model = model
        .parent()
        .context("primary parent")?
        .canonicalize()?
        .join(model.file_name().context("primary filename")?);
    regular(&model)?;
    let primary = model
        .file_name()
        .and_then(|n| n.to_str())
        .context("primary filename")?;
    let inventory = crate::source_inventory::SourceInventory::read_local(&model, primary)?;
    let expected = pins(expected)?;
    ensure!(
        inventory
            .shards
            .first()
            .context("source shard roster empty")?
            .path
            .as_path()
            == model.as_path(),
        "source primary must be the first complete ordered shard"
    );
    let names: BTreeSet<_> = inventory
        .shards
        .iter()
        .map(|s| {
            s.path
                .file_name()
                .and_then(|n| n.to_str())
                .context("shard filename")
        })
        .collect::<Result<_>>()?;
    ensure!(
        names == expected.keys().map(String::as_str).collect(),
        "complete shard pin roster required"
    );
    let mut files = Vec::new();
    for shard in &inventory.shards {
        let canonical = regular(&shard.path)?;
        let name = shard
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .context("canonical shard filename")?;
        ensure!(
            expected.get(name) == Some(&shard.source_file.sha256),
            "shard SHA mismatch"
        );
        files.push(json!({"logical_path":shard.path,"path":canonical,"sha256":shard.source_file.sha256,"byte_size":shard.source_file.byte_size}));
    }
    let shape = dimensions(&inventory.shards[0].directory.metadata, minimum)?;
    ensure!(
        shape["layer_count"] == inventory.layer_count,
        "source layer projection mismatch"
    );
    Ok(
        json!({"schema_version":1,"kind":"gguf-source","primary":model,"ordered_files":files,"dimensions":shape,"scope":"complete_current_native_source_inventory_and_supplied_byte_pins_not_inference"}),
    )
}
pub(crate) fn package(
    root: &Path,
    expected: &str,
    model_id: &str,
    start: u32,
    end: u32,
    minimum: u64,
) -> Result<Value> {
    ensure!(root.is_absolute(), "package must be absolute local path");
    let root = root.canonicalize()?;
    ensure!(root.is_dir(), "package root must be directory");
    let info = inspect_layer_package(root.to_str().context("package UTF8")?)?;
    ensure!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            && info.manifest_sha256 == expected,
        "package manifest pin mismatch"
    );
    ensure!(
        info.model_id == model_id && start < end && end <= info.layer_count,
        "package model/range mismatch"
    );
    let mut files = BTreeMap::new();
    for (label, a, b, source) in [
        ("requested", start, end, start == 0),
        ("tokenizer", 0, 1, true),
    ] {
        let selection = select_layer_package_parts_with_integrity(
            &PackageStageRequest {
                model_id: model_id.into(),
                topology_id: "cache-family-artifact-admission".into(),
                package_ref: root.to_string_lossy().into_owned(),
                stage_id: label.into(),
                layer_start: a,
                layer_end: b,
                source_stage: source,
                terminal_stage: b == info.layer_count,
            },
            &PackageIntegrityOptions::verify_without_cache(),
        )?;
        ensure!(
            selection.manifest_sha256 == expected
                && selection.integrity.artifacts == selection.integrity.verified_artifacts,
            "package selected bytes unverified"
        );
        for part in selection.selected_parts {
            let path = regular(&root.join(&part.path))?;
            let value = json!({"logical_path":root.join(&part.path),"path":path,"role":part.role,"layer_index":part.layer_index,"sha256":part.sha256,"byte_size":part.artifact_bytes});
            if let Some(old) = files.insert(path.clone(), value.clone()) {
                ensure!(old == value, "conflicting package part projection");
            }
        }
    }
    let metadata = files
        .values()
        .find(|v| v["role"] == "metadata")
        .and_then(|v| v["path"].as_str())
        .context("selected package metadata absent")?;
    let catalog = skippy_model::gguf_catalog::read_gguf_catalog(Path::new(metadata))?;
    let shape = dimensions(&catalog.metadata, minimum)?;
    ensure!(
        shape["layer_count"] == info.layer_count,
        "package native dimensions mismatch"
    );
    Ok(
        json!({"schema_version":1,"kind":"layer-package","root":root,"manifest_sha256":expected,"model_id":info.model_id,"dimensions":shape,"state_layer_start":start,"state_layer_end":end,"selected_files":files.into_values().collect::<Vec<_>>(),"source_model_sha256_declared":info.source_model_sha256,"independent_full_source_verified":false,"baseline":"n/a-package-only","scope":"native_selected_stage_and_tokenizer_parts_verified_without_independent_full_source_or_inference"}),
    )
}

#[cfg(test)]
#[path = "artifact_admission/tests.rs"]
mod tests;
