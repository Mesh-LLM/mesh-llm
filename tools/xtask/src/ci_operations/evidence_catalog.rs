//! Catalog-wide evidence checks and the `bind` proposal writer
//! (`validate_catalog`, `require_platforms`, `bind` in
//! `scripts/runner-image-evidence.py`). A proposal is a fresh directory; the
//! input catalog is never adopted or modified.

use crate::ci_operations::evidence_binding::{
    candidate_architectures, provenance, validate_binding,
};
use crate::ci_operations::evidence_input::{decode, evidence_path, fields, read_bytes, require};
use crate::ci_operations::python_access::{Outcome, contains_key, eq, item, object, string};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_json::dumps_indented;
use crate::prepared_input::python_value::display;
use std::path::Path;

type Platforms = Vec<(String, Vec<String>)>;

fn is_symlink(path: &Path) -> bool {
    path.symlink_metadata()
        .is_ok_and(|meta| meta.file_type().is_symlink())
}

fn entries(value: &Json) -> Outcome<&[(String, Json)]> {
    value.as_object().ok_or_else(|| {
        format!(
            "'{}' object has no attribute 'items'",
            crate::ci_operations::python_access::type_name(value)
        )
    })
}

pub(crate) fn validate_catalog(catalog: &Json, root: &Path) -> Outcome<()> {
    let mut platforms: Platforms = Vec::new();
    for (image_id, image) in entries(item(catalog, "images")?)? {
        let (receipt, proof) = (item(image, "receipt")?, item(image, "provenance")?);
        if *receipt == Json::Null && *proof == Json::Null {
            continue;
        }
        require(
            *receipt != Json::Null && *proof != Json::Null,
            "receipt/provenance must be paired",
        )?;
        provenance(proof)?;
        let raw = read_bytes(&evidence_path(root, item(proof, "cohort_sha256")?)?)?;
        let cohort = validate_binding(image, &raw)?;
        platforms.push((image_id.clone(), candidate_architectures(image, &cohort)?));
    }
    require_platforms(catalog, &platforms)
}

fn qualified<'a>(platforms: &'a Platforms, image_id: &Json) -> Option<&'a [String]> {
    let id = image_id.as_str()?;
    platforms
        .iter()
        .find(|(key, _)| key == id)
        .map(|(_, arches)| arches.as_slice())
}

fn has_architecture(arches: &[String], value: &Json) -> bool {
    value
        .as_str()
        .is_some_and(|arch| arches.iter().any(|known| known == arch))
}

pub(crate) fn require_platforms(catalog: &Json, platforms: &Platforms) -> Outcome<()> {
    for (row_id, row) in entries(item(catalog, "runtime_rows")?)? {
        if let Some(arches) = qualified(platforms, item(row, "image_id")?) {
            require(
                has_architecture(arches, item(row, "architecture")?),
                &format!("{row_id}: qualified image lacks runtime row platform"),
            )?;
        }
    }
    let seed = item(catalog, "compiler_seed")?;
    if let Some(arches) = qualified(platforms, item(seed, "image_id")?) {
        require(
            has_architecture(arches, item(seed, "architecture")?),
            "qualified image lacks compiler seed platform",
        )?;
    }
    Ok(())
}

/// A request to write a proposal directory.
pub(crate) struct BindRequest<'a> {
    pub(crate) image_id: &'a str,
    pub(crate) cohort: &'a Path,
    pub(crate) anchor: &'a Path,
    pub(crate) output: &'a Path,
    pub(crate) root: &'a Path,
}

fn replace_image(catalog: &Json, image_id: &str, image: Json) -> Json {
    let mut result = catalog.clone();
    if let Json::Object(top) = &mut result
        && let Some((_, Json::Object(images))) = top.iter_mut().find(|(key, _)| key == "images")
        && let Some((_, slot)) = images.iter_mut().find(|(key, _)| key == image_id)
    {
        *slot = image;
    }
    result
}

/// `dict.update(anchor)`: existing keys keep their position.
fn updated(current: &Json, anchor: &Json) -> Json {
    let mut merged = current.as_object().unwrap_or_default().to_vec();
    for (key, value) in anchor.as_object().unwrap_or_default() {
        match merged.iter_mut().find(|(name, _)| name == key) {
            Some(slot) => slot.1 = value.clone(),
            None => merged.push((key.clone(), value.clone())),
        }
    }
    Json::Object(merged)
}

pub(crate) fn bind(catalog: &Json, request: &BindRequest<'_>) -> Outcome<Json> {
    let images = item(catalog, "images")?;
    require(
        contains_key(images, &string(request.image_id))?,
        "unknown image ID",
    )?;
    let anchor = decode(&read_bytes(request.anchor)?)?;
    fields(&anchor, "receipt provenance")?;
    let current = item(images, request.image_id)?;
    let (receipt, proof) = (item(current, "receipt")?, item(current, "provenance")?);
    require(
        *receipt == Json::Null && *proof == Json::Null
            || eq(receipt, item(&anchor, "receipt")?) && eq(proof, item(&anchor, "provenance")?),
        "immutable binding conflict",
    )?;
    let current = updated(current, &anchor);
    let result = replace_image(catalog, request.image_id, current.clone());
    let raw = read_bytes(request.cohort)?;
    validate_binding(&current, &raw)?;
    let mut evidence: Vec<(Json, Vec<u8>)> = vec![(
        item(item(&current, "provenance")?, "cohort_sha256")?.clone(),
        raw,
    )];
    let mut platforms: Platforms = Vec::new();
    for (qualified_id, image) in entries(item(&result, "images")?)? {
        let proof = item(image, "provenance")?;
        if *proof == Json::Null {
            continue;
        }
        let sha = item(proof, "cohort_sha256")?;
        let known = evidence.iter().position(|(known, _)| eq(known, sha));
        let index = match known {
            Some(index) => index,
            None => {
                let payload = read_bytes(&evidence_path(request.root, sha)?)?;
                evidence.push((sha.clone(), payload));
                evidence.len() - 1
            }
        };
        let cohort = validate_binding(image, &evidence[index].1)?;
        platforms.push((
            qualified_id.clone(),
            candidate_architectures(image, &cohort)?,
        ));
    }
    require_platforms(&result, &platforms)?;
    write_proposal(&result, &evidence, request.output)
}

fn write_proposal(result: &Json, evidence: &[(Json, Vec<u8>)], output: &Path) -> Outcome<Json> {
    require(
        !output.exists() && !is_symlink(output),
        "proposal output must not exist",
    )?;
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    require(
        parent.is_dir() && !is_symlink(parent),
        "invalid proposal parent",
    )?;
    let shown = python_path_display(output);
    let io = |error: std::io::Error, path: &str| os_error_text(&error, path);
    std::fs::create_dir(output).map_err(|error| io(error, &shown))?;
    let directory = output.join("ci/runner-image-evidence");
    std::fs::create_dir_all(&directory)
        .map_err(|error| io(error, &python_path_display(&directory)))?;
    for (sha, payload) in evidence {
        let path = evidence_path(output, sha)?;
        std::fs::write(&path, payload).map_err(|error| io(error, &python_path_display(&path)))?;
    }
    let catalog = output.join("ci/runner-images.json");
    std::fs::write(&catalog, dumps_indented(result) + "\n")
        .map_err(|error| io(error, &python_path_display(&catalog)))?;
    Ok(object(&[
        ("proposal", string(&display(Some(&string(&shown))))),
        ("scope", string("reviewed_producer_admission")),
        ("validation", string("offline_binding_only")),
    ]))
}
