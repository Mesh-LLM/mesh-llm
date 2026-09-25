//! The consumer contract of a generated `test-model-artifacts` manifest:
//! artifact selection, cadence authorization and per-file integrity shape.
//! [`Resolved`] is the only way a consumer sees an artifact.

use super::fields::{ModelResult, escapes_root, fail, get, is_lower_hex};
use crate::ci_plan::document::Json;
use crate::repository::python_text::repr;

/// One integrity-pinned file of a resolved artifact.
pub(super) struct PinnedFile {
    pub(super) name: String,
    pub(super) url: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
}

/// An artifact authorized for the requested cadence.
pub(super) struct Resolved<'a> {
    pub(super) row: &'a [(String, Json)],
    pub(super) id: &'a str,
    pub(super) files: Vec<PinnedFile>,
}

/// What the consumer asked for.
pub(super) struct Selection<'a> {
    pub(super) artifact_id: Option<&'a str>,
    pub(super) cadence: &'a str,
}

fn nonempty(value: Option<&Json>) -> Option<&str> {
    value.and_then(Json::as_str).filter(|text| !text.is_empty())
}

/// `_load`: select, authorize and shape-check one artifact.
pub(super) fn resolve<'a>(
    manifest: &'a Json,
    selection: &Selection<'_>,
) -> ModelResult<Resolved<'a>> {
    if manifest.get("manifest_kind").and_then(Json::as_str) != Some("test-model-artifacts") {
        return fail("manifest_kind must be test-model-artifacts");
    }
    let artifacts = match manifest.get("artifacts").and_then(Json::as_array) {
        Some(artifacts) if !artifacts.is_empty() => artifacts,
        _ => return fail("manifest must contain artifacts"),
    };
    let default = match manifest.get_present("default_artifact_id") {
        None => None,
        Some(value) => match nonempty(Some(value)) {
            Some(id) => Some(id),
            None => return fail("default_artifact_id must be a non-empty string"),
        },
    };
    let row = select(artifacts, selection.artifact_id.or(default))?;
    for key in ["id", "repo", "revision", "selector", "model_ref"] {
        if nonempty(get(row, key)).is_none() {
            return fail(format!("artifact.{key} must be a non-empty string"));
        }
    }
    let id = nonempty(get(row, "id")).unwrap_or_default();
    authorize(row, id, selection.cadence)?;
    Ok(Resolved {
        row,
        id,
        files: pinned_files(row)?,
    })
}

fn select<'a>(artifacts: &'a [Json], wanted: Option<&str>) -> ModelResult<&'a [(String, Json)]> {
    let Some(wanted) = wanted else {
        return match artifacts {
            [only] => only
                .as_object()
                .ok_or_else(|| super::fields::ModelError(MULTIPLE.to_owned())),
            _ => fail(MULTIPLE),
        };
    };
    let matches = artifacts
        .iter()
        .filter_map(Json::as_object)
        .filter(|row| get(row, "id").and_then(Json::as_str) == Some(wanted))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [row] => Ok(row),
        [] => fail(format!("artifact id is absent from manifest: {wanted}")),
        _ => fail(format!("artifact id is duplicated in manifest: {wanted}")),
    }
}

const MULTIPLE: &str = "manifest contains multiple artifacts; pass --artifact-id";

fn authorize(row: &[(String, Json)], id: &str, cadence: &str) -> ModelResult<()> {
    let cadences = match get(row, "cadences").and_then(Json::as_array) {
        Some(items)
            if !items.is_empty() && items.iter().all(|item| nonempty(Some(item)).is_some()) =>
        {
            items
        }
        _ => return fail("artifact.cadences must be a non-empty string list"),
    };
    if cadences.iter().any(|item| item.as_str() == Some(cadence)) {
        return Ok(());
    }
    fail(format!(
        "artifact {id} is not allowed at cadence {}",
        repr(cadence)
    ))
}

fn pinned_files(row: &[(String, Json)]) -> ModelResult<Vec<PinnedFile>> {
    let files = get(row, "files")
        .and_then(Json::as_array)
        .filter(|files| !files.is_empty());
    let integrity = get(row, "file_integrity").and_then(Json::as_object);
    let (Some(files), Some(integrity)) = (files, integrity) else {
        return fail("artifact files and file_integrity are required");
    };
    let urls = match get(row, "urls").and_then(Json::as_array) {
        Some(urls) if urls.len() == files.len() => urls,
        _ => return fail("artifact urls must exactly cover files"),
    };
    files
        .iter()
        .zip(urls)
        .enumerate()
        .map(|(index, (name, url))| pinned_file(index, name, url, integrity))
        .collect()
}

fn pinned_file(
    index: usize,
    name: &Json,
    url: &Json,
    integrity: &[(String, Json)],
) -> ModelResult<PinnedFile> {
    let Some(name) = nonempty(Some(name)) else {
        return fail(format!(
            "artifact.files[{index}] must be a non-empty string"
        ));
    };
    if escapes_root(name) || name.contains('\\') {
        return fail(format!("artifact.files[{index}] is unsafe: {}", repr(name)));
    }
    let Some(record) = get(integrity, name).and_then(Json::as_object) else {
        return fail(format!("artifact.file_integrity is missing {name}"));
    };
    let size = get(record, "size_bytes")
        .and_then(Json::as_int)
        .filter(|size| *size > 0);
    let Some(size_bytes) = size.and_then(|size| u64::try_from(size).ok()) else {
        return fail(format!("artifact size is invalid for {name}"));
    };
    let sha256 = match get(record, "blob_id").and_then(Json::as_str) {
        Some(digest) if is_lower_hex(digest, 64..=64) => digest,
        _ => return fail(format!("artifact SHA-256 is invalid for {name}")),
    };
    let Some(url) = nonempty(Some(url)) else {
        return fail(format!("artifact URL is invalid for {name}"));
    };
    Ok(PinnedFile {
        name: name.to_owned(),
        url: url.to_owned(),
        size_bytes,
        sha256: sha256.to_owned(),
    })
}
