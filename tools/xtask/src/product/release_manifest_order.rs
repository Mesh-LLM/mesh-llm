//! Discovery and ordering for `product runtime-release-manifest`: the
//! `os.walk` search for `manifest.json` files and
//! `artifacts.sort(key=lambda item: item["id"])`.

use crate::ci_plan::document::Json;
use std::fs;
use std::path::Path;

/// Every `manifest.json` that `os.walk(root)` lists as a file: symlinked
/// directories are neither descended nor counted, unreadable directories
/// are skipped, and a symlink named `manifest.json` counts unless it
/// points at a directory.
pub(super) fn manifest_paths(root: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_owned()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = format!("{}/{name}", dir.trim_end_matches('/'));
            let is_dir = Path::new(&path).is_dir();
            let is_link = entry.file_type().is_ok_and(|kind| kind.is_symlink());
            if is_dir && !is_link {
                pending.push(path);
            } else if !is_dir && name == "manifest.json" {
                found.push(path);
            }
        }
    }
    found
}

/// A stable sort by string `id` (code-point order, as Python compares
/// `str`). With fewer than two artifacts Python never compares, so any id
/// type passes; otherwise a non-string id fails with the `TypeError` of
/// Python's first comparison, `ids[1] < ids[0]`.
pub(super) fn sort_by_id(artifacts: &mut [Json]) -> Result<(), String> {
    let id = |item: &Json| item.get("id").cloned().unwrap_or(Json::Null);
    let ids: Vec<Json> = artifacts.iter().map(id).collect();
    if ids.iter().any(|value| !matches!(value, Json::String(_))) {
        return Err("runtime artifact id must be a string".into());
    }
    let text = |value: &Json| value.as_str().unwrap_or("").to_owned();
    artifacts.sort_by_key(|item| text(&id(item)));
    Ok(())
}
