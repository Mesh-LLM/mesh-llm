//! `registry.derived_suites`: suites whose members are declared as a pinned
//! evals model list (the kv-validation four) times a selector list, so the
//! registry states which rows must exist and `generate` refuses a registry
//! that lacks one. `models pin` registers exactly the missing rows.
//!
//! Each list entry at each selector derives one artifact: the file name is
//! the list's with its selector suffix swapped (`Qwen3.5-4B-Q4_K_M.gguf` at
//! `BF16` is `Qwen3.5-4B-BF16.gguf`), the id is `<id_prefix>-<key>` with the
//! selector slug swapped (`perf-qwen3.5-4b-bf16`), the family is the list's
//! and the tags are its parts plus `gguf`. The list's digest is enforced when
//! the selector is the list's own.

use super::fields::{ModelError, ModelResult, fail, get, is_lower_hex};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use std::path::Path;

/// One artifact a derived suite requires.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub(super) suites: Vec<String>,
    pub(super) id: String,
    pub(super) family: String,
    pub(super) tags: Vec<String>,
    pub(super) repo: String,
    pub(super) revision: String,
    pub(super) file: String,
    pub(super) selector: String,
    pub(super) expected: Option<String>,
    pub(super) notes: Option<String>,
}

/// Every artifact the registry's derived suites require, in declaration
/// order; the lists are read under `root`.
pub(super) fn expected(root: &Path, registry: &[(String, Json)]) -> ModelResult<Vec<Entry>> {
    let declared = get(registry, "derived_suites")
        .and_then(Json::as_array)
        .unwrap_or_default();
    declared
        .iter()
        .filter_map(Json::as_object)
        .map(|derived| {
            let text = |key: &str| get(derived, key).and_then(Json::as_str).unwrap_or_default();
            let selectors = get(derived, "selectors")
                .and_then(Json::as_array)
                .unwrap_or_default()
                .iter()
                .filter_map(Json::as_str)
                .collect::<Vec<_>>();
            let list = root.join(text("from"));
            let models = read_list(&list)?;
            selectors
                .iter()
                .map(|selector| {
                    entries(
                        &models,
                        &python_path_display(&list),
                        text("suite"),
                        selector,
                        text("id_prefix"),
                    )
                })
                .collect::<ModelResult<Vec<_>>>()
                .map(|per_selector| per_selector.into_iter().flatten().collect::<Vec<_>>())
        })
        .collect::<ModelResult<Vec<_>>>()
        .map(|per_suite| per_suite.into_iter().flatten().collect())
}

/// The required artifacts the registry does not hold yet. One it holds under
/// the same id must match the derivation and belong to the suite.
pub(super) fn missing(root: &Path, registry: &[(String, Json)]) -> ModelResult<Vec<Entry>> {
    let rows = get(registry, "artifacts")
        .and_then(Json::as_array)
        .unwrap_or_default();
    expected(root, registry)?
        .into_iter()
        .filter_map(|entry| {
            let row = rows
                .iter()
                .filter_map(Json::as_object)
                .find(|row| get(row, "id").and_then(Json::as_str) == Some(entry.id.as_str()));
            match row {
                None => Some(Ok(entry)),
                Some(row) => matches(row, &entry).err().map(Err),
            }
        })
        .collect()
}

fn matches(row: &[(String, Json)], entry: &Entry) -> ModelResult<()> {
    let artifact = get(row, "artifact")
        .and_then(Json::as_object)
        .unwrap_or_default();
    let text = |key: &str| {
        get(artifact, key)
            .and_then(Json::as_str)
            .unwrap_or_default()
    };
    let file = get(artifact, "files")
        .and_then(Json::as_array)
        .and_then(|files| files.first())
        .and_then(|file| file.get("path"))
        .and_then(Json::as_str)
        .unwrap_or_default();
    let declared = get(row, "suites")
        .and_then(Json::as_array)
        .unwrap_or_default();
    let outside = entry
        .suites
        .iter()
        .find(|suite| !declared.iter().any(|item| item.as_str() == Some(suite)));
    let same = text("repo") == entry.repo
        && text("revision") == entry.revision
        && text("selector") == entry.selector
        && file == entry.file;
    if !same {
        return fail(format!(
            "artifact {} is registered as something other than {}@{} {} ({})",
            entry.id, entry.repo, entry.revision, entry.file, entry.selector
        ));
    }
    if let Some(suite) = outside {
        return fail(format!("artifact {} is not in suite {suite}", entry.id));
    }
    Ok(())
}

/// `Q4_K_M` as it appears in ids: `q4-k-m`.
fn slug(selector: &str) -> String {
    selector.to_ascii_lowercase().replace('_', "-")
}

/// The selector suffix of a GGUF file name: `Qwen3.5-4B-Q4_K_M.gguf` carries
/// `Q4_K_M`.
fn selector_of(file: &str) -> Option<(&str, &str)> {
    let stem = file.strip_suffix(".gguf")?;
    let (base, selector) = stem.rsplit_once('-')?;
    (!base.is_empty() && !selector.is_empty()).then_some((base, selector))
}

fn read_list(list: &Path) -> ModelResult<Vec<Json>> {
    let shown = python_path_display(list);
    let bytes = std::fs::read(list).map_err(|error| ModelError(os_error_text(&error, &shown)))?;
    let models = Json::parse(&bytes).map_err(|error| ModelError(format!("{shown}: {error}")))?;
    match models.as_array() {
        Some(models) if !models.is_empty() => Ok(models.to_vec()),
        _ => fail(format!("{shown} must be a non-empty array of models")),
    }
}

/// Every model of a list, re-targeted at `selector` for `suite`.
fn entries(
    models: &[Json],
    shown: &str,
    suite: &str,
    selector: &str,
    prefix: &str,
) -> ModelResult<Vec<Entry>> {
    models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let field = format!("{shown}[{index}]");
            let model = model
                .as_object()
                .ok_or_else(|| ModelError(format!("{field} must be an object")))?;
            let text = |key: &str| {
                get(model, key)
                    .and_then(Json::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| ModelError(format!("{field}.{key} must be a non-empty string")))
            };
            let key = text("key")?;
            let family = text("family")?;
            let revision = text("revision")?;
            if !is_lower_hex(revision, 40..=40) {
                return fail(format!("{field}.revision must be a commit: {revision}"));
            }
            let files = get(model, "files")
                .and_then(Json::as_array)
                .unwrap_or_default();
            let [file] = files else {
                return fail(format!("{field}.files must hold exactly one file"));
            };
            let name = file
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| ModelError(format!("{field}.files[0].name must be a string")))?;
            let sha256 = file
                .get("sha256")
                .and_then(Json::as_str)
                .filter(|sha| is_lower_hex(sha, 64..=64))
                .ok_or_else(|| ModelError(format!("{field}.files[0].sha256 must be 64 hex")))?;
            let Some((base, own)) = selector_of(name) else {
                return fail(format!(
                    "{field}: cannot read a selector suffix from {name}"
                ));
            };
            let stem = key.strip_suffix(&format!("-{}", slug(own))).unwrap_or(key);
            let mut tags = family
                .split('-')
                .chain(["gguf"])
                .map(str::to_owned)
                .collect::<Vec<_>>();
            tags.sort();
            tags.dedup();
            Ok(Entry {
                suites: vec![suite.to_owned()],
                id: format!("{prefix}-{stem}-{}", slug(selector)),
                family: family.to_owned(),
                tags,
                repo: text("repo")?.to_owned(),
                revision: revision.to_owned(),
                file: format!("{base}-{selector}.gguf"),
                selector: selector.to_owned(),
                expected: (own == selector).then(|| sha256.to_owned()),
                notes: Some(format!("Derived from {shown}#{key} at {selector}.")),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const KV_VALIDATION: &str = "skippy/evals/kv-validation/models.json";

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn with(selectors: &[&str], rows: &str) -> Json {
        let selectors = selectors
            .iter()
            .map(|selector| format!("\"{selector}\""))
            .collect::<Vec<_>>()
            .join(",");
        Json::parse(
            format!(
                r#"{{"suites":["performance-parity"],"derived_suites":[{{"suite":"performance-parity","from":"{KV_VALIDATION}","selectors":[{selectors}],"id_prefix":"perf"}}],"artifacts":[{rows}]}}"#
            )
            .as_bytes(),
        )
        .unwrap()
    }

    fn row(id: &str, suite: &str, file: &str, selector: &str) -> String {
        format!(
            r#"{{"id":"{id}","suites":["{suite}"],"artifact":{{"repo":"unsloth/Qwen3-4B-Instruct-2507-GGUF","revision":"a06e946bb6b655725eafa393f4a9745d460374c9","selector":"{selector}","files":[{{"path":"{file}"}}]}}}}"#
        )
    }

    #[test]
    fn the_kv_validation_four_derive_one_entry_per_selector() {
        let registry = with(&["Q4_K_M", "BF16"], "");
        let entries = expected(&root(), registry.as_object().unwrap()).unwrap();
        assert_eq!(entries.len(), 8);
        let qwen3 = &entries[0];
        assert_eq!(qwen3.id, "perf-qwen3-4b-q4-k-m");
        assert_eq!(qwen3.file, "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        assert_eq!(qwen3.family, "dense");
        assert_eq!(qwen3.tags, ["dense", "gguf"]);
        assert!(
            qwen3
                .expected
                .as_deref()
                .is_some_and(|sha| sha.starts_with("3605803b"))
        );
        let hybrid = &entries[1];
        assert_eq!(hybrid.id, "perf-qwen3.5-4b-q4-k-m");
        assert_eq!(hybrid.tags, ["dense", "gguf", "hybrid"]);
        let moe_bf16 = &entries[6];
        assert_eq!(moe_bf16.id, "perf-qwen3-coder-30b-a3b-bf16");
        assert_eq!(moe_bf16.file, "Qwen3-Coder-30B-A3B-Instruct-BF16.gguf");
        assert_eq!(moe_bf16.expected, None);
        assert_eq!(moe_bf16.suites, ["performance-parity"]);
        assert_eq!(slug("Q4_K_M"), "q4-k-m");
        assert_eq!(selector_of("x.gguf"), None);
    }

    #[test]
    fn a_registered_matching_row_is_not_missing() {
        let rows = row(
            "perf-qwen3-4b-q8-0",
            "performance-parity",
            "Qwen3-4B-Instruct-2507-Q8_0.gguf",
            "Q8_0",
        );
        let registry = with(&["Q8_0"], &rows);
        let missing = missing(&root(), registry.as_object().unwrap()).unwrap();
        assert_eq!(
            missing
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            [
                "perf-qwen3.5-4b-q8-0",
                "perf-qwen3-coder-30b-a3b-q8-0",
                "perf-qwen3.5-35b-a3b-q8-0"
            ]
        );
    }

    #[test]
    fn a_row_registered_differently_or_outside_the_suite_is_an_error() {
        let other_file = row(
            "perf-qwen3-4b-q8-0",
            "performance-parity",
            "Qwen3-4B-Instruct-2507-BF16.gguf",
            "Q8_0",
        );
        let registry = with(&["Q8_0"], &other_file);
        let error = missing(&root(), registry.as_object().unwrap()).unwrap_err();
        assert!(
            error.0.contains("registered as something other than"),
            "{}",
            error.0
        );
        let other_suite = row(
            "perf-qwen3-4b-q8-0",
            "elsewhere",
            "Qwen3-4B-Instruct-2507-Q8_0.gguf",
            "Q8_0",
        );
        let registry = with(&["Q8_0"], &other_suite);
        let error = missing(&root(), registry.as_object().unwrap()).unwrap_err();
        assert!(
            error.0.contains("is not in suite performance-parity"),
            "{}",
            error.0
        );
    }
}
