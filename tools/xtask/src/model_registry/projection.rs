//! Projects a validated registry into its generated files: one
//! `test-model-artifacts` manifest per suite plus the family roster, each
//! byte-identical to the legacy generator.

use super::family_roster::{
    Entries, entry, field, files_of, in_suite, integrity, object_of, paths, render, str_field,
};
use super::fields::{ModelResult, fail, get, has};
use super::python_json::{PRETTY, dumps};
use crate::ci_plan::document::Json;

const MANIFEST_DIR: &str = "ci/model-artifacts/manifests";
const FAMILY_MANIFEST: &str = "ci/llama-canary/family-certified.json";
const SUITES: [&str; 11] = [
    "product-smoke",
    "scripted-binary-smoke",
    "sdk-smoke",
    "hf-download-smoke",
    "openai-smoke",
    "skippy-correctness",
    "safetensors-runtime-smoke",
    "skippy-ci-smoke",
    "skippy-parity",
    "competitive-benchmark",
    "radix-cache",
];
const SUITE_DEFAULT_ARTIFACTS: [(&str, &str); 2] = [
    ("product-smoke", "smollm2-q8-inference"),
    ("scripted-binary-smoke", "smollm2-q8-inference"),
];

/// One generated file: repository-relative path and exact bytes.
pub(super) struct Output {
    pub(super) path: String,
    pub(super) text: String,
}

fn text(value: impl Into<String>) -> Json {
    Json::String(value.into())
}

fn suite_row(row: &Entries) -> Json {
    let artifact = object_of(get(row, "artifact"));
    let files = files_of(artifact);
    let (repo, revision) = (str_field(artifact, "repo"), str_field(artifact, "revision"));
    let urls = files
        .iter()
        .map(|file| {
            let path = str_field(file, "path");
            text(format!(
                "https://huggingface.co/{repo}/resolve/{revision}/{path}"
            ))
        })
        .collect::<Vec<_>>();
    let mut result = vec![
        entry("id", field(row, "id")),
        entry("family", field(row, "family")),
    ];
    for key in ["capability_tags", "suites", "cadences"] {
        result.push(entry(key, field(row, key)));
    }
    for key in ["repo", "revision", "selector"] {
        result.push(entry(key, field(artifact, key)));
    }
    result.push(entry("files", paths(&files)));
    result.push(entry("file_integrity", integrity(&files)));
    let model_ref = format!("{repo}:{}", str_field(artifact, "selector"));
    result.push(entry("model_ref", text(model_ref)));
    result.push(entry("urls", Json::Array(urls.clone())));
    match (files.as_slice(), urls.as_slice()) {
        ([file], [url]) => {
            result.push(entry("file", field(file, "path")));
            result.push(entry("size_bytes", field(file, "size_bytes")));
            result.push(entry("sha256", field(file, "sha256")));
            result.push(entry("url", url.clone()));
        }
        _ => result.push(entry("size_bytes", total_size(&files))),
    }
    for key in ["quantizations", "notes"] {
        if has(row, key) {
            result.push(entry(key, field(row, key)));
        }
    }
    Json::Object(result)
}

/// Sum of validated positive sizes; a multi-part model fits in `i64`.
fn total_size(files: &[&Entries]) -> Json {
    let total = files
        .iter()
        .filter_map(|file| get(file, "size_bytes")?.as_int())
        .sum::<i128>();
    Json::Number(i64::try_from(total).unwrap_or(i64::MAX).into())
}

fn suite_manifest(
    suite: &str,
    members: &[&&Entries],
    registry_sha256: &str,
) -> ModelResult<Output> {
    let mut manifest = vec![
        entry("schema_version", Json::Number(1.into())),
        entry("manifest_kind", text("test-model-artifacts")),
        entry("suite", text(suite)),
        entry("registry_sha256", text(registry_sha256)),
        entry(
            "artifacts",
            Json::Array(members.iter().map(|row| suite_row(row)).collect()),
        ),
    ];
    if let Some((_, default)) = SUITE_DEFAULT_ARTIFACTS
        .iter()
        .find(|(name, _)| *name == suite)
    {
        if !members.iter().any(|row| str_field(row, "id") == *default) {
            return fail(format!(
                "suite {suite} default artifact is not registered: {default}"
            ));
        }
        manifest.push(entry("default_artifact_id", text(*default)));
    }
    Ok(Output {
        path: format!("{MANIFEST_DIR}/{suite}.json"),
        text: dumps(&Json::Object(manifest), PRETTY) + "\n",
    })
}

/// Every generated file for `registry`, family roster first.
pub(super) fn outputs(registry: &Entries, registry_sha256: &str) -> ModelResult<Vec<Output>> {
    let rows = get(registry, "artifacts")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .map(|row| object_of(Some(row)))
        .collect::<Vec<_>>();
    let mut outputs = vec![Output {
        path: FAMILY_MANIFEST.to_owned(),
        text: render(registry, &rows)?,
    }];
    for suite in SUITES {
        let members = rows
            .iter()
            .filter(|row| in_suite(row, suite))
            .collect::<Vec<_>>();
        if members.is_empty() {
            return fail(format!("suite {suite} has no registered artifacts"));
        }
        outputs.push(suite_manifest(suite, &members, registry_sha256)?);
    }
    Ok(outputs)
}
