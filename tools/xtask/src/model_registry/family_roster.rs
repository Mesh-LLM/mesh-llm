//! `ci/llama-canary/family-certified.json`: the family certification roster
//! (`_dump_family(_family_manifest(registry))`), one compact JSON value per
//! field line so reviewers see each model's policy on a few lines.

use super::certification::OPTIONAL_ARTIFACTS;
use super::fields::{ModelResult, fail, get};
use super::python_json::{INLINE, dumps};
use super::registry::FAMILY_SUITE;
use crate::ci_plan::document::Json;

pub(super) type Entries = [(String, Json)];

pub(super) fn entry(key: &str, value: Json) -> (String, Json) {
    (key.to_owned(), value)
}

pub(super) fn field(object: &Entries, key: &str) -> Json {
    get(object, key).cloned().unwrap_or(Json::Null)
}

pub(super) fn str_field<'a>(object: &'a Entries, key: &str) -> &'a str {
    get(object, key).and_then(Json::as_str).unwrap_or_default()
}

pub(super) fn object_of(value: Option<&Json>) -> &Entries {
    value.and_then(Json::as_object).unwrap_or_default()
}

pub(super) fn files_of(artifact: &Entries) -> Vec<&Entries> {
    get(artifact, "files")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .map(|file| object_of(Some(file)))
        .collect()
}

pub(super) fn in_suite(row: &Entries, suite: &str) -> bool {
    get(row, "suites")
        .and_then(Json::as_array)
        .is_some_and(|suites| suites.iter().any(|name| name.as_str() == Some(suite)))
}

pub(super) fn paths(files: &[&Entries]) -> Json {
    Json::Array(files.iter().map(|file| field(file, "path")).collect())
}

/// `{path: {size_bytes, blob_id}}` in file order.
pub(super) fn integrity(files: &[&Entries]) -> Json {
    let records = files.iter().map(|file| {
        let record = vec![
            entry("size_bytes", field(file, "size_bytes")),
            entry("blob_id", field(file, "sha256")),
        ];
        (str_field(file, "path").to_owned(), Json::Object(record))
    });
    Json::Object(records.collect())
}

fn family_artifact(artifact: &Entries) -> Json {
    let files = files_of(artifact);
    Json::Object(vec![
        entry("repo", field(artifact, "repo")),
        entry("revision", field(artifact, "revision")),
        entry("files", paths(&files)),
        entry("file_integrity", integrity(&files)),
        entry("selector", field(artifact, "selector")),
    ])
}

fn compact(value: &Json) -> String {
    dumps(value, INLINE)
}

fn trailing_comma(index: usize, count: usize) -> &'static str {
    if index + 1 < count { "," } else { "" }
}

pub(super) fn render(registry: &Entries, rows: &[&Entries]) -> ModelResult<String> {
    let models = rows
        .iter()
        .filter(|row| in_suite(row, FAMILY_SUITE))
        .collect::<Vec<_>>();
    if models.is_empty() {
        return fail(format!("suite {FAMILY_SUITE} has no registered artifacts"));
    }
    let mut lines = vec![
        "{",
        "  \"schema_version\": 1,",
        "  \"policy\": {",
        "    \"profiles\": {",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    let profiles = object_of(get(object_of(get(registry, "family_policy")), "profiles"));
    for (index, (name, profile)) in profiles.iter().enumerate() {
        let profile = object_of(Some(profile));
        lines.push(format!(
            "      {}: {{",
            compact(&Json::String(name.clone()))
        ));
        for (key, last) in [
            ("status", false),
            ("oracle", false),
            ("required_lanes", true),
        ] {
            let comma = if last { "" } else { "," };
            lines.push(format!(
                "        \"{key}\": {}{comma}",
                compact(&field(profile, key))
            ));
        }
        lines.push(format!("      }}{}", trailing_comma(index, profiles.len())));
    }
    lines.extend(["    }", "  },", "  \"models\": ["].map(str::to_owned));
    for (index, row) in models.iter().enumerate() {
        lines.push("    {".to_owned());
        model_lines(row, &mut lines);
        lines.push(format!("    }}{}", trailing_comma(index, models.len())));
    }
    lines.extend(["  ]", "}"].map(str::to_owned));
    Ok(lines.join("\n") + "\n")
}

fn model_lines(row: &Entries, lines: &mut Vec<String>) {
    let certification = object_of(get(row, "certification"));
    lines.push(format!(
        "      \"family\": {},",
        compact(&field(row, "family"))
    ));
    for key in ["class", "architecture", "profile"] {
        lines.push(format!(
            "      \"{key}\": {},",
            compact(&field(certification, key))
        ));
    }
    let artifact = family_artifact(object_of(get(row, "artifact")));
    lines.push(format!("      \"artifact\": {},", compact(&artifact)));
    for optional in OPTIONAL_ARTIFACTS {
        if let Some(extra) = get(certification, optional) {
            let extra = family_artifact(object_of(Some(extra)));
            lines.push(format!("      \"{optional}\": {},", compact(&extra)));
        }
    }
    if let Some(evidence) = get(certification, "evidence") {
        lines.push(format!("      \"evidence\": {},", compact(evidence)));
    }
    for key in ["execution", "resources"] {
        lines.push(format!(
            "      \"{key}\": {},",
            compact(&field(certification, key))
        ));
    }
    lines.push(format!(
        "      \"notes\": {}",
        compact(&field(certification, "notes"))
    ));
}
