//! `_validate_registry`: the top-level registry contract and its artifact
//! rows, checked in the legacy order so the first diagnostic is identical.

use super::certification::{artifact, certification};
use super::fields::{
    ModelResult, exact_keys, fail, get, has, is_identifier, object, string, string_list,
};
use crate::ci_plan::document::Json;

const TOP_KEYS: &[&str] = &[
    "schema_version",
    "cadences",
    "suites",
    "family_policy",
    "artifacts",
    "unverified_consumers",
];
const ROW_KEYS: &[&str] = &[
    "id",
    "family",
    "suites",
    "cadences",
    "capability_tags",
    "artifact",
    "certification",
    "quantizations",
    "notes",
];
pub(super) const PROFILES: [&str; 5] = [
    "full",
    "package-oracle",
    "graph-only",
    "workload-smoke",
    "workload-oracle",
];
pub(super) const FAMILY_SUITE: &str = "llama-family-certification";

/// Validates `raw`, returning the registry object entries.
pub(super) fn validate(raw: &Json) -> ModelResult<&[(String, Json)]> {
    let registry = object(Some(raw), "registry")?;
    exact_keys(registry, TOP_KEYS, "registry")?;
    if !get(registry, "schema_version").is_some_and(Json::equals_one) {
        return fail("registry.schema_version must be 1");
    }
    let cadences = string_list(get(registry, "cadences"), "registry.cadences")?;
    let suites = string_list(get(registry, "suites"), "registry.suites")?;
    let profiles = family_profiles(registry)?;
    let rows = match get(registry, "artifacts").and_then(Json::as_array) {
        Some(rows) if !rows.is_empty() => rows,
        _ => return fail("registry.artifacts must be a non-empty array"),
    };
    let declared = Declared {
        cadences: &cadences,
        suites: &suites,
        profiles: &profiles,
    };
    let mut seen = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        validate_row(
            row,
            &format!("registry.artifacts[{index}]"),
            &declared,
            &mut seen,
        )?;
    }
    unverified_consumers(registry)?;
    Ok(registry)
}

/// The names declared at the registry level that rows may reference.
struct Declared<'a> {
    cadences: &'a [&'a str],
    suites: &'a [&'a str],
    profiles: &'a [&'a str],
}

fn family_profiles(registry: &[(String, Json)]) -> ModelResult<Vec<&str>> {
    let policy = object(get(registry, "family_policy"), "registry.family_policy")?;
    exact_keys(policy, &["profiles"], "registry.family_policy")?;
    let profiles = object(get(policy, "profiles"), "registry.family_policy.profiles")?;
    let names = profiles
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    let complete =
        names.len() == PROFILES.len() && PROFILES.iter().all(|name| names.contains(name));
    if !complete {
        return fail("registry.family_policy.profiles must contain the five family profiles");
    }
    for (name, profile) in profiles {
        let field = format!("profile {name}");
        let profile = object(
            Some(profile),
            &format!("registry.family_policy.profiles.{name}"),
        )?;
        exact_keys(profile, &["status", "oracle", "required_lanes"], &field)?;
        string(get(profile, "status"), &format!("{field}.status"))?;
        string(get(profile, "oracle"), &format!("{field}.oracle"))?;
        string_list(
            get(profile, "required_lanes"),
            &format!("{field}.required_lanes"),
        )?;
    }
    Ok(names)
}

fn validate_row<'a>(
    row: &'a Json,
    field: &str,
    declared: &Declared<'_>,
    seen: &mut Vec<&'a str>,
) -> ModelResult<()> {
    let row = object(Some(row), field)?;
    exact_keys(row, ROW_KEYS, field)?;
    let id = string(get(row, "id"), &format!("{field}.id"))?;
    if !is_identifier(id) {
        return fail(format!("{field}.id has invalid characters"));
    }
    if seen.contains(&id) {
        return fail(format!("duplicate artifact id: {id}"));
    }
    seen.push(id);
    string(get(row, "family"), &format!("{field}.family"))?;
    let suites = string_list(get(row, "suites"), &format!("{field}.suites"))?;
    let cadences = string_list(get(row, "cadences"), &format!("{field}.cadences"))?;
    if suites.iter().any(|suite| !declared.suites.contains(suite)) {
        return fail(format!("{field}.suites contains an undeclared suite"));
    }
    if cadences
        .iter()
        .any(|cadence| !declared.cadences.contains(cadence))
    {
        return fail(format!("{field}.cadences contains an undeclared cadence"));
    }
    let tags = string_list(
        get(row, "capability_tags"),
        &format!("{field}.capability_tags"),
    )?;
    if tags.iter().any(|tag| !is_identifier(tag)) {
        return fail(format!("{field}.capability_tags contains an invalid tag"));
    }
    artifact(get(row, "artifact"), &format!("{field}.artifact"))?;
    if has(row, "quantizations") {
        string_list(get(row, "quantizations"), &format!("{field}.quantizations"))?;
    }
    if has(row, "notes") {
        string(get(row, "notes"), &format!("{field}.notes"))?;
    }
    if suites.contains(&FAMILY_SUITE) {
        certification(get(row, "certification"), field, declared.profiles)?;
    }
    Ok(())
}

fn unverified_consumers(registry: &[(String, Json)]) -> ModelResult<()> {
    let items = match get(registry, "unverified_consumers") {
        None => return Ok(()),
        Some(value) => match value.as_array() {
            Some(items) => items,
            None => return fail("registry.unverified_consumers must be an array"),
        },
    };
    for (index, item) in items.iter().enumerate() {
        let field = format!("registry.unverified_consumers[{index}]");
        let item = object(Some(item), &field)?;
        exact_keys(item, &["path", "reason"], &field)?;
        string(get(item, "path"), &format!("{field}.path"))?;
        string(get(item, "reason"), &format!("{field}.reason"))?;
    }
    Ok(())
}
