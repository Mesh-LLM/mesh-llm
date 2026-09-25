//! `_artifact` and the family-certification block of `_validate_registry`:
//! immutable download identity, safe relative file paths, positive sizes,
//! SHA-256 digests, and the workload class/profile pairing.

use super::fields::{
    ModelResult, escapes_root, exact_keys, fail, get, has, is_identifier, is_lower_hex, object,
    string,
};
use crate::ci_plan::document::Json;
use crate::repository::python_text::repr;

const CERTIFICATION_KEYS: &[&str] = &[
    "class",
    "architecture",
    "profile",
    "execution",
    "resources",
    "notes",
    "evidence",
    "draft_artifact",
    "mmproj_artifact",
];
const WORKLOAD_CLASSES: [&str; 7] = [
    "causal_generation",
    "embedding",
    "rerank",
    "encoder_decoder",
    "ocr",
    "speech_synthesis",
    "speech_recognition",
];
pub(super) const OPTIONAL_ARTIFACTS: [&str; 2] = ["draft_artifact", "mmproj_artifact"];

/// One immutable Hugging Face download: `owner/repo`, a commit SHA, a
/// selector and one or more integrity-pinned files.
pub(super) fn artifact(value: Option<&Json>, field: &str) -> ModelResult<()> {
    let artifact = object(value, field)?;
    exact_keys(artifact, &["repo", "revision", "selector", "files"], field)?;
    let repo = string(get(artifact, "repo"), &format!("{field}.repo"))?;
    if repo.matches('/').count() != 1 || repo.starts_with('/') || repo.ends_with('/') {
        return fail(format!(
            "{field}.repo must be an owner/repository coordinate"
        ));
    }
    let revision = string(get(artifact, "revision"), &format!("{field}.revision"))?;
    if !is_lower_hex(revision, 40..=64) {
        return fail(format!(
            "{field}.revision must be a lowercase immutable SHA"
        ));
    }
    string(get(artifact, "selector"), &format!("{field}.selector"))?;
    let files = match get(artifact, "files").and_then(Json::as_array) {
        Some(files) if !files.is_empty() => files,
        _ => return fail(format!("{field}.files must be a non-empty array")),
    };
    let mut paths = Vec::with_capacity(files.len());
    for (index, file) in files.iter().enumerate() {
        let record_field = format!("{field}.files[{index}]");
        let file = object(Some(file), &record_field)?;
        exact_keys(file, &["path", "size_bytes", "sha256"], &record_field)?;
        let path = safe_path(file, &record_field)?;
        if paths.contains(&path) {
            return fail(format!("{field}.files contains duplicate path: {path}"));
        }
        paths.push(path);
        integrity(file, &record_field)?;
    }
    Ok(())
}

/// A relative POSIX path that cannot leave the download directory.
fn safe_path<'a>(file: &'a [(String, Json)], field: &str) -> ModelResult<&'a str> {
    let path = string(get(file, "path"), &format!("{field}.path"))?;
    if escapes_root(path) || path.ends_with('/') || path.contains('\\') {
        return fail(format!("{field}.path is unsafe: {}", repr(path)));
    }
    Ok(path)
}

/// A positive integral size and a lowercase SHA-256.
fn integrity(file: &[(String, Json)], field: &str) -> ModelResult<()> {
    if !get(file, "size_bytes")
        .and_then(Json::as_int)
        .is_some_and(|size| size > 0)
    {
        return fail(format!("{field}.size_bytes must be positive"));
    }
    let digest = string(get(file, "sha256"), &format!("{field}.sha256"))?;
    if !is_lower_hex(digest, 64..=64) {
        return fail(format!("{field}.sha256 must be a SHA-256"));
    }
    Ok(())
}

/// The certification block every `llama-family-certification` row needs.
pub(super) fn certification(value: Option<&Json>, row: &str, profiles: &[&str]) -> ModelResult<()> {
    let field = format!("{row}.certification");
    let certification = object(value, &field)?;
    exact_keys(certification, CERTIFICATION_KEYS, &field)?;
    let class = string(get(certification, "class"), &format!("{field}.class"))?;
    if !WORKLOAD_CLASSES.contains(&class) {
        return fail(format!("{field}.class is not a workload class"));
    }
    let architecture = string(
        get(certification, "architecture"),
        &format!("{field}.architecture"),
    )?;
    if !is_identifier(architecture) {
        return fail(format!("{field}.architecture has invalid characters"));
    }
    let profile = string(get(certification, "profile"), &format!("{field}.profile"))?;
    if !profiles.contains(&profile) {
        return fail(format!("{field}.profile is not a family profile"));
    }
    let workload_profile = matches!(profile, "workload-smoke" | "workload-oracle");
    if workload_profile != (class != "causal_generation") {
        return fail(format!("{field} class and profile are incompatible"));
    }
    evidence(certification, &field, profile)?;
    object(
        get(certification, "execution"),
        &format!("{field}.execution"),
    )?;
    object(
        get(certification, "resources"),
        &format!("{field}.resources"),
    )?;
    string(get(certification, "notes"), &format!("{field}.notes"))?;
    for optional in OPTIONAL_ARTIFACTS {
        if has(certification, optional) {
            artifact(get(certification, optional), &format!("{field}.{optional}"))?;
        }
    }
    Ok(())
}

/// `workload-oracle` requires fixture/comparison evidence; others forbid it.
fn evidence(certification: &[(String, Json)], field: &str, profile: &str) -> ModelResult<()> {
    let field = format!("{field}.evidence");
    if profile != "workload-oracle" {
        return match has(certification, "evidence") {
            true => fail(format!("{field} requires workload-oracle")),
            false => Ok(()),
        };
    }
    let evidence = object(get(certification, "evidence"), &field)?;
    exact_keys(evidence, &["fixture", "comparison"], &field)?;
    string(get(evidence, "fixture"), &format!("{field}.fixture"))?;
    string(get(evidence, "comparison"), &format!("{field}.comparison"))?;
    Ok(())
}
