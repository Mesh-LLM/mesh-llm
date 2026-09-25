//! `native-sdk-manifest ARTIFACT_DIR MANIFEST`: the native SDK artifact
//! contract from the inline program in `scripts/verify-native-sdk-package.sh`.
//! Checks run in the legacy order so the first reported violation matches.

use super::sdk_artifact_file::{artifact_file, sha256_file};
use super::sdk_identity_contract::{Manifest, check_identity, reject};
use super::{Checked, Rejected, positional, python_json, python_value};
use crate::ci_plan::document::Json;
use std::path::Path;

const REQUIRED: [&str; 16] = [
    "schema_version",
    "artifact_id",
    "native_runtime_id",
    "sdk_version",
    "mesh_version",
    "target_triple",
    "platform",
    "os",
    "arch",
    "backend",
    "flavor",
    "library",
    "library_paths",
    "library_sha256",
    "requirements",
    "features",
];
const STRINGS: [&str; 12] = [
    "artifact_id",
    "native_runtime_id",
    "sdk_version",
    "mesh_version",
    "target_triple",
    "platform",
    "os",
    "arch",
    "backend",
    "flavor",
    "library",
    "library_sha256",
];
const FEATURES: [&str; 5] = [
    "mesh-inference",
    "model-management",
    "local-serving",
    "chat",
    "responses",
];

pub(super) fn run(args: &[String]) -> Checked<String> {
    let [artifact_dir, manifest_path] = positional(args, "ARTIFACT_DIR MANIFEST")?;
    let root = Path::new(artifact_dir)
        .canonicalize()
        .map_err(|error| Rejected(super::python_io::os_error(Path::new(artifact_dir), &error)))?;
    let value = python_json::load(Path::new(manifest_path))
        .map_err(|error| format!("native SDK manifest is not valid JSON: {error}"))?;
    if value.as_object().is_none() {
        return reject("native SDK manifest must be a JSON object".to_owned());
    }
    let missing: Vec<&str> = REQUIRED
        .into_iter()
        .filter(|key| value.get(key).is_none())
        .collect();
    if !missing.is_empty() {
        return reject(format!("missing manifest field(s): {}", missing.join(", ")));
    }
    let schema = value.get("schema_version");
    if !schema.is_some_and(Json::equals_one) {
        return reject(format!(
            "unsupported schema_version: {}",
            python_value::repr(schema)
        ));
    }
    for field in STRINGS {
        if value
            .get(field)
            .and_then(Json::as_str)
            .is_none_or(str::is_empty)
        {
            return reject(format!("{field} must be a non-empty string"));
        }
    }
    let manifest = Manifest(&value);
    check_identity(&manifest, artifact_dir)?;
    check_files(&manifest, &root)?;
    check_features(&manifest)?;
    check_library_kind(&manifest)?;
    Ok(String::new())
}

fn check_files(manifest: &Manifest<'_>, root: &Path) -> Checked<()> {
    let library = manifest.text("library");
    let paths = match manifest.0.get("library_paths").and_then(Json::as_array) {
        Some(paths) if !paths.is_empty() => paths,
        _ => return reject("library_paths must be a non-empty list".to_owned()),
    };
    if !paths.iter().any(|path| path.as_str() == Some(library)) {
        return reject("library_paths must include the primary library".to_owned());
    }
    if manifest
        .0
        .get("requirements")
        .and_then(Json::as_array)
        .is_none()
    {
        return reject("requirements must be a list".to_owned());
    }
    for path in paths {
        artifact_file(root, "library_paths entry", Some(path))?;
    }
    let library_path = artifact_file(root, "library", manifest.0.get("library"))?;
    let actual = sha256_file(&library_path)?;
    let recorded = manifest.text("library_sha256");
    if actual != recorded {
        return reject(format!(
            "library_sha256 mismatch for {library}: {actual} != {recorded}"
        ));
    }
    let uniffi = manifest.0.get("uniffi_library");
    if uniffi.is_some_and(truthy) {
        let uniffi_actual = sha256_file(&artifact_file(root, "uniffi_library", uniffi)?)?;
        if uniffi_actual != actual {
            return reject(format!(
                "uniffi_library checksum mismatch: {uniffi_actual} != {actual}"
            ));
        }
    }
    Ok(())
}

/// Python truthiness of a JSON value.
fn truthy(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(flag) => *flag,
        Json::Number(number) => number.as_f64() != Some(0.0),
        Json::String(text) => !text.is_empty(),
        Json::Array(items) => !items.is_empty(),
        Json::Object(entries) => !entries.is_empty(),
    }
}

fn check_features(manifest: &Manifest<'_>) -> Checked<()> {
    let features: Option<Vec<&str>> = manifest
        .0
        .get("features")
        .and_then(Json::as_array)
        .and_then(|items| {
            items
                .iter()
                .map(|item| item.as_str().filter(|text| !text.is_empty()))
                .collect()
        });
    let Some(features) = features else {
        return reject("features must be a list of non-empty strings".to_owned());
    };
    match FEATURES
        .into_iter()
        .find(|marker| !features.contains(marker))
    {
        Some(marker) => reject(format!("missing feature marker: {marker}")),
        None => Ok(()),
    }
}

fn check_library_kind(manifest: &Manifest<'_>) -> Checked<()> {
    let platform = manifest.text("platform");
    let library = manifest.text("library");
    let name = library
        .rsplit('/')
        .find(|part| !part.is_empty())
        .unwrap_or_default();
    let (prefixes, suffix, message): (&[&str], &str, String) = match platform {
        _ if platform.starts_with("darwin-") => (
            &["darwin-"],
            ".dylib",
            format!("darwin artifact must contain a dylib: {name}"),
        ),
        _ if platform.starts_with("windows-") => (
            &["windows-"],
            ".dll",
            format!("windows artifact must contain a .dll: {name}"),
        ),
        _ => (
            &["linux-", "android-"],
            ".so",
            format!("{platform} artifact must contain a .so: {name}"),
        ),
    };
    let applies = prefixes.iter().any(|prefix| platform.starts_with(prefix));
    if applies && !name.ends_with(suffix) {
        return reject(message);
    }
    Ok(())
}
