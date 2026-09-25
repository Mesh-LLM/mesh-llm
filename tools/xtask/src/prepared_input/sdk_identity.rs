//! Native SDK consumers that bind a verified artifact to the requested
//! build: `native-sdk-identity` (the inline program shared by
//! `prepare-native-sdk-input` and `scripts/restore-native-sdk-input.sh`) and
//! `native-sdk-library-dir` (the JNA library directory the Kotlin smoke uses).

use super::{Checked, Rejected, positional, python_json, python_value};
use crate::ci_plan::document::Json;
use std::path::Path;

fn load(manifest: &str) -> Checked<Json> {
    python_json::load(Path::new(manifest))
        .map_err(|error| Rejected(format!("native SDK manifest is not valid JSON: {error}")))
}

/// Args: MANIFEST TARGET BACKEND PROFILE.
pub(super) fn identity(args: &[String]) -> Checked<String> {
    let [manifest, target, backend, profile] = positional(args, "MANIFEST TARGET BACKEND PROFILE")?;
    let manifest = load(manifest)?;
    for (field, expected) in [
        ("target_triple", target),
        ("backend", backend),
        ("cargo_profile", profile),
    ] {
        let actual = manifest.get(field);
        if actual.and_then(Json::as_str) != Some(expected) {
            return Err(Rejected(format!(
                "native SDK manifest {field} mismatch: expected {}, got {}",
                crate::repository::python_text::repr(expected),
                python_value::repr(actual)
            )));
        }
    }
    Ok(String::new())
}

/// Args: MANIFEST. Prints `os.path.dirname(uniffi_library or library)`.
pub(super) fn library_dir(args: &[String]) -> Checked<String> {
    let [manifest] = positional(args, "MANIFEST")?;
    let manifest = load(manifest)?;
    let uniffi = manifest.get("uniffi_library").filter(|value| match value {
        Json::Null | Json::Bool(false) => false,
        Json::String(text) => !text.is_empty(),
        _ => true,
    });
    let library = uniffi
        .or_else(|| manifest.get("library"))
        .ok_or_else(|| Rejected("native SDK manifest is missing library".to_owned()))?;
    let Some(path) = library.as_str() else {
        return Err(Rejected(
            "native SDK library must be a string path".to_owned(),
        ));
    };
    Ok(format!("{}\n", dirname(path)))
}

/// `os.path.dirname` on POSIX: the head before the last `/`, with trailing
/// slashes stripped unless the head is all slashes.
fn dirname(path: &str) -> &str {
    let Some(split) = path.rfind('/') else {
        return "";
    };
    let head = &path[..=split];
    let trimmed = head.trim_end_matches('/');
    if trimmed.is_empty() { head } else { trimmed }
}

#[cfg(test)]
mod tests {
    use super::dirname;

    #[test]
    fn migration_prepared_inputs_dirname_matches_posixpath() {
        for (path, expected) in [
            ("jna/libuniffi.so", "jna"),
            ("lib/deep//libmesh.so", "lib/deep"),
            ("libmesh.so", ""),
            ("/libmesh.so", "/"),
            ("//x", "//"),
            ("a/b/", "a/b"),
        ] {
            assert_eq!(dirname(path), expected, "{path}");
        }
    }
}
