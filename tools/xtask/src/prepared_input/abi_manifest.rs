//! `static-abi-manifest {describe|verify}`: the immutable static ABI input
//! manifest written by `prepare-static-abi-input` and checked by
//! `scripts/restore-static-abi-input.sh` (their inline programs). Verify
//! reads only; a mismatched input is rejected, never rebuilt.

use super::{Checked, Rejected, positional, python_io, python_json, python_value};
use crate::ci_plan::document::Json;
use crate::repository::python_text::strip;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const SCHEMA_VERSION: i64 = 3;
const CONTRACT: &str = "mesh-llm-static-abi-v3";

/// The identity fields both sides bind, in the legacy check order.
fn identity(target: &str, backend: &str, build_dir: &str, epoch: &str) -> Vec<(String, Json)> {
    let text = |value: &str| Json::String(value.to_owned());
    vec![
        (
            "schema_version".to_owned(),
            Json::Number(SCHEMA_VERSION.into()),
        ),
        ("contract".to_owned(), text(CONTRACT)),
        ("target_triple".to_owned(), text(target)),
        ("backend".to_owned(), text(backend)),
        ("build_directory".to_owned(), text(build_dir)),
        ("toolchain_epoch".to_owned(), text(epoch)),
    ]
}

fn stamp_sha256(path: &Path) -> Checked<String> {
    let bytes = fs::read(path).map_err(|error| python_io::os_error(path, &error))?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

/// Args: MANIFEST TARGET BACKEND BUILD_DIR EPOCH PATCHED_SHA_FILE STAMP.
pub(super) fn describe(args: &[String]) -> Checked<String> {
    let [
        manifest,
        target,
        backend,
        build_dir,
        epoch,
        patched_sha,
        stamp,
    ] = positional(
        args,
        "MANIFEST TARGET BACKEND BUILD_DIR EPOCH PATCHED_SHA_FILE STAMP",
    )?;
    let sha_path = Path::new(patched_sha);
    let recorded = python_io::read_text(sha_path)?;
    if strip(&recorded).is_empty() {
        return Err("prepared llama.cpp patched SHA is empty".into());
    }
    let mut fields = identity(target, backend, build_dir, epoch);
    let digest = stamp_sha256(Path::new(stamp))?;
    fields.push(("build_stamp_sha256".to_owned(), Json::String(digest)));
    let bytes = python_json::dumps_indented(&Json::Object(fields)) + "\n";
    let path = Path::new(manifest);
    fs::write(path, bytes).map_err(|error| python_io::os_error(path, &error))?;
    Ok(String::new())
}

/// Args: MANIFEST STAMP TARGET BACKEND BUILD_DIR EPOCH.
pub(super) fn verify(args: &[String]) -> Checked<String> {
    let [manifest, stamp, target, backend, build_dir, epoch] =
        positional(args, "MANIFEST STAMP TARGET BACKEND BUILD_DIR EPOCH")?;
    let path = Path::new(manifest);
    let recorded = python_json::load(path)
        .map_err(|error| format!("static ABI manifest is not valid JSON: {error}"))?;
    if recorded.as_object().is_none() {
        return Err("static ABI manifest must be a JSON object".into());
    }
    for (field, expected) in identity(target, backend, build_dir, epoch) {
        let actual = recorded.get(&field);
        if !actual.is_some_and(|actual| python_json::equal(actual, &expected)) {
            return Err(Rejected(format!(
                "static ABI manifest {field} mismatch: expected {}, got {}",
                python_value::repr(Some(&expected)),
                python_value::repr(actual)
            )));
        }
    }
    let digest = stamp_sha256(Path::new(stamp))?;
    if recorded.get("build_stamp_sha256").and_then(Json::as_str) != Some(digest.as_str()) {
        return Err("static ABI build stamp checksum mismatch".into());
    }
    Ok(String::new())
}
