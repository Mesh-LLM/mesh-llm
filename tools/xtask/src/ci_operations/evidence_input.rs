//! Bounded evidence input from `scripts/runner-image-evidence.py`: strict
//! JSON decoding, bounded regular-file reads, the evidence path layout and
//! the scalar checks shared by provenance and cohort validation. Every
//! failure is the legacy `runner evidence: ...` text.

use crate::ci_operations::python_access::{Outcome, type_name};
use crate::ci_operations::python_json_decode::{self, DecodeError, Hooks};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const MAX_SAFE: i128 = 9_007_199_254_740_991;
pub(crate) const IMAGE: &str = "ghcr.io/mesh-llm/mesh-llm-cuda-runner";

pub(crate) fn require(ok: bool, message: &str) -> Outcome<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("runner evidence: {message}"))
    }
}

/// `isinstance(value, dict) and set(value) == set(names.split())`.
pub(crate) fn fields(value: &Json, names: &str) -> Outcome<()> {
    let exact = crate::ci_operations::python_access::has_exact_fields(value, names);
    require(exact, &format!("invalid fields: {names}"))
}

pub(crate) fn text(value: &Json) -> Outcome<()> {
    let bounded = value.as_str().is_some_and(|text| {
        let count = text.chars().count();
        count > 0 && count <= 256 && !text.chars().any(|ch| (ch as u32) < 32 || ch as u32 == 127)
    });
    require(bounded, "invalid bounded string")
}

fn lower_hex(text: &str, len: usize) -> bool {
    text.len() == len
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// `sha256:` followed by 64 lowercase hex digits.
pub(crate) fn is_digest(value: &Json) -> bool {
    value
        .as_str()
        .and_then(|text| text.strip_prefix("sha256:"))
        .is_some_and(|hex| lower_hex(hex, 64))
}

pub(crate) fn digest(value: &Json) -> Outcome<()> {
    require(is_digest(value), "invalid SHA256")
}

pub(crate) fn revision(value: &Json) -> Outcome<()> {
    require(
        value.as_str().is_some_and(|text| lower_hex(text, 40)),
        "invalid revision",
    )
}

pub(crate) fn hash_bytes(value: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(value)))
}

fn pairs(items: Vec<(String, Json)>) -> Result<Json, String> {
    for (index, (key, _)) in items.iter().enumerate() {
        if items[..index].iter().any(|(seen, _)| seen == key) {
            return Err("runner evidence: duplicate JSON key".to_owned());
        }
    }
    Ok(Json::Object(items))
}

fn nonfinite(_: &str) -> Result<Json, String> {
    Err("runner evidence: nonfinite JSON".to_owned())
}

/// `decode(raw)`: strict JSON within the size, depth and integer bounds.
pub(crate) fn decode(raw: &[u8]) -> Outcome<Json> {
    require(raw.len() as u64 <= MAX_BYTES, "JSON exceeds 32 MiB")?;
    // Nesting up to CPython's scanner limit recurses deeper than the main
    // thread's stack allows, so decoding (and dropping a rejected deep
    // value) happens on a thread sized for it.
    let owned = raw.to_vec();
    let worker = std::thread::Builder::new()
        .stack_size(DECODE_STACK_BYTES)
        .spawn(move || decode_bounded(&owned))
        .map_err(|error| error.to_string())?;
    worker
        .join()
        .unwrap_or_else(|_| Err("runner evidence: JSON decoder failed".to_owned()))
}

const DECODE_STACK_BYTES: usize = 64 * 1024 * 1024;

fn decode_bounded(raw: &[u8]) -> Outcome<Json> {
    let hooks = Hooks {
        pairs,
        constant: nonfinite,
    };
    let value = python_json_decode::loads(raw, &hooks).map_err(|error| match error {
        DecodeError::Value(text) => text,
        DecodeError::Recursion => "runner evidence: JSON nesting exceeds parser limit".to_owned(),
    })?;
    bounded(&value, 0)?;
    Ok(value)
}

fn bounded(node: &Json, depth: usize) -> Outcome<()> {
    require(depth <= 64, "JSON nesting exceeds 64 levels")?;
    match node {
        Json::Object(entries) => {
            for (key, child) in entries {
                text(&Json::String(key.clone()))?;
                bounded(child, depth + 1)?;
            }
        }
        Json::Array(items) => {
            require(items.len() <= 4096, "array exceeds 4096 entries")?;
            for child in items {
                bounded(child, depth + 1)?;
            }
        }
        Json::Number(_) => {
            let safe =
                type_name(node) == "int" && node.as_int().is_some_and(|int| int.abs() <= MAX_SAFE);
            require(safe, "expected safe JSON integer")?;
        }
        _ => {}
    }
    Ok(())
}

fn is_symlink(path: &Path) -> bool {
    path.symlink_metadata()
        .is_ok_and(|meta| meta.file_type().is_symlink())
}

/// `read_bytes(path)`: a bounded regular file that is not a symlink.
pub(crate) fn read_bytes(path: &Path) -> Outcome<Vec<u8>> {
    let regular = path.is_file()
        && !is_symlink(path)
        && path.metadata().is_ok_and(|meta| meta.len() <= MAX_BYTES);
    require(regular, "expected bounded regular file")?;
    let shown = python_path_display(path);
    let file = std::fs::File::open(path).map_err(|error| os_error_text(&error, &shown))?;
    let mut raw = Vec::new();
    file.take(MAX_BYTES + 1)
        .read_to_end(&mut raw)
        .map_err(|error| os_error_text(&error, &shown))?;
    require(raw.len() as u64 <= MAX_BYTES, "file exceeds 32 MiB")?;
    Ok(raw)
}

/// `evidence_path(root, sha)`: `<root>/ci/runner-image-evidence/<hex>.json`
/// under the resolved root, refusing symlinked components.
pub(crate) fn evidence_path(root: &Path, sha: &Json) -> Outcome<PathBuf> {
    digest(sha)?;
    require(root.is_dir() && !is_symlink(root), "invalid evidence root")?;
    let mut path = std::fs::canonicalize(root)
        .map_err(|error| os_error_text(&error, &python_path_display(root)))?;
    let hex = sha
        .as_str()
        .unwrap_or_default()
        .trim_start_matches("sha256:");
    for part in ["ci", "runner-image-evidence", &format!("{hex}.json")] {
        path.push(part);
        require(!is_symlink(&path), "symlink evidence path")?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_operations_decode_rejects_duplicates_depth_and_floats() {
        let error = |raw: &[u8]| decode(raw).expect_err("invalid");
        assert_eq!(
            error(br#"{"a":1,"a":2}"#),
            "runner evidence: duplicate JSON key"
        );
        assert_eq!(error(b"1.0"), "runner evidence: expected safe JSON integer");
        assert_eq!(
            error(b"9007199254740992"),
            "runner evidence: expected safe JSON integer"
        );
        assert_eq!(error(b"NaN"), "runner evidence: nonfinite JSON");
        let deep = format!("{}0{}", "[".repeat(66), "]".repeat(66));
        assert_eq!(
            error(deep.as_bytes()),
            "runner evidence: JSON nesting exceeds 64 levels"
        );
        let deeper = format!("{}0{}", "[".repeat(9999), "]".repeat(9999));
        assert_eq!(
            error(deeper.as_bytes()),
            "runner evidence: JSON nesting exceeds parser limit"
        );
        assert_eq!(
            error(br#"{"":1}"#),
            "runner evidence: invalid bounded string"
        );
    }
}
