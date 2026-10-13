//! `artifact verify-checksum <artifact>`: require and verify the canonical
//! `<sha256>  <name>` sidecar beside one artifact. Diagnostics, streams and
//! statuses are byte-compatible with `scripts/verify-checksum-sidecar.py`.

use super::argv::Program;
use crate::ci_plan::catalog::os_error_text;
use crate::repository::check_report::CheckReport;
use crate::repository::text::{repr, splitlines};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;

const PROGRAM: Program = Program {
    name: "verify-checksum-sidecar.py",
    positionals: &["artifact"],
};

const DIGEST_LEN: usize = 64;

pub(super) fn run(args: &[String]) -> CheckReport {
    match PROGRAM.parse(args) {
        Ok(values) => match verify(values[0]) {
            Ok(()) => CheckReport::success(String::new()),
            Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
        },
        Err(report) => report,
    }
}

fn verify(argument: &str) -> Result<(), String> {
    let artifact = pure_posix(argument);
    let name = artifact.rsplit('/').next().unwrap_or_default();
    if name.is_empty() || name == "." {
        return Err(format!("artifact path has an empty filename: {artifact:?}"));
    }
    let sidecar = format!("{artifact}.sha256");
    let populated = std::fs::metadata(&sidecar).is_ok_and(|meta| meta.is_file() && meta.len() > 0);
    if !populated {
        return Err(format!(
            "archive checksum sidecar is missing or empty: {sidecar}"
        ));
    }
    let bytes = std::fs::read(&sidecar).map_err(|error| os_error_text(&error, &sidecar))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("checksum sidecar is not valid UTF-8: {error}"))?;
    let [line] = splitlines(text)[..] else {
        return Err("checksum sidecar must contain exactly one canonical line".to_owned());
    };
    let Some((expected, recorded)) = canonical_line(line) else {
        return Err("checksum sidecar must use '<sha256>  <archive-name>' format".to_owned());
    };
    if recorded != name {
        return Err(format!(
            "checksum sidecar names {}, expected {}",
            repr(recorded),
            repr(name)
        ));
    }
    let actual = sha256_file(&artifact).map_err(|error| os_error_text(&error, &artifact))?;
    if actual != expected {
        return Err(format!(
            "archive checksum mismatch: {artifact}\n  expected: {expected}\n  actual:   {actual}"
        ));
    }
    Ok(())
}

/// `([0-9a-f]{64}) {2}([^/\\\r\n]+)` as a full match.
fn canonical_line(line: &str) -> Option<(&str, &str)> {
    let digest = line.get(..DIGEST_LEN)?;
    let name = line.get(DIGEST_LEN..)?.strip_prefix("  ")?;
    let digest_valid = digest
        .bytes()
        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    let name_valid = !name.is_empty() && !name.contains(['/', '\\', '\r', '\n']);
    (digest_valid && name_valid).then_some((digest, name))
}

fn sha256_file(path: &str) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut chunk = vec![0; 1024 * 1024];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            return Ok(hex::encode(digest.finalize()));
        }
        digest.update(&chunk[..read]);
    }
}

/// `str(PurePosixPath(text))`: drop empty and `.` segments, keep a leading
/// `/` (or exactly `//`, which POSIX reserves), and render the empty path `.`.
fn pure_posix(text: &str) -> String {
    let parts = text
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>()
        .join("/");
    let anchor = match text.strip_prefix("//") {
        Some(rest) if !rest.starts_with('/') => "//",
        _ if text.starts_with('/') => "/",
        _ => "",
    };
    match (anchor, parts.is_empty()) {
        ("", true) => ".".to_owned(),
        _ => format!("{anchor}{parts}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_posix_paths_preserve_required_component_and_anchor_rules() {
        for (input, expected) in [
            ("./dist//runtime.tar.gz", "dist/runtime.tar.gz"),
            ("//runtime", "//runtime"),
            ("///runtime", "/runtime"),
            ("", "."),
            ("/", "/"),
            ("a/..", "a/.."),
            ("dist/", "dist"),
        ] {
            assert_eq!(pure_posix(input), expected, "{input:?}");
        }
    }

    #[test]
    fn migration_archives_canonical_line_requires_exact_shape() {
        let digest = "a".repeat(DIGEST_LEN);
        assert_eq!(
            canonical_line(&format!("{digest}  x y")),
            Some((digest.as_str(), "x y"))
        );
        for line in [
            format!("{digest} x"),
            format!("{digest}  "),
            format!("{digest}  a/b"),
            format!("{digest}  a\\b"),
            format!("{}  x", "A".repeat(DIGEST_LEN)),
            format!("é{}  x", "a".repeat(DIGEST_LEN - 1)),
        ] {
            assert_eq!(canonical_line(&line), None, "{line:?}");
        }
    }
}
