//! No private key, access token or excluded file is tracked under the
//! plugin. A hit prints the path, the line and the kind of secret, never the
//! matched text, so a failing CI log does not republish it.
//!
//! Public key material is not matched: key ids, public keys and digests are
//! 64-hex values too, so the hex rule fires only after a private-key field
//! name.

use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

pub(super) fn check(root: &Path, plugin: &Path) -> DynResult<CheckReport> {
    // A CI container checks out as another user; trust this checkout only.
    let listed = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={}", root.display()))
        .args(["ls-files", "-z", "--", "."])
        .current_dir(plugin)
        .output()?;
    if !listed.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&listed.stderr)
        )
        .into());
    }
    let mut errors = String::new();
    for path in listed
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let path = String::from_utf8_lossy(path);
        if let Some(kind) = excluded_kind(&path) {
            writeln!(errors, "no-keys: {path}: excluded file ({kind})")?;
        }
        let bytes = std::fs::read(plugin.join(path.as_ref()))
            .map_err(|error| format!("{path}: {error}"))?;
        if bytes.contains(&0) {
            continue; // binary, as grep -I skips it
        }
        for (number, line) in String::from_utf8_lossy(&bytes).lines().enumerate() {
            for kind in secret_kinds(line) {
                writeln!(errors, "no-keys: {path}:{}: {kind}", number + 1)?;
            }
        }
    }
    Ok(if errors.is_empty() {
        CheckReport::success("no-keys: clean\n".into())
    } else {
        CheckReport::failure(String::new(), errors)
    })
}

/// The kinds of secret one line carries.
pub(super) fn secret_kinds(line: &str) -> Vec<&'static str> {
    let alnum = |b: u8| b.is_ascii_alphanumeric();
    let upper_digit = |b: u8| b.is_ascii_uppercase() || b.is_ascii_digit();
    let mut kinds = Vec::new();
    if private_key_block(line) {
        kinds.push("private-key-block");
    }
    if ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
        .iter()
        .any(|prefix| run_after(line, prefix, 36, alnum))
        || run_after(line, "github_pat_", 40, |b| alnum(b) || b == b'_')
    {
        kinds.push("github-token");
    }
    if run_after(line, "cio", 32, alnum) {
        kinds.push("crates-io-token");
    }
    if run_after(line, "hf_", 34, alnum) {
        kinds.push("hugging-face-token");
    }
    if run_after(line, "AKIA", 16, upper_digit) {
        kinds.push("aws-access-key");
    }
    if private_key_hex(line) {
        kinds.push("private-key-hex");
    }
    kinds
}

/// Some occurrence of `prefix` is followed by at least `min` bytes in `class`.
fn run_after(line: &str, prefix: &str, min: usize, class: impl Fn(u8) -> bool) -> bool {
    line.match_indices(prefix).any(|(at, _)| {
        line.as_bytes()[at + prefix.len()..]
            .iter()
            .take_while(|b| class(**b))
            .count()
            >= min
    })
}

/// `-----BEGIN ([A-Z0-9]+ )*PRIVATE KEY( BLOCK)?-----`
fn private_key_block(line: &str) -> bool {
    line.match_indices("-----BEGIN ").any(|(at, marker)| {
        let rest = &line[at + marker.len()..];
        let Some(end) = rest.find("-----") else {
            return false;
        };
        let label = &rest[..end];
        let words = label
            .strip_suffix("PRIVATE KEY BLOCK")
            .or_else(|| label.strip_suffix("PRIVATE KEY"));
        words.is_some_and(|words| {
            words.is_empty()
                || words.strip_suffix(' ').is_some_and(|words| {
                    words.split(' ').all(|word| {
                        !word.is_empty()
                            && word
                                .bytes()
                                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
                    })
                })
        })
    })
}

/// `(private_key|secret_key|signing_key|privkey|seed|secret)["']?\s*[:=]\s*["']?[0-9a-fA-F]{64}`
fn private_key_hex(line: &str) -> bool {
    [
        "private_key",
        "secret_key",
        "signing_key",
        "privkey",
        "seed",
        "secret",
    ]
    .iter()
    .any(|name| {
        line.match_indices(name).any(|(at, _)| {
            let mut rest = &line.as_bytes()[at + name.len()..];
            if let [b'"' | b'\'', tail @ ..] = rest {
                rest = tail;
            }
            rest = trim_space(rest);
            let [b':' | b'=', tail @ ..] = rest else {
                return false;
            };
            rest = trim_space(tail);
            if let [b'"' | b'\'', tail @ ..] = rest {
                rest = tail;
            }
            rest.iter().take_while(|b| b.is_ascii_hexdigit()).count() >= 64
        })
    })
}

fn trim_space(bytes: &[u8]) -> &[u8] {
    let skip = bytes.iter().take_while(|b| b.is_ascii_whitespace()).count();
    &bytes[skip..]
}

/// Files that never belong in the plugin (matched case-insensitively): Python,
/// demo or red-team material, key or token files, node data, a NOTICE file.
pub(super) fn excluded_kind(path: &str) -> Option<&'static str> {
    let path = path.to_ascii_lowercase();
    let name = path.rsplit('/').next().unwrap_or(&path);
    let parts: Vec<&str> = path.split('/').collect();
    let directories = &parts[..parts.len() - 1];
    let extension = name.rsplit_once('.').map(|(_, ext)| ext);
    if matches!(extension, Some("py" | "pyc" | "pyd" | "pyo" | "pyw"))
        || directories.contains(&"__pycache__")
    {
        return Some("python-file");
    }
    if matches!(name, "demo.sh" | "demo.md") {
        return Some("demo-material");
    }
    if parts
        .iter()
        .any(|part| part.contains("redteam") || part.contains("red-team"))
    {
        return Some("red-team-material");
    }
    let notice_suffix = name.strip_prefix("notice");
    if notice_suffix.is_some_and(|suffix| {
        suffix.is_empty()
            || suffix
                .strip_prefix('.')
                .is_some_and(|ext| !ext.is_empty() && ext.bytes().all(|b| b.is_ascii_lowercase()))
    }) {
        return Some("notice-file");
    }
    if matches!(
        extension,
        Some("pem" | "key" | "p8" | "p12" | "pfx" | "jwk")
    ) || ["id_ed25519", "id_rsa", "id_ecdsa"]
        .iter()
        .any(|key| name.starts_with(key))
    {
        return Some("key-file");
    }
    if extension == Some("token") || name == ".env" || name.starts_with(".env.") {
        return Some("token-file");
    }
    if matches!(extension, Some("jsonl" | "sqlite" | "sqlite3" | "db"))
        || directories
            .iter()
            .any(|dir| matches!(*dir, "capsule-data" | "ledger"))
    {
        return Some("node-data");
    }
    None
}
