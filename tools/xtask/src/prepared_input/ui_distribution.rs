//! `ui-distribution {stamp|verify}`: binds a prepared release UI to its
//! source revision and release tag and verifies its exact bytes, replacing
//! `scripts/ui-distribution.py`. The manifest name, schema, identity rules,
//! symlink refusal and module-reference requirement are unchanged.

use super::html_modules::module_sources;
use super::{Checked, Rejected, python_io, python_json};
use crate::ci_plan::document::Json;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const MANIFEST: &str = ".mesh-llm-ui-release.json";

const GRAMMAR: Grammar = Grammar {
    usage: "ui-distribution.py [-h] --dist DIST --source-sha SOURCE_SHA \
            --release-tag RELEASE_TAG {stamp,verify}",
    values: &["--dist", "--source-sha", "--release-tag"],
    flags: &[],
};

/// The release identity every manifest is bound to.
struct Identity<'a> {
    source_sha: &'a str,
    release_tag: &'a str,
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let [operation] = parsed.positionals.as_slice() else {
        return GRAMMAR.error("expected exactly one operation");
    };
    let stamp = match operation.as_str() {
        "stamp" => true,
        "verify" => false,
        other => {
            let message = format!(
                "argument operation: invalid choice: '{other}' (choose from 'stamp', 'verify')"
            );
            return GRAMMAR.error(&message);
        }
    };
    let (Some(dist), Some(source_sha), Some(release_tag)) = (
        parsed.last("--dist"),
        parsed.last("--source-sha"),
        parsed.last("--release-tag"),
    ) else {
        return GRAMMAR
            .error("the following arguments are required: --dist, --source-sha, --release-tag");
    };
    let identity = Identity {
        source_sha,
        release_tag,
    };
    let outcome = if stamp {
        write_stamp(Path::new(dist), &identity)
    } else {
        verify(Path::new(dist), &identity)
    };
    match outcome {
        Ok(()) => CheckReport::success(String::new()),
        Err(Rejected(message)) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

fn write_stamp(root: &Path, identity: &Identity<'_>) -> Checked<()> {
    let manifest = describe(root, identity)?;
    let path = root.join(MANIFEST);
    let bytes = python_json::dumps_indented(&manifest) + "\n";
    fs::write(&path, bytes).map_err(|error| Rejected(python_io::os_error(&path, &error)))
}

fn verify(root: &Path, identity: &Identity<'_>) -> Checked<()> {
    let expected = describe(root, identity)?;
    let path = root.join(MANIFEST);
    let regular = fs::symlink_metadata(&path).is_ok_and(|meta| meta.is_file());
    if !regular {
        return Err("UI distribution is missing its release manifest".into());
    }
    let text = python_io::read_text(&path)?;
    let recorded = Json::parse(text.as_bytes())
        .map_err(|error| format!("UI release manifest is not valid JSON: {error}"))?;
    if !python_json::equal(&recorded, &expected) {
        return Err("UI release identity or file checksums do not match".into());
    }
    Ok(())
}

/// The manifest the distribution must carry: identity plus the SHA-256 of
/// every regular file except the manifest itself.
fn describe(root: &Path, identity: &Identity<'_>) -> Checked<Json> {
    if !is_source_sha(identity.source_sha) {
        return Err("source SHA must be 40 lowercase hexadecimal characters".into());
    }
    if !is_release_tag(identity.release_tag) {
        return Err("release tag must be a versioned v-prefixed tag".into());
    }
    let real_dir = fs::symlink_metadata(root).is_ok_and(|meta| meta.is_dir());
    if !real_dir {
        return Err("UI distribution must be a real directory".into());
    }
    let mut files = BTreeMap::new();
    collect(root, "", &mut files)?;
    files.remove(MANIFEST);
    if !files.contains_key("index.html") {
        return Err("UI distribution is missing index.html".into());
    }
    let index = python_io::read_text(&root.join("index.html"))?;
    references_local_module(&index, &files)?;
    let files = files
        .into_iter()
        .map(|(name, digest)| (name, Json::String(digest)))
        .collect();
    Ok(Json::Object(vec![
        ("schema".to_owned(), Json::Number(1.into())),
        (
            "source_sha".to_owned(),
            Json::String(identity.source_sha.to_owned()),
        ),
        (
            "release_tag".to_owned(),
            Json::String(identity.release_tag.to_owned()),
        ),
        ("files".to_owned(), Json::Object(files)),
    ]))
}

/// Every regular file below `dir` by POSIX relative path. Any symbolic link
/// rejects the distribution; other special files are ignored.
fn collect(dir: &Path, prefix: &str, files: &mut BTreeMap<String, String>) -> Checked<()> {
    let io = |error: std::io::Error| Rejected(python_io::os_error(dir, &error));
    for entry in fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
        let kind = entry.file_type().map_err(io)?;
        if kind.is_symlink() {
            return Err("UI distribution must not contain symbolic links".into());
        }
        if kind.is_dir() {
            collect(&entry.path(), &format!("{name}/"), files)?;
        } else if kind.is_file() {
            let path = entry.path();
            let bytes = fs::read(&path).map_err(|error| python_io::os_error(&path, &error))?;
            files.insert(name, hex::encode(Sha256::digest(&bytes)));
        }
    }
    Ok(())
}

/// Python evaluates `any(...)` lazily: a valueless `src` fails only when no
/// earlier module script already matched.
fn references_local_module(index: &str, files: &BTreeMap<String, String>) -> Checked<()> {
    for source in module_sources(index) {
        let Some(source) = source else {
            return Err("UI index module script has a src attribute without a value".into());
        };
        let local = source.strip_prefix('/').unwrap_or(&source);
        if files.contains_key(local) && source.ends_with(".js") {
            return Ok(());
        }
    }
    Err("UI index must reference a built local JavaScript module".into())
}

fn is_source_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// `v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?` as a full match.
fn is_release_tag(value: &str) -> bool {
    let Some(version) = value.strip_prefix('v') else {
        return false;
    };
    let (core, suffix) = match version.split_once('-') {
        Some((core, suffix)) => (core, Some(suffix)),
        None => (version, None),
    };
    let numbers: Vec<&str> = core.split('.').collect();
    let digits = |part: &&str| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit());
    numbers.len() == 3
        && numbers.iter().all(digits)
        && suffix.is_none_or(|suffix| {
            !suffix.is_empty()
                && suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
        })
}

#[cfg(test)]
mod tests {
    use super::{is_release_tag, is_source_sha};

    #[test]
    fn migration_prepared_inputs_ui_identity_shapes_match_legacy_regexes() {
        for tag in ["v1.2.3", "v0.74.0-rc.1", "v10.0.0-a-b.c"] {
            assert!(is_release_tag(tag), "{tag}");
        }
        for tag in [
            "1.2.3",
            "v1.2",
            "v1.2.3-",
            "v1.2.3.4",
            "v1.2.3-rc+1",
            "v1..3",
            "v١.2.3",
        ] {
            assert!(!is_release_tag(tag), "{tag}");
        }
        assert!(is_source_sha(&"0f".repeat(20)));
        assert!(!is_source_sha(&"0F".repeat(20)) && !is_source_sha("abc"));
    }
}
