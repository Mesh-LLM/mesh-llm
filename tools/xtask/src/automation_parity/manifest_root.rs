//! The inert PR manifest root, derived as the protected `plan-ci` action
//! derives it: a 40-hex source commit, both catalogs present as mode-100644
//! blobs, byte-identical to the protected checkout's copies, then extracted
//! into a fresh directory. Cargo discovery never moves to this directory.

use crate::command::DynResult;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) const CATALOGS: [&str; 2] = ["ci/ownership.yml", "ci/slices.yml"];

/// A caller-contract rejection, worded as the protected action words it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Rejected(pub(super) String);

/// Where the source revision lives and which revision to read.
pub(super) struct Source<'a> {
    pub(super) repo: &'a Path,
    pub(super) sha: &'a str,
}

fn git(repo: &Path, args: &[&str]) -> DynResult<std::process::Output> {
    Ok(Command::new("git").current_dir(repo).args(args).output()?)
}

fn well_formed(sha: &str) -> bool {
    sha.len() == 40
        && sha
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// The blob of one catalog, or the action's rejection.
fn regular_blob(source: &Source<'_>, manifest: &str) -> DynResult<Result<String, Rejected>> {
    let listing = git(source.repo, &["ls-tree", source.sha, "--", manifest])?;
    let text = String::from_utf8_lossy(&listing.stdout);
    let Some(line) = text.lines().next().filter(|line| !line.trim().is_empty()) else {
        return Ok(Err(Rejected(format!(
            "source manifest is missing: {manifest}"
        ))));
    };
    let (meta, path) = line.split_once('\t').unwrap_or((line, ""));
    let fields = meta.split_whitespace().collect::<Vec<_>>();
    match fields.as_slice() {
        ["100644", "blob", object] if path == manifest => Ok(Ok((*object).to_owned())),
        _ => Ok(Err(Rejected(format!(
            "source manifest is missing or is not a regular file: {manifest}"
        )))),
    }
}

fn blob_bytes(repo: &Path, object: &str) -> DynResult<Vec<u8>> {
    let output = git(repo, &["cat-file", "blob", object])?;
    if !output.status.success() {
        return Err(format!("git cat-file blob {object} failed").into());
    }
    Ok(output.stdout)
}

/// Checks the source revision and extracts its catalogs into `destination`
/// (which must not yet exist). `protected` is the protected checkout.
pub(super) fn materialize(
    protected: &Path,
    source: &Source<'_>,
    destination: &Path,
) -> DynResult<Result<PathBuf, Rejected>> {
    if !well_formed(source.sha) {
        return Ok(Err(Rejected(
            "pull request source SHA is malformed".to_owned(),
        )));
    }
    let commit = format!("{}^{{commit}}", source.sha);
    if !git(source.repo, &["cat-file", "-e", &commit])?
        .status
        .success()
    {
        return Ok(Err(Rejected(format!(
            "source commit is missing: {}",
            source.sha
        ))));
    }
    let mut blobs = Vec::new();
    for manifest in CATALOGS {
        match regular_blob(source, manifest)? {
            Ok(object) => blobs.push((manifest, blob_bytes(source.repo, &object)?)),
            Err(rejected) => return Ok(Err(rejected)),
        }
    }
    // The action compares the slice catalog first, then ownership.
    for (manifest, noun) in [
        ("ci/slices.yml", "slice"),
        ("ci/ownership.yml", "ownership"),
    ] {
        let bytes = blobs
            .iter()
            .find(|(name, _)| *name == manifest)
            .map(|(_, bytes)| bytes.as_slice())
            .unwrap_or_default();
        if fs::read(protected.join(manifest))? != bytes {
            return Ok(Err(Rejected(format!(
                "source {noun} catalog differs from the protected planner catalog"
            ))));
        }
    }
    fs::create_dir_all(destination.join("ci"))?;
    for (manifest, bytes) in &blobs {
        fs::write(destination.join(manifest), bytes)?;
        set_regular_mode(&destination.join(manifest))?;
    }
    Ok(Ok(destination.to_path_buf()))
}

/// Every regular file below `root`, as sorted relative paths.
pub(super) fn entries(root: &Path) -> DynResult<Vec<String>> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let path = entry?.path();
            if fs::symlink_metadata(&path)?.is_dir() {
                pending.push(path);
            } else {
                let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
                found.push(relative);
            }
        }
    }
    found.sort();
    Ok(found)
}

#[cfg(unix)]
fn set_regular_mode(path: &Path) -> DynResult<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o644))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_regular_mode(_path: &Path) -> DynResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_shadow_source_sha_is_forty_lowercase_hex() {
        assert!(well_formed(&"a".repeat(40)));
        assert!(!well_formed(&"A".repeat(40)));
        assert!(!well_formed(&"a".repeat(39)));
        assert!(!well_formed(&"g".repeat(40)));
    }
}
