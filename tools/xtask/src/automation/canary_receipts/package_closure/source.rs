use super::process;
use crate::automation::canary_receipts::Digest;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

pub(super) const MARKERS: [&str; 4] = [
    ".mesh-llm-upstream-sha",
    ".mesh-llm-patch-digest",
    ".mesh-llm-patched-sha",
    ".mesh-llm-prepare-schema",
];

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Provenance {
    pub(super) head: String,
    pub(super) markers: BTreeMap<String, String>,
}

pub(super) fn revision(value: &str) -> DynResult<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("expected lowercase 40-hex source revision".into());
    }
    Ok(())
}

pub(super) fn branch(value: &str) -> DynResult<()> {
    if !value.starts_with("llama-canary/repair-")
        || value.ends_with(['.', '/'])
        || value.contains("..")
        || value.contains("//")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._/-".contains(&byte))
    {
        return Err("invalid canary candidate branch".into());
    }
    Ok(())
}

fn patch_file(path: &Path, root: &Path) -> DynResult<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || !path.canonicalize()?.starts_with(root) {
        return Err("patch recipe member is not a contained regular file".into());
    }
    Ok(())
}

fn names(directory: &Path) -> DynResult<Vec<String>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "non-UTF8 patch name")?;
        if name.ends_with(".patch") {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

fn patch_name(name: &str, index: usize, lane: Option<&str>) -> DynResult<()> {
    let prefix = format!("{:04}-", index + 1);
    let suffix = name
        .strip_prefix(&prefix)
        .and_then(|name| name.strip_suffix(".patch"))
        .ok_or("noncontiguous patch sequence")?;
    if suffix.is_empty() || suffix.contains(['/', '\\', '\n', '\r', '\0']) {
        return Err("unsafe patch name".into());
    }
    match lane {
        Some("model_support")
            if !suffix.as_bytes()[0].is_ascii_lowercase()
                && !suffix.as_bytes()[0].is_ascii_digit() =>
        {
            return Err("invalid model-support patch name".into());
        }
        Some("generated") if !suffix.starts_with("family-") || suffix.len() == 7 => {
            return Err("invalid generated family patch name".into());
        }
        _ => {}
    }
    if lane.is_some()
        && !suffix
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b".-".contains(&byte))
    {
        return Err("invalid patch lane suffix".into());
    }
    Ok(())
}

/// Exact preparation order: numbered core, model_support series, generated series.
pub(super) fn ordered_patch_members(directory: &Path) -> DynResult<(PathBuf, Vec<PathBuf>)> {
    let root = directory.canonicalize()?;
    let core = names(&root)?;
    let mut members = Vec::<PathBuf>::new();
    for (index, name) in core.iter().enumerate() {
        patch_name(name, index, None)?;
        members.push(root.join(name));
    }
    for lane in ["model_support", "generated"] {
        let directory = root.join(lane);
        if !directory.exists() {
            continue;
        }
        if fs::symlink_metadata(&directory)?.file_type().is_symlink() {
            return Err("symlinked patch lane".into());
        }
        patch_file(&directory.join("series"), &root)?;
        let series = fs::read_to_string(directory.join("series"))?;
        let order = series.lines().collect::<Vec<_>>();
        let disk = names(&directory)?;
        if order.is_empty() || order.len() != disk.len() {
            return Err("patch series does not cover lane".into());
        }
        let mut listed = order
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        listed.sort();
        if listed != disk {
            return Err("patch series membership mismatch".into());
        }
        for (index, name) in order.iter().enumerate() {
            patch_name(name, index, Some(lane))?;
            members.push(directory.join(name));
        }
    }
    for path in &members {
        patch_file(path, &root)?;
    }
    Ok((root, members))
}

pub(super) fn patch_digest(directory: &Path) -> DynResult<Digest> {
    let (root, members) = ordered_patch_members(directory)?;
    let mut hash = Sha256::new();
    for path in members {
        patch_file(&path, &root)?;
        let name = path
            .strip_prefix(&root)?
            .to_str()
            .ok_or("non-UTF8 patch path")?;
        hash.update(format!("{name}\n{}\n", Digest::of_file(&path)?.as_str()).as_bytes());
    }
    Ok(Digest::try_from(hex::encode(hash.finalize()))?)
}

// Revision and digest metadata is tiny; regular-file admission also prevents
// caller-supplied FIFOs/devices from blocking before bounded Git supervision.
fn revision_metadata(path: &Path) -> DynResult<String> {
    const MAXIMUM: u64 = 4096;
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("prepared-source metadata must be a regular file".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.file_type().is_file() {
        return Err("opened prepared-source metadata must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(MAXIMUM + 1).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAXIMUM {
        return Err("prepared-source revision metadata exceeds 4096 bytes".into());
    }
    Ok(String::from_utf8(bytes)?)
}

pub(super) fn validate_recipe(root: &Path, provenance: &Provenance) -> DynResult<()> {
    revision(&provenance.head)?;
    if provenance.markers.len() != MARKERS.len()
        || MARKERS
            .iter()
            .any(|name| !provenance.markers.contains_key(*name))
    {
        return Err("prepared source marker set mismatch".into());
    }
    let marker = |name: &str| provenance.markers[name].trim();
    let upstream = revision_metadata(&root.join("third_party/llama.cpp/upstream.txt"))?;
    revision(upstream.trim())?;
    if marker(".mesh-llm-prepare-schema") != "5"
        || marker(".mesh-llm-upstream-sha") != upstream.trim()
        || marker(".mesh-llm-patched-sha") != provenance.head
        || marker(".mesh-llm-patch-digest")
            != patch_digest(&root.join("third_party/llama.cpp/patches"))?.as_str()
    {
        return Err("prepared source differs from current pinned patch recipe".into());
    }
    Ok(())
}

pub(super) fn prepared(root: &Path) -> DynResult<Provenance> {
    let checkout = root.join(".deps/llama.cpp");
    let mut markers = BTreeMap::new();
    for name in MARKERS {
        markers.insert(name.to_owned(), revision_metadata(&checkout.join(name))?);
    }
    let provenance = Provenance {
        head: process::text(&checkout, &["rev-parse", "HEAD"])?,
        markers,
    };
    validate_recipe(root, &provenance)?;
    process::text(&checkout, &["diff-index", "--quiet", "HEAD", "--"])?;
    Ok(provenance)
}

/// Git advertises a branch bundle using its full ref; identity stores the admitted branch name.
pub(super) fn candidate_bundle_identity(
    bytes: &[u8],
    candidate: &str,
    name: &str,
) -> DynResult<()> {
    revision(candidate)?;
    branch(name)?;
    let reference = format!("refs/heads/{name}");
    if std::str::from_utf8(bytes)?
        .split_whitespace()
        .collect::<Vec<_>>()
        != [candidate, reference.as_str()]
    {
        return Err("candidate bundle head/branch differs from exact identity".into());
    }
    Ok(())
}

#[cfg(test)]
mod bundle_identity_tests {
    use super::candidate_bundle_identity;
    #[test]
    fn admitted_bundle_ref_is_exact_and_never_accepts_foreign_or_extra_heads() {
        let candidate = "a".repeat(40);
        let branch = "llama-canary/repair-finite";
        let valid = format!("{candidate} refs/heads/{branch}\n");
        assert!(candidate_bundle_identity(valid.as_bytes(), &candidate, branch).is_ok());
        for advertised in [
            format!("{candidate} {branch}\n"),
            format!("{candidate} refs/tags/{branch}\n"),
            format!("{} refs/heads/{branch}\n", "b".repeat(40)),
            format!("{valid}{candidate} refs/heads/other\n"),
        ] {
            assert!(candidate_bundle_identity(advertised.as_bytes(), &candidate, branch).is_err());
        }
    }
}
