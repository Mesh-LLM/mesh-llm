use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use anyhow::{Context, Result};

use crate::{package::SkippyPackageIdentity, source::synthetic_content_addressed_gguf_package};

pub(super) const CONTENT_ADDRESSED_GGUF_PREFIX: &str = "local-gguf://sha256/";

static CONTENT_ADDRESSED_SOURCES: OnceLock<Mutex<HashMap<String, BTreeSet<PathBuf>>>> =
    OnceLock::new();
static VERIFIED_CONTENT_IDENTITIES: OnceLock<
    Mutex<HashMap<String, HashMap<PathBuf, VerifiedContentIdentity>>>,
> = OnceLock::new();

#[derive(Clone)]
struct VerifiedContentIdentity {
    identity: SkippyPackageIdentity,
    proof: VerifiedIdentityProof,
}

#[derive(Clone)]
enum VerifiedIdentityProof {
    FileFingerprint(Vec<VerifiedFileFingerprint>),
    ImmutableSourceMetadata,
}

#[derive(Clone, Eq, PartialEq)]
pub(super) struct VerifiedFileFingerprint {
    path: PathBuf,
    bytes: u64,
    mtime_nanos: u128,
    ctime_nanos: i128,
    device: u64,
    inode: u128,
}

/// What a verified fingerprint pins beyond size and mtime: the file's identity
/// on its volume and a change time that ordinary copy tools do not restore.
struct FileIdentity {
    device: u64,
    inode: u128,
    ctime_nanos: i128,
}

fn source_registry() -> &'static Mutex<HashMap<String, BTreeSet<PathBuf>>> {
    CONTENT_ADDRESSED_SOURCES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn verified_identity_registry()
-> &'static Mutex<HashMap<String, HashMap<PathBuf, VerifiedContentIdentity>>> {
    VERIFIED_CONTENT_IDENTITIES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn content_addressed_package_ref(sha256: &str) -> Result<String> {
    anyhow::ensure!(
        is_sha256(sha256),
        "content-addressed GGUF identity must be 64 lowercase hex characters"
    );
    Ok(format!("{CONTENT_ADDRESSED_GGUF_PREFIX}{sha256}"))
}

pub fn is_content_addressed_gguf_ref(package_ref: &str) -> bool {
    package_ref
        .strip_prefix(CONTENT_ADDRESSED_GGUF_PREFIX)
        .is_some_and(is_sha256)
}

pub(super) fn register_content_addressed_identity(
    identity: &SkippyPackageIdentity,
    fingerprint: Option<Vec<VerifiedFileFingerprint>>,
    allow_immutable_source_metadata: bool,
) {
    if !is_content_addressed_gguf_ref(&identity.package_ref) {
        return;
    }
    if let Ok(mut sources) = source_registry().lock() {
        sources
            .entry(identity.package_ref.clone())
            .or_default()
            .insert(identity.source_model_path.clone());
    }
    if let Ok(mut identities) = verified_identity_registry().lock() {
        let entries = identities.entry(identity.package_ref.clone()).or_default();
        match (fingerprint, allow_immutable_source_metadata) {
            (Some(fingerprint), _) => {
                entries.insert(
                    identity.source_model_path.clone(),
                    VerifiedContentIdentity {
                        identity: identity.clone(),
                        proof: VerifiedIdentityProof::FileFingerprint(fingerprint),
                    },
                );
            }
            (None, true) => {
                entries.insert(
                    identity.source_model_path.clone(),
                    VerifiedContentIdentity {
                        identity: identity.clone(),
                        proof: VerifiedIdentityProof::ImmutableSourceMetadata,
                    },
                );
            }
            (None, false) => {
                // A platform or transient filesystem state that cannot produce
                // a complete fingerprint must invalidate any older cache entry.
                entries.remove(&identity.source_model_path);
            }
        }
    }
}

pub fn into_content_addressed_identity(
    mut identity: SkippyPackageIdentity,
) -> Result<SkippyPackageIdentity> {
    identity.package_ref = content_addressed_package_ref(&identity.source_model_sha256)?;
    let fingerprint = verified_file_fingerprint(&identity);
    anyhow::ensure!(
        fingerprint.is_some(),
        "content-addressed package source contains an unsupported file or symlink"
    );
    register_content_addressed_identity(&identity, fingerprint, false);
    Ok(identity)
}

#[cfg(test)]
pub(super) fn register_content_addressed_source(package_ref: &str, path: &Path) {
    if !is_content_addressed_gguf_ref(package_ref) {
        return;
    }
    if let Ok(mut sources) = source_registry().lock() {
        sources
            .entry(package_ref.to_string())
            .or_default()
            .insert(path.to_path_buf());
    }
}

pub(super) fn verified_path_fingerprint(paths: &[PathBuf]) -> Option<Vec<VerifiedFileFingerprint>> {
    paths
        .iter()
        .map(|path| {
            let metadata = std::fs::symlink_metadata(path).ok()?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return None;
            }
            // Without a file identity and a change timestamp, a same-size
            // rewrite with restored mtime cannot be distinguished. Rehash instead.
            let identity = file_identity(path, &metadata)?;
            Some(VerifiedFileFingerprint {
                path: path.clone(),
                bytes: metadata.len(),
                mtime_nanos: super::hash_cache::file_mtime_nanos(&metadata)?,
                ctime_nanos: identity.ctime_nanos,
                device: identity.device,
                inode: identity.inode,
            })
        })
        .collect()
}

#[cfg(unix)]
fn file_identity(_path: &Path, metadata: &std::fs::Metadata) -> Option<FileIdentity> {
    use std::os::unix::fs::MetadataExt;

    Some(FileIdentity {
        device: metadata.dev(),
        inode: u128::from(metadata.ino()),
        ctime_nanos: super::hash_cache::file_ctime_nanos(metadata)?,
    })
}

/// std exposes no device, inode or change time on Windows, so read them from
/// an open handle: the volume serial number and 128-bit file ID (`FileIdInfo`,
/// which also covers ReFS) identify the file, and `ChangeTime` advances on any
/// content or metadata change. Unlike Unix ctime, a process with
/// write-attributes access can set `ChangeTime`; like the rest of this check it
/// gates reuse of a verified hash and is not a defence against a local writer.
#[cfg(windows)]
fn file_identity(path: &Path, _metadata: &std::fs::Metadata) -> Option<FileIdentity> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_BASIC_INFO, FILE_ID_INFO, FileBasicInfo, FileIdInfo, GetFileInformationByHandleEx,
    };

    let file = std::fs::File::open(path).ok()?;
    let handle = file.as_raw_handle();
    let mut id = FILE_ID_INFO::default();
    let mut basic = FILE_BASIC_INFO::default();
    // SAFETY: `handle` stays open for both calls because `file` outlives them,
    // and each buffer is the structure its information class documents,
    // passed with that structure's exact size.
    let queried = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            (&raw mut id).cast(),
            size_of::<FILE_ID_INFO>() as u32,
        ) != 0
            && GetFileInformationByHandleEx(
                handle,
                FileBasicInfo,
                (&raw mut basic).cast(),
                size_of::<FILE_BASIC_INFO>() as u32,
            ) != 0
    };
    queried.then(|| FileIdentity {
        device: id.VolumeSerialNumber,
        inode: u128::from_le_bytes(id.FileId.Identifier),
        // 100 ns ticks since 1601, only ever compared for equality.
        ctime_nanos: i128::from(basic.ChangeTime) * 100,
    })
}

#[cfg(not(any(unix, windows)))]
fn file_identity(_path: &Path, _metadata: &std::fs::Metadata) -> Option<FileIdentity> {
    None
}

fn verified_file_fingerprint(
    identity: &SkippyPackageIdentity,
) -> Option<Vec<VerifiedFileFingerprint>> {
    let paths = identity
        .source_files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    verified_path_fingerprint(&paths)
}

fn cached_verified_identity(package_ref: &str, path: &Path) -> Option<SkippyPackageIdentity> {
    let cached = verified_identity_registry()
        .lock()
        .ok()?
        .get(package_ref)?
        .get(path)?
        .clone();
    match cached.proof {
        VerifiedIdentityProof::FileFingerprint(expected) => {
            (verified_file_fingerprint(&cached.identity)? == expected).then_some(cached.identity)
        }
        VerifiedIdentityProof::ImmutableSourceMetadata => Some(cached.identity),
    }
}

#[cfg(test)]
fn registered_content_addressed_source(package_ref: &str) -> Option<PathBuf> {
    registered_content_addressed_sources(package_ref)
        .into_iter()
        .next()
}

fn registered_content_addressed_sources(package_ref: &str) -> Vec<PathBuf> {
    if !is_content_addressed_gguf_ref(package_ref) {
        return Vec::new();
    }
    source_registry()
        .lock()
        .ok()
        .and_then(|sources| sources.get(package_ref).cloned())
        .unwrap_or_default()
        .into_iter()
        .filter(|path| path.is_file())
        .collect()
}

/// Resolve a content-addressed source from this process's registry and verify
/// its path-free synthetic manifest at the point of use. Explicit local files
/// are hashed on the first strict verification; later checks may reuse only the
/// identity whose hash-bound strong file fingerprint is still unchanged.
/// Immutable Hugging Face artifacts reuse the canonical Hub identity recorded
/// during startup, keeping integrity verification separate from mesh identity.
///
/// The registry is only a locator. It is never accepted as proof that a path
/// still contains the content observed during startup or inventory.
pub fn verify_registered_content_source(
    model_id: &str,
    package_ref: &str,
    expected_manifest_sha256: &str,
    expected_source_sha256: &str,
) -> Result<SkippyPackageIdentity> {
    anyhow::ensure!(
        is_content_addressed_gguf_ref(package_ref),
        "unsupported content-addressed GGUF reference: {package_ref}"
    );
    anyhow::ensure!(
        is_sha256(expected_source_sha256),
        "expected content-addressed GGUF SHA-256 is invalid"
    );
    let candidates = registered_content_addressed_sources(package_ref);
    anyhow::ensure!(
        !candidates.is_empty(),
        "local GGUF content {package_ref} is not registered"
    );
    let mut failure_count = 0_usize;
    for path in candidates {
        let identity_result = cached_verified_identity(package_ref, &path)
            .map(Ok)
            .unwrap_or_else(|| {
                synthetic_content_addressed_gguf_package(model_id, &path)
                    .with_context(|| format!("verify local GGUF content at {}", path.display()))
            });
        let identity = match identity_result {
            Ok(identity) => identity,
            Err(error) => {
                failure_count += 1;
                tracing::debug!(
                    path = %path.display(),
                    package_ref,
                    error = %error,
                    "registered local GGUF failed content verification"
                );
                continue;
            }
        };
        if identity.package_ref == package_ref
            && identity.source_model_sha256 == expected_source_sha256
            && identity.manifest_sha256 == expected_manifest_sha256
        {
            return Ok(identity);
        }
        failure_count += 1;
        tracing::debug!(
            path = %path.display(),
            package_ref,
            "registered local GGUF content identity mismatched"
        );
    }
    anyhow::bail!(
        "no registered local GGUF matches {package_ref} ({failure_count} candidate(s) failed verification)"
    )
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_addressed_ref_requires_full_lowercase_sha256() {
        let digest = "a".repeat(64);
        assert_eq!(
            content_addressed_package_ref(&digest).unwrap(),
            format!("{CONTENT_ADDRESSED_GGUF_PREFIX}{digest}")
        );
        assert!(content_addressed_package_ref(&"a".repeat(63)).is_err());
        assert!(content_addressed_package_ref(&"A".repeat(64)).is_err());
        assert!(content_addressed_package_ref(&"g".repeat(64)).is_err());
    }

    #[test]
    fn registry_is_only_available_for_existing_content_addressed_sources() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"not-a-real-model").unwrap();
        let package_ref = content_addressed_package_ref(&"b".repeat(64)).unwrap();

        register_content_addressed_source(&package_ref, &path);
        assert_eq!(
            registered_content_addressed_source(&package_ref),
            Some(path.clone())
        );

        std::fs::remove_file(path).unwrap();
        assert_eq!(registered_content_addressed_source(&package_ref), None);
    }
    #[cfg(windows)]
    #[test]
    fn windows_fingerprint_is_reusable_and_catches_a_same_size_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"first bytes").unwrap();
        let paths = [path.clone()];

        let first = verified_path_fingerprint(&paths)
            .expect("a regular file needs a fingerprint so its verified hash can be reused");
        assert!(verified_path_fingerprint(&paths) == Some(first.clone()));

        // Same length with the original mtime restored: only the change time
        // records the rewrite.
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&path, b"other bytes").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();

        let rewritten = verified_path_fingerprint(&paths).unwrap();
        assert_eq!(rewritten[0].bytes, first[0].bytes);
        assert_eq!(rewritten[0].mtime_nanos, first[0].mtime_nanos);
        assert!(
            rewritten != first,
            "a same-size rewrite with a restored mtime must not reuse the old hash"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_fingerprint_tells_a_copy_from_the_original() {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("a.gguf");
        let copy = dir.path().join("b.gguf");
        std::fs::write(&original, b"model bytes").unwrap();
        std::fs::copy(&original, &copy).unwrap();

        let original = verified_path_fingerprint(&[original]).unwrap();
        let copy = verified_path_fingerprint(&[copy]).unwrap();
        assert_eq!(original[0].device, copy[0].device);
        assert_ne!(original[0].inode, copy[0].inode);
    }
}
