//! The legacy composer's digests: `file_sha256` (SHA-256 of a file's bytes)
//! and `tree_sha256`, which hashes every regular file below a directory as
//! `len(path) (8 bytes, big-endian) || path || sha256(bytes)` in sorted
//! relative-POSIX-path order. Like `Path.rglob("*")` with `is_file()`,
//! symlinks to files count (by target bytes), symlinked directories are
//! not descended, and a missing or non-directory root has no files.

use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::Path;

/// An I/O failure and the path Python would have reported it for.
pub(crate) struct IoFailure {
    pub(crate) error: std::io::Error,
    pub(crate) path: std::path::PathBuf,
}

fn failure(error: std::io::Error, path: &Path) -> IoFailure {
    IoFailure {
        error,
        path: path.to_path_buf(),
    }
}

fn file_digest(path: &Path) -> Result<[u8; 32], IoFailure> {
    file_digest_with_guard(path, &mut || Ok(()))
}
fn file_digest_with_guard(
    path: &Path,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> Result<[u8; 32], IoFailure> {
    guard().map_err(|error| failure(error, path))?;
    let mut handle = fs::File::open(path).map_err(|error| failure(error, path))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        guard().map_err(|error| failure(error, path))?;
        let read = handle
            .read(&mut buffer)
            .map_err(|error| failure(error, path))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    guard().map_err(|error| failure(error, path))?;
    Ok(digest.finalize().into())
}

pub(crate) fn file_sha256(path: &Path) -> Result<String, IoFailure> {
    file_digest(path).map(hex::encode)
}

pub(crate) fn tree_sha256(root: &Path) -> Result<String, IoFailure> {
    let mut files = Vec::new();
    collect(root, "", &mut files);
    files.sort();
    let mut digest = Sha256::new();
    for relative in files {
        let bytes = relative.as_bytes();
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
        // `Path(".").rglob` yields `d/f`, not `./d/f`.
        let path = if root == Path::new(".") {
            Path::new(&relative).to_path_buf()
        } else {
            root.join(&relative)
        };
        digest.update(file_digest(&path)?);
    }
    Ok(hex::encode(digest.finalize()))
}

/// Relative paths of every file below `dir`; unreadable directories are
/// skipped, as `rglob` ignores their `OSError`.
fn collect(dir: &Path, prefix: &str, files: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let path = entry.path();
        let is_link = entry.file_type().is_ok_and(|kind| kind.is_symlink());
        if !is_link && path.is_dir() {
            collect(&path, &relative, files);
        } else if path.is_file() {
            files.push(relative);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tree_sha256;

    /// `scripts/tests/test_compose_product_bundle.py`: files are ordered by
    /// ordinal relative path, so `README.md` precedes `lib/runtime.dll`.
    #[test]
    fn migration_product_tree_digest_uses_ordinal_path_order() {
        let root = std::env::temp_dir().join(format!("xtask-product-tree-{}", std::process::id()));
        let lib = root.join("lib");
        std::fs::create_dir_all(&lib).expect("scratch tree");
        std::fs::write(root.join("README.md"), b"upper sorts first ordinally\n").expect("file");
        std::fs::write(lib.join("runtime.dll"), b"lower sorts second ordinally\n").expect("file");
        let digest = tree_sha256(&root).map_err(|failure| failure.error.to_string());
        let _cleanup = std::fs::remove_dir_all(&root);
        assert_eq!(
            digest.as_deref(),
            Ok("01df8a658501c6798530548aa7ca5a15ce02059d66b8ab87df4150811b55c7e1")
        );
    }
}

pub(crate) fn file_sha256_with_guard(
    path: &Path,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> Result<String, IoFailure> {
    file_digest_with_guard(path, guard).map(hex::encode)
}

#[cfg(test)]
mod guarded_tests {
    // Append within existing product::digest owning tests module.
    #[test]
    fn guarded_digest_preopen_interruption_precedes_missing_file_error() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("does-not-exist");
        let mut calls = 0;
        let result = super::file_sha256_with_guard(&path, &mut || {
            calls += 1;
            Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "fixture cancelled",
            ))
        });
        let error = result.err().unwrap();
        assert_eq!(error.error.kind(), std::io::ErrorKind::Interrupted);
        assert_eq!(calls, 1);
        root.close().unwrap();
    }
    #[test]
    fn guarded_digest_stops_after_first_chunk_before_remaining_bytes() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("bytes");
        std::fs::write(&path, vec![17u8; 1024 * 1024 + 37]).unwrap();
        let mut calls = 0;
        let result = super::file_sha256_with_guard(&path, &mut || {
            calls += 1;
            if calls == 3 {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "after first chunk",
                ))
            } else {
                Ok(())
            }
        });
        assert_eq!(
            result.err().unwrap().error.kind(),
            std::io::ErrorKind::Interrupted
        );
        assert_eq!(calls, 3);
        root.close().unwrap();
    }
    #[test]
    fn guarded_digest_matches_existing_file_bytes_without_changing_tree_semantics() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("bytes");
        std::fs::write(&path, vec![23u8; 1024 * 1024 + 37]).unwrap();
        let legacy = super::file_sha256(&path).map_err(|e| e.error).unwrap();
        let guarded = super::file_sha256_with_guard(&path, &mut || Ok(()))
            .map_err(|e| e.error)
            .unwrap();
        assert_eq!(guarded, legacy);
        use sha2::Digest as _;
        assert_eq!(
            guarded,
            hex::encode(sha2::Sha256::digest(vec![23u8; 1024 * 1024 + 37]))
        );
        root.close().unwrap();
    }
}
