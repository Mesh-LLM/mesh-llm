//! Where the evidence of failed audits is kept: `receipt-evidence/` under
//! the node's identity state directory, for `skippy recheck-receipt
//! --evidence`.
//!
//! Evidence holds the prompt's token ids and activations of the turn, and
//! a peer decides how often and how large it is. So:
//!
//! - the directory is the owner's alone, and so is each file (on Unix:
//!   `0700` and `0600`);
//! - a file's name is a time and a digest, never text the peer chose, so a
//!   long or reused receipt id cannot fail a write or replace earlier
//!   evidence, and a file is never overwritten;
//! - a file is written compactly and never past [`MAX_FILE_BYTES`], and the
//!   directory keeps at most [`MAX_FILES`] files and [`MAX_BYTES`], the
//!   oldest going first.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use iroh::EndpointId;
use sha2::{Digest, Sha256};

/// The most evidence files kept.
pub(super) const MAX_FILES: usize = 64;
/// The most bytes of evidence kept.
pub(super) const MAX_BYTES: u64 = 1 << 30;
/// The largest evidence file: an opening and attention rows of the largest
/// size the audit reads, as base64.
pub(super) const MAX_FILE_BYTES: u64 = 384 << 20;
const DIRECTORY: &str = "receipt-evidence";
const SUFFIX: &str = ".json";

/// The directory evidence is kept in, and its bounds.
#[derive(Clone, Debug)]
pub(super) struct EvidenceStore {
    directory: PathBuf,
    max_files: usize,
    max_bytes: u64,
    max_file_bytes: u64,
}

impl EvidenceStore {
    /// The store under `state_dir`, the node's identity state directory.
    /// `None` when the node has none, or only one relative to the working
    /// directory: evidence is then not kept, never written where the
    /// process happens to run.
    pub(super) fn under(state_dir: Option<&Path>) -> Option<Self> {
        let state_dir = state_dir.filter(|dir| dir.is_absolute())?;
        Some(Self {
            directory: state_dir.join(DIRECTORY),
            max_files: MAX_FILES,
            max_bytes: MAX_BYTES,
            max_file_bytes: MAX_FILE_BYTES,
        })
    }

    /// Keeps `evidence` of a failed audit of `peer`'s turn `receipt_id`, and
    /// returns the file. Never replaces a file.
    pub(super) fn keep<T: serde::Serialize>(
        &self,
        peer: &EndpointId,
        receipt_id: &str,
        evidence: &T,
        now_ms: u64,
    ) -> Result<PathBuf> {
        create_private_dir(&self.directory)?;
        let (path, file) = self.create(peer, receipt_id, now_ms)?;
        let written = self
            .write(file, evidence)
            .with_context(|| format!("write {}", path.display()));
        if let Err(error) = written {
            // a partial file is no evidence
            let _ = std::fs::remove_file(&path);
            return Err(error);
        }
        self.prune(&path);
        Ok(path)
    }

    /// Creates a file no earlier evidence has: its name is the time and a
    /// digest of the peer, the receipt id and the time.
    fn create(
        &self,
        peer: &EndpointId,
        receipt_id: &str,
        now_ms: u64,
    ) -> Result<(PathBuf, std::fs::File)> {
        for attempt in 0u8..16 {
            let mut digest = Sha256::new();
            digest.update(peer.as_bytes());
            digest.update((receipt_id.len() as u64).to_le_bytes());
            digest.update(receipt_id.as_bytes());
            digest.update(now_ms.to_le_bytes());
            digest.update([attempt]);
            let digest = hex::encode(&digest.finalize()[..16]);
            // the time first, in digits that sort as numbers do
            let path = self
                .directory
                .join(format!("{now_ms:020}-{digest}{SUFFIX}"));
            match private_file().open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(error).with_context(|| format!("create {}", path.display()));
                }
            }
        }
        bail!("no free evidence file name in {}", self.directory.display())
    }

    fn write<T: serde::Serialize>(&self, file: std::fs::File, evidence: &T) -> Result<()> {
        let mut out = Capped {
            inner: std::io::BufWriter::new(file),
            left: self.max_file_bytes,
        };
        serde_json::to_writer(&mut out, evidence)?;
        out.flush()?;
        Ok(())
    }

    /// Removes the oldest files until the directory is within its bounds.
    /// `kept` is the file just written, which stays.
    fn prune(&self, kept: &Path) {
        let Ok(entries) = std::fs::read_dir(&self.directory) else {
            return;
        };
        // names start with the time, so they sort oldest first
        let mut files: Vec<(PathBuf, u64)> = entries
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(SUFFIX))
            .filter_map(|entry| Some((entry.path(), entry.metadata().ok()?.len())))
            .collect();
        files.sort();
        let mut bytes: u64 = files.iter().map(|(_, len)| len).sum();
        let mut count = files.len();
        for (path, len) in &files {
            if count <= self.max_files && bytes <= self.max_bytes {
                break;
            }
            if path == kept {
                continue;
            }
            match std::fs::remove_file(path) {
                Ok(()) => {
                    count -= 1;
                    bytes = bytes.saturating_sub(*len);
                }
                Err(error) => {
                    tracing::warn!(
                        "old receipt evidence {} not removed: {error}",
                        path.display()
                    );
                }
            }
        }
    }
}

/// A writer that refuses to write past `left` bytes.
struct Capped<W> {
    inner: W,
    left: u64,
}

impl<W: Write> Write for Capped<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.left {
            return Err(std::io::Error::other(
                "the evidence is larger than an evidence file may be",
            ));
        }
        self.left -= bytes.len() as u64;
        self.inner.write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Creates `directory` for its owner alone, and takes access from others
/// when an earlier version made it with more.
fn create_private_dir(directory: &Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(directory)
        .with_context(|| format!("create {}", directory.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("make {} private", directory.display()))?;
    }
    Ok(())
}

/// Options that create a new file its owner alone reads and writes, and
/// never open one that exists.
fn private_file() -> std::fs::OpenOptions {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(seed: u8) -> EndpointId {
        EndpointId::from(iroh::SecretKey::from_bytes(&[seed; 32]).public())
    }

    fn store(state: &Path) -> EvidenceStore {
        EvidenceStore::under(Some(state)).expect("an absolute state directory")
    }

    fn names(store: &EvidenceStore) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&store.directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn evidence_is_kept_only_under_an_absolute_state_directory() {
        assert!(EvidenceStore::under(None).is_none());
        // without a home the state directory is `./.mesh-llm`: the working
        // directory is never where evidence goes
        assert!(EvidenceStore::under(Some(Path::new("./.mesh-llm"))).is_none());
        assert!(EvidenceStore::under(Some(Path::new(".mesh-llm"))).is_none());
        let state = tempfile::tempdir().unwrap();
        let store = store(state.path());
        assert_eq!(store.directory, state.path().join("receipt-evidence"));
    }

    #[test]
    fn a_file_is_named_by_a_digest_and_never_by_the_peers_id() {
        let state = tempfile::tempdir().unwrap();
        let store = store(state.path());
        let evidence = serde_json::json!({"opening": "abc"});
        // ids a peer would choose to break the write or leave the directory
        let long = "r".repeat(4096);
        for id in ["../../escape", "a/b\\c\n", long.as_str(), "", "."] {
            let path = store.keep(&peer(1), id, &evidence, 1_000).unwrap();
            assert_eq!(path.parent(), Some(store.directory.as_path()), "{id}");
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            assert_eq!(name.len(), 20 + 1 + 32 + SUFFIX.len(), "{name}");
            let digits = |text: &str| text.bytes().all(|b| b.is_ascii_hexdigit());
            assert!(digits(&name[..20]) && digits(&name[21..53]), "{name}");
            assert_eq!(
                std::fs::read(&path).unwrap(),
                br#"{"opening":"abc"}"#,
                "written compactly"
            );
        }
        assert_eq!(names(&store).len(), 5);
    }

    #[test]
    fn a_reused_id_never_replaces_earlier_evidence() {
        let state = tempfile::tempdir().unwrap();
        let store = store(state.path());
        // the same peer, id and time; another peer with the same id
        let first = store.keep(&peer(1), "r", &"first", 7).unwrap();
        let again = store.keep(&peer(1), "r", &"again", 7).unwrap();
        let other = store.keep(&peer(2), "r", &"other", 7).unwrap();
        let later = store.keep(&peer(1), "r", &"later", 8).unwrap();
        let all = [&first, &again, &other, &later];
        let distinct: std::collections::BTreeSet<_> = all.iter().collect();
        assert_eq!(distinct.len(), 4);
        assert_eq!(std::fs::read(&first).unwrap(), b"\"first\"");
        assert_eq!(std::fs::read(&again).unwrap(), b"\"again\"");
        assert_eq!(std::fs::read(&other).unwrap(), b"\"other\"");
    }

    #[cfg(unix)]
    #[test]
    fn the_directory_and_its_files_are_the_owners_alone() {
        use std::os::unix::fs::PermissionsExt;
        let state = tempfile::tempdir().unwrap();
        let store = store(state.path());
        // a directory an earlier version left open to others
        std::fs::create_dir(&store.directory).unwrap();
        std::fs::set_permissions(&store.directory, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = store.keep(&peer(1), "r", &"evidence", 1).unwrap();
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&store.directory), 0o700);
        assert_eq!(mode(&path), 0o600);
    }

    #[test]
    fn the_oldest_evidence_goes_first_past_the_count() {
        let state = tempfile::tempdir().unwrap();
        let store = EvidenceStore {
            max_files: 3,
            ..store(state.path())
        };
        let kept: Vec<PathBuf> = (1..=5u64)
            .map(|at| store.keep(&peer(1), "r", &at, at).unwrap())
            .collect();
        let left = names(&store);
        assert_eq!(left.len(), 3);
        for (at, path) in kept.iter().enumerate() {
            assert_eq!(path.exists(), at >= 2, "evidence {}", at + 1);
        }
        // a file that is not evidence is left alone
        std::fs::write(store.directory.join("notes.txt"), b"mine").unwrap();
        store.keep(&peer(1), "r", &6u64, 6).unwrap();
        assert!(store.directory.join("notes.txt").exists());
        assert_eq!(names(&store).len(), 4);
    }

    #[test]
    fn the_oldest_evidence_goes_first_past_the_bytes() {
        let state = tempfile::tempdir().unwrap();
        let store = EvidenceStore {
            max_bytes: 250,
            ..store(state.path())
        };
        let big = "x".repeat(98); // 100 bytes as JSON
        let first = store.keep(&peer(1), "a", &big, 1).unwrap();
        let second = store.keep(&peer(1), "b", &big, 2).unwrap();
        assert!(first.exists() && second.exists());
        let third = store.keep(&peer(1), "c", &big, 3).unwrap();
        assert!(!first.exists(), "300 bytes are past the bound");
        assert!(second.exists() && third.exists());
    }

    #[test]
    fn evidence_past_the_file_cap_is_not_kept() {
        let state = tempfile::tempdir().unwrap();
        let store = EvidenceStore {
            max_file_bytes: 64,
            ..store(state.path())
        };
        let error = store.keep(&peer(1), "r", &"x".repeat(1000), 1).unwrap_err();
        assert!(format!("{error:#}").contains("larger than"), "{error:#}");
        assert!(names(&store).is_empty(), "no partial file is left");
        store.keep(&peer(1), "r", &"small", 2).unwrap();
        assert_eq!(names(&store).len(), 1);
    }
}
