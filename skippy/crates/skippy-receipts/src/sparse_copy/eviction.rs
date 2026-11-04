//! Keeps the sparse copies under a root within a size: the copies used
//! longest ago go first, and a copy that any opener holds open stays.
//!
//! A copy's size is the bytes its record says it holds, which is what it
//! takes on disk. Its last use is when an opener last opened it, which
//! [`SparseCopy::open`](super::SparseCopy::open) stamps on its lock file.
//! A copy is removed only under the exclusive lock of that file, which no
//! one gets while an opener holds the shared one. The record goes first,
//! then the data, so a crash leaves no record of bytes that are gone. The
//! lock file stays, so an opener waiting on it finds the copy gone and
//! starts it over.

use std::fs::{OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};

use super::{Record, sidecar};

/// How deep under the root a copy may be: the repo, the commit and the
/// parts of the file's path.
const MAX_DEPTH: usize = 24;

/// What an eviction did.
#[derive(Debug, PartialEq, Eq)]
pub struct Evicted {
    /// Copies removed.
    pub removed: usize,
    /// Bytes the copies under the root hold after it.
    pub held: u64,
}

struct Copy {
    path: PathBuf,
    held: u64,
    used: SystemTime,
}

/// Removes the copies under `root` used longest ago until they hold at most
/// `max_bytes`, except `keep` and the copies that are open.
pub fn evict(root: &Path, max_bytes: u64, keep: &Path) -> Result<Evicted> {
    let mut copies = Vec::new();
    collect(root, 0, &mut copies)?;
    let mut held = copies
        .iter()
        .fold(0u64, |sum, c| sum.saturating_add(c.held));
    copies.sort_by_key(|copy| copy.used);
    let mut removed = 0;
    for copy in copies {
        if held <= max_bytes {
            break;
        }
        if copy.path == keep {
            continue;
        }
        if remove_unused(&copy.path)? {
            held = held.saturating_sub(copy.held);
            removed += 1;
        }
    }
    Ok(Evicted { removed, held })
}

/// The copies under `dir`: each data file with a record next to it. Links
/// are not followed.
fn collect(dir: &Path, depth: usize, copies: &mut Vec<Copy>) -> Result<()> {
    if depth > MAX_DEPTH {
        return Ok(());
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("list {}", dir.display())),
    };
    for entry in entries {
        let entry = entry.with_context(|| format!("list {}", dir.display()))?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            collect(&path, depth + 1, copies)?;
            continue;
        }
        let name = entry.file_name();
        let Some(data) = name.to_str().and_then(|n| n.strip_suffix(".ranges")) else {
            continue;
        };
        if kind.is_file() {
            copies.extend(copy_of(&path.with_file_name(data), &path));
        }
    }
    Ok(())
}

/// The copy whose record is at `record`, when the record reads.
fn copy_of(path: &Path, record: &Path) -> Option<Copy> {
    let bytes = std::fs::read(record).ok()?;
    let read: Record = serde_json::from_slice(&bytes).ok()?;
    let held = read.ranges.iter().fold(0u64, |sum, (start, end)| {
        sum.saturating_add(end.saturating_sub(*start))
    });
    let stamp = |file: &Path| std::fs::metadata(file).and_then(|m| m.modified()).ok();
    let used = stamp(&sidecar(path, "lock"))
        .or_else(|| stamp(record))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    Some(Copy {
        path: path.to_owned(),
        held,
        used,
    })
}

/// Removes the copy at `path` unless an opener holds it: whether it did.
fn remove_unused(path: &Path) -> Result<bool> {
    let lock_path = sidecar(path, "lock");
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("open {}", lock_path.display()))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return Ok(false),
        Err(TryLockError::Error(error)) => {
            return Err(error).with_context(|| format!("lock {}", lock_path.display()));
        }
    }
    for file in [sidecar(path, "ranges"), path.to_owned()] {
        match std::fs::remove_file(&file) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("remove {}", file.display()));
            }
        }
    }
    lock.unlock()
        .with_context(|| format!("unlock {}", lock_path.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::{RangeFetch, RemoteIdentity, SparseCopy};
    use super::*;
    use crate::weights::ByteSource;

    struct Zeros;

    impl RangeFetch for Zeros {
        fn fetch(&self, _offset: u64, len: u64) -> Result<Vec<u8>> {
            Ok(vec![7; len as usize])
        }
    }

    /// A copy at `root/name` of a file of `size` bytes that holds `held`
    /// of them, last used `age` seconds ago.
    fn copy(root: &Path, name: &str, size: u64, held: u64, age: u64) -> SparseCopy {
        let path = root.join(name).join("model.gguf");
        let identity = RemoteIdentity {
            source: name.into(),
            size,
        };
        let copy = SparseCopy::open(&path, identity, Box::new(Zeros)).unwrap();
        copy.fetch(&[(0, held)]).unwrap();
        let used = SystemTime::now() - Duration::from_secs(age);
        let lock = OpenOptions::new()
            .write(true)
            .open(sidecar(&path, "lock"))
            .unwrap();
        lock.set_modified(used).unwrap();
        copy
    }

    #[test]
    fn the_copies_used_longest_ago_go_first() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for (name, age) in [("old", 300), ("mid", 200), ("new", 100)] {
            drop(copy(root, name, 1 << 20, 1000, age));
        }
        let evicted = evict(root, 2000, Path::new("")).unwrap();
        assert_eq!(
            evicted,
            Evicted {
                removed: 1,
                held: 2000
            }
        );
        assert!(!root.join("old/model.gguf").exists());
        assert!(!root.join("old/model.gguf.ranges").exists());
        assert!(root.join("mid/model.gguf").exists());
        // within the cap, nothing goes
        assert_eq!(evict(root, 2000, Path::new("")).unwrap().removed, 0);
        // a copy removed starts over when it is opened again
        let again = copy(root, "old", 1 << 20, 10, 0);
        assert_eq!(again.held_bytes(), 10);
    }

    #[test]
    fn an_open_copy_and_the_one_to_keep_stay() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let open = copy(root, "open", 1 << 20, 1000, 300);
        drop(copy(root, "kept", 1 << 20, 1000, 200));
        let evicted = evict(root, 0, &root.join("kept/model.gguf")).unwrap();
        assert_eq!(
            evicted,
            Evicted {
                removed: 0,
                held: 2000
            }
        );
        assert_eq!(open.read_at(0, 4).unwrap(), vec![7; 4]);
        // once closed, it goes
        drop(open);
        let evicted = evict(root, 0, &root.join("kept/model.gguf")).unwrap();
        assert_eq!(
            evicted,
            Evicted {
                removed: 1,
                held: 1000
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_root_is_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        drop(copy(elsewhere.path(), "theirs", 1 << 20, 1000, 300));
        std::os::unix::fs::symlink(elsewhere.path(), dir.path().join("link")).unwrap();
        let evicted = evict(dir.path(), 0, Path::new("")).unwrap();
        assert_eq!(
            evicted,
            Evicted {
                removed: 0,
                held: 0
            }
        );
        assert!(elsewhere.path().join("theirs/model.gguf").exists());
    }
}
