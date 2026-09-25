//! Filesystem measurement for `ci-ops build-cache`: the legacy
//! `tree_metrics` (`os.walk(followlinks=False)` with `lstat` sizes),
//! `immediate_entries`, `artifact_roots` globbing and the guarded
//! `remove_tree`.

use crate::ci_operations::build_cache_values::resolve;
use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

/// Python's `st_mtime`: `sec + nsec * 1e-9` in binary64.
pub(crate) fn mtime(metadata: &Metadata) -> f64 {
    metadata.mtime() as f64 + metadata.mtime_nsec() as f64 * 1e-9
}

fn size(metadata: &Metadata) -> i128 {
    i128::from(metadata.size())
}

/// `(total lstat bytes, newest mtime)` for `path`; `(0, 0.0)` when missing.
pub(crate) fn tree_metrics(path: &Path) -> (i128, f64) {
    let Ok(followed) = std::fs::metadata(path) else {
        return (0, 0.0);
    };
    let is_symlink = path.is_symlink();
    if followed.is_file() || is_symlink {
        return std::fs::symlink_metadata(path)
            .map_or((0, 0.0), |stat| (size(&stat), mtime(&stat)));
    }
    let mut total = 0;
    let mut newest = mtime(&followed);
    walk(path, &mut total, &mut newest);
    (total, newest)
}

/// One `os.walk` level: symlinked directories are counted but not entered,
/// unreadable directories are skipped like `os.walk` without `onerror`.
fn walk(directory: &Path, total: &mut i128, newest: &mut f64) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(stat) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if stat.is_dir() {
            walk(&path, total, newest);
            continue;
        }
        *total += size(&stat);
        *newest = newest.max(mtime(&stat));
    }
}

pub(crate) struct Entry {
    pub(crate) path: PathBuf,
    pub(crate) bytes: i128,
    pub(crate) newest: f64,
}

/// Directory entries in `readdir` order (as `Path.iterdir`).
pub(crate) fn children(path: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(path).map_or_else(
        |_| Vec::new(),
        |entries| entries.flatten().map(|entry| entry.path()).collect(),
    )
}

/// `immediate_entries`: children by size, largest first, ties in
/// directory order.
pub(crate) fn immediate_entries(path: &Path) -> Vec<Entry> {
    let mut entries: Vec<Entry> = if path.is_dir() {
        children(path)
            .into_iter()
            .map(|child| {
                let (bytes, newest) = tree_metrics(&child);
                Entry {
                    path: child,
                    bytes,
                    newest,
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.bytes));
    entries
}

fn subdirectories(path: &Path) -> Vec<PathBuf> {
    children(path)
        .into_iter()
        .filter(|child| child.is_dir())
        .collect()
}

/// `artifact_roots`: `*/leaf` and `*/*/leaf` real directories, sorted by
/// path components. Wildcard levels follow directory symlinks, as
/// `pathlib` globbing does.
pub(crate) fn artifact_roots(target: &Path, leaf: &str) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for first in subdirectories(target) {
        roots.push(first.join(leaf));
    }
    for first in subdirectories(target) {
        for second in subdirectories(&first) {
            roots.push(second.join(leaf));
        }
    }
    let mut roots: Vec<PathBuf> = roots
        .into_iter()
        .filter(|root| root.is_dir() && !root.is_symlink())
        .collect();
    roots.sort();
    roots
}

/// `remove_tree`: only strict descendants of `target`, never through a
/// parent that resolves outside it; a symlink is unlinked, not followed.
pub(crate) fn remove_tree(path: &Path, target: &Path) -> Result<(), String> {
    if path == target || !path.starts_with(target) {
        return Err(format!(
            "refusing to remove path outside target: {}",
            path.display()
        ));
    }
    if path.is_symlink() {
        return std::fs::remove_file(path).map_err(|error| io_text(&error, path));
    }
    let target_resolved = resolve(target);
    let parent_resolved = resolve(path.parent().unwrap_or(path));
    if !parent_resolved.starts_with(&target_resolved) {
        return Err(format!(
            "refusing to remove path through a parent outside target: {}",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!(
            "refusing to remove non-directory path: {}",
            path.display()
        ));
    }
    std::fs::remove_dir_all(path).map_err(|error| io_text(&error, path))
}

/// Python's `OSError.__str__` for a failed operation on `path`.
pub(crate) fn io_text(error: &std::io::Error, path: &Path) -> String {
    crate::ci_plan::catalog::os_error_text(error, &path.to_string_lossy())
}
