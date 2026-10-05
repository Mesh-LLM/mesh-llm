use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::output::print_path_event;

pub fn with_manifest_lock<T>(
    manifest_path: &Path,
    action: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _guard = ManifestLock::acquire(manifest_path)?;
    action()
}

struct ManifestLock {
    path: PathBuf,
    #[allow(dead_code)]
    file: File,
}

impl ManifestLock {
    fn acquire(manifest_path: &Path) -> Result<Self> {
        let path = lock_path(manifest_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open manifest lock {}", path.display()))?;
        lock_file(&file).with_context(|| format!("lock manifest {}", manifest_path.display()))?;
        print_path_event("🔒", "Manifest lock acquired", &path);
        Ok(Self { path, file })
    }
}

impl Drop for ManifestLock {
    fn drop(&mut self) {
        unlock_file(&self.file);
        print_path_event("🔓", "Manifest lock released", &self.path);
    }
}

fn lock_path(manifest_path: &Path) -> PathBuf {
    let mut lock_name = manifest_path.as_os_str().to_os_string();
    lock_name.push(".lock");
    PathBuf::from(lock_name)
}

#[cfg(unix)]
fn lock_file(file: &File) -> Result<()> {
    use std::io;
    use std::os::fd::AsRawFd;

    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 {
        return Ok(());
    }
    let err = io::Error::last_os_error();
    if err.kind() == io::ErrorKind::WouldBlock {
        anyhow::bail!("another skippy-quantize process holds this manifest lock");
    }
    Err(err).context("flock failed")
}

#[cfg(not(unix))]
fn lock_file(_file: &File) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn unlock_file(file: &File) {
    use std::os::fd::AsRawFd;

    let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
}

#[cfg(not(unix))]
fn unlock_file(_file: &File) {}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::records::unix_timestamp_ms;

    use super::*;

    #[test]
    fn manifest_lock_holds_during_action() {
        let root =
            std::env::temp_dir().join(format!("skippy-quantize-lock-test-{}", unix_timestamp_ms()));
        let manifest = root.join("job.json");
        let calls = AtomicUsize::new(0);

        with_manifest_lock(&manifest, || {
            calls.fetch_add(1, Ordering::SeqCst);
            assert!(lock_path(&manifest).is_file());
            Ok(())
        })
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
