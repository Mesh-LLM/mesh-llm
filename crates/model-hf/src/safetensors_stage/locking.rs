use std::{
    fs::{self, File, OpenOptions},
    path::Path,
};

use anyhow::{Context, Result};

pub(super) struct CacheKeyLock {
    #[allow(dead_code)]
    file: File,
}

impl CacheKeyLock {
    pub(super) fn acquire(cache_root: &Path, cache_key: &str) -> Result<Self> {
        fs::create_dir_all(cache_root)
            .with_context(|| format!("create SafeTensors stage cache {}", cache_root.display()))?;
        let path = cache_root.join(format!(".{cache_key}.lock"));
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open SafeTensors stage cache lock {}", path.display()))?;
        lock_file(&file)
            .with_context(|| format!("lock SafeTensors stage cache key {cache_key}"))?;
        Ok(Self { file })
    }
}

impl Drop for CacheKeyLock {
    fn drop(&mut self) {
        unlock_file(&self.file);
    }
}

pub(super) struct AdvisoryFileLock {
    file: File,
}

impl AdvisoryFileLock {
    pub(super) fn acquire(path: &Path) -> Result<Self> {
        let file = open_lock_file(path)?;
        lock_file(&file).with_context(|| format!("lock {}", path.display()))?;
        Ok(Self { file })
    }

    pub(super) fn try_acquire(path: &Path) -> Result<Option<Self>> {
        let file = open_lock_file(path)?;
        if try_lock_file(&file).with_context(|| format!("try lock {}", path.display()))? {
            Ok(Some(Self { file }))
        } else {
            Ok(None)
        }
    }
}

impl Drop for AdvisoryFileLock {
    fn drop(&mut self) {
        unlock_file(&self.file);
    }
}

fn open_lock_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open lock file {}", path.display()))
}

#[cfg(unix)]
fn lock_file(file: &File) -> Result<()> {
    use std::os::fd::AsRawFd;

    // SAFETY: `file` owns a valid descriptor for the duration of this call.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).context("flock failed")
    }
}

#[cfg(unix)]
fn try_lock_file(file: &File) -> Result<bool> {
    use std::os::fd::AsRawFd;

    // SAFETY: `file` owns a valid descriptor for the duration of this call.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if error
        .raw_os_error()
        .is_some_and(|code| code == libc::EWOULDBLOCK || code == libc::EAGAIN)
    {
        Ok(false)
    } else {
        Err(error).context("nonblocking flock failed")
    }
}

#[cfg(not(unix))]
fn lock_file(_file: &File) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn try_lock_file(_file: &File) -> Result<bool> {
    // No portable advisory lock is available here, so never treat a visit as
    // abandoned and risk removing another process's active tensor.
    Ok(false)
}

#[cfg(unix)]
fn unlock_file(file: &File) {
    use std::os::fd::AsRawFd;

    // SAFETY: `file` still owns the descriptor locked by `lock_file`.
    let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
}

#[cfg(not(unix))]
fn unlock_file(_file: &File) {}
