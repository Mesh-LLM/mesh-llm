//! One writer at a time in a plugin store.
//!
//! Every write to installed packages, their metadata or the set offer record
//! takes this lock, so a check made under it (is this plugin installed? what
//! does the offer record say?) still holds when the write lands. Writers hold
//! it only for the write itself, never across a download.
use std::{fs, fs::OpenOptions, path::Path};

use anyhow::{Context, Result};

const LOCK_FILE: &str = ".write.lock";

/// Held until dropped. Advisory: it orders plugin store writers, which all
/// take it; readers do not.
#[derive(Debug)]
pub struct StoreWriteLock {
    file: fs::File,
}

impl StoreWriteLock {
    /// Wait for the store's write lock. A lock is per open file, so a holder
    /// must not take it again: the second wait would never end.
    pub(crate) fn acquire(store_root: &Path) -> Result<Self> {
        fs::create_dir_all(store_root)
            .with_context(|| format!("create plugin store {}", store_root.display()))?;
        let path = store_root.join(LOCK_FILE);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("open plugin store lock {}", path.display()))?;
        fs2::FileExt::lock_exclusive(&file)
            .with_context(|| format!("lock plugin store {}", path.display()))?;
        Ok(Self { file })
    }
}

impl Drop for StoreWriteLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, thread, time::Duration};

    #[test]
    fn a_second_writer_waits_for_the_first() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let first = StoreWriteLock::acquire(dir.path())?;
        let root = dir.path().to_path_buf();
        let (sent, received) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let lock = StoreWriteLock::acquire(&root);
            sent.send(()).ok();
            lock.map(drop)
        });
        // The waiter cannot get the lock while `first` holds it.
        assert!(received.recv_timeout(Duration::from_millis(200)).is_err());
        drop(first);
        received.recv_timeout(Duration::from_secs(60))?;
        waiter.join().expect("waiter thread")?;
        Ok(())
    }
}
