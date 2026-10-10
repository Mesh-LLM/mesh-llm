//! Bounded deletion of owned state after process and output handles close.
use std::{fs, io, path::Path};

pub(super) fn remove(path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        // Windows scanners can briefly retain a log handle after child exit.
        retry(
            || fs::remove_dir_all(path),
            || {
                std::thread::sleep(std::time::Duration::from_secs(1));
            },
        )
    }
    #[cfg(not(windows))]
    {
        fs::remove_dir_all(path)
    }
}

#[cfg(any(windows, test))]
fn retry(mut delete: impl FnMut() -> io::Result<()>, mut wait: impl FnMut()) -> io::Result<()> {
    for attempt in 1..=5 {
        match delete() {
            Ok(()) => return Ok(()),
            Err(error) if attempt == 5 => return Err(error),
            Err(_) => wait(),
        }
    }
    unreachable!("the final deletion attempt returns its result")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_lock_retries_before_removing_owned_log() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("owned-state");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("stdout.log"), b"readiness evidence").unwrap();
        let mut attempts = 0;
        let mut waits = 0;
        retry(
            || {
                attempts += 1;
                if attempts < 3 {
                    Err(io::Error::from_raw_os_error(32))
                } else {
                    fs::remove_dir_all(&root)
                }
            },
            || waits += 1,
        )
        .unwrap();
        assert_eq!((attempts, waits), (3, 2));
        assert!(!root.exists());
    }

    #[test]
    fn persistent_lock_fails_after_five_attempts_and_retains_evidence() {
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("stdout.log");
        fs::write(&log, b"readiness evidence").unwrap();
        let mut attempts = 0;
        let mut waits = 0;
        let error = retry(
            || {
                attempts += 1;
                Err(io::Error::from_raw_os_error(32))
            },
            || waits += 1,
        )
        .unwrap_err();
        assert_eq!((attempts, waits), (5, 4));
        assert_eq!(error.raw_os_error(), Some(32));
        assert_eq!(fs::read(log).unwrap(), b"readiness evidence");
    }

    #[test]
    fn immediate_deletion_success_does_not_wait() {
        retry(|| Ok(()), || panic!("successful cleanup must not wait")).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_transient_handle_lock_is_removed_after_release() {
        use std::os::windows::fs::OpenOptionsExt;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("owned-state");
        fs::create_dir(&root).unwrap();
        let log = root.join("stdout.log");
        fs::write(&log, b"readiness evidence").unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&log)
            .unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(handle);
        });
        let result = remove(&root);
        release.join().unwrap();
        result.unwrap();
        assert!(!root.exists());
    }

    #[cfg(windows)]
    #[test]
    fn windows_persistent_handle_lock_retains_log_and_reports_failure() {
        use std::os::windows::fs::OpenOptionsExt;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("owned-state");
        fs::create_dir(&root).unwrap();
        let log = root.join("stdout.log");
        fs::write(&log, b"readiness evidence").unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&log)
            .unwrap();
        let result = remove(&root);
        drop(handle);
        assert!(result.is_err());
        assert_eq!(fs::read(&log).unwrap(), b"readiness evidence");
        fs::remove_dir_all(root).unwrap();
    }
}
