//! Removal of runtime slots left by stopped instances.
//!
//! Every non-client run creates `{runtime_root}/{pid}/` (see [`super::instance`])
//! and nothing removed it when the process ended, so the root grew by one
//! directory per run, and `mesh-llm stop` and `uninstall` printed one line for
//! each of them. A slot holds that run's native logs, which are what a person
//! reads after a crash, so a dead slot is kept for [`DEAD_SLOT_RETENTION`]
//! after its last write and removed after that.
//!
//! Only a directory with the `lock` file every instance opens counts as a
//! slot, and a slot is removed only once claimed through that lock, so an
//! instance that starts under a reused pid is never stripped of its files.

use super::instance::validate::{self, PidIdentity};
use serde::Deserialize;
use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a stopped instance's slot, and so its logs, stays on disk.
pub(crate) const DEAD_SLOT_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// The `owner.json` fields that identify the process that wrote it.
#[derive(Deserialize)]
struct SlotOwner {
    pid: u32,
    started_at_unix: Option<i64>,
    mesh_llm_binary: Option<String>,
}

/// Removes the slots under `root` whose owner is no longer running and whose
/// last write is older than `retention` at `now`. Returns how many it removed.
///
/// A slot is only removed when its process is gone, or its id now belongs to
/// another process; a slot whose owner could not be read is kept, as is the
/// caller's own. Every failure is logged and skipped: this never stops a run.
pub(crate) fn prune_dead_runtime_slots(
    root: &Path,
    my_pid: u32,
    now: SystemTime,
    retention: Duration,
) -> usize {
    let Ok(entries) = fs::read_dir(root) else {
        return 0;
    };
    let removed = entries
        .flatten()
        .filter(|entry| prune_slot(root, &entry.path(), my_pid, now, retention))
        .count();
    if removed > 0 {
        tracing::debug!(removed, root = %root.display(), "removed stopped instances' runtime directories");
    }
    removed
}

/// Prunes one entry of the runtime root, and says whether it removed a slot.
fn prune_slot(root: &Path, dir: &Path, my_pid: u32, now: SystemTime, retention: Duration) -> bool {
    if is_renamed_for_removal(dir) {
        // A cleanup renamed it, then could not remove all of it.
        if let Err(error) = fs::remove_dir_all(dir) {
            tracing::debug!(
                path = %dir.display(),
                %error,
                "could not finish removing a stopped instance's runtime directory"
            );
        }
        return false;
    }
    let Some(dir_pid) = dir
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.parse::<u32>().ok())
    else {
        return false;
    };
    if dir_pid == my_pid || !is_instance_slot(dir) || owner_may_be_running(dir, dir_pid) {
        return false;
    }
    let Some(last_write) = newest_write(dir) else {
        return false;
    };
    if now.duration_since(last_write).unwrap_or_default() < retention {
        return false;
    }
    match remove_unclaimed_slot(root, dir, dir_pid, last_write) {
        Ok(removed) => removed,
        Err(error) => {
            tracing::debug!(
                path = %dir.display(),
                %error,
                "could not remove a stopped instance's runtime directory"
            );
            false
        }
    }
}

/// Whether `dir` is a slot an instance created: it holds the `lock` file
/// every instance opens, and nothing but what an instance writes there
/// (`owner.json`, the temporary file it is written through, and `logs/`),
/// each of the kind the instance makes, symlinks never counting. Any other
/// numeric directory under a custom root, such as an archive named after a
/// year, is not ours to remove.
fn is_instance_slot(dir: &Path) -> bool {
    let kind = |path: &Path| fs::symlink_metadata(path).map(|meta| meta.file_type());
    if !kind(&dir.join("lock")).is_ok_and(|kind| kind.is_file()) {
        return false;
    }
    let Ok(mut entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.all(|entry| {
        entry.is_ok_and(|entry| {
            let Ok(kind) = kind(&entry.path()) else {
                return false;
            };
            match entry.file_name().to_str() {
                Some("lock" | "owner.json" | "owner.json.tmp") => kind.is_file(),
                Some("logs") => kind.is_dir(),
                _ => false,
            }
        })
    })
}

/// A slot a cleanup renamed out of the way before removing it.
fn is_renamed_for_removal(dir: &Path) -> bool {
    dir.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(REMOVING_PREFIX))
        && fs::symlink_metadata(dir).is_ok_and(|meta| meta.is_dir())
}

const REMOVING_PREFIX: &str = ".removing-";

/// Removes `dir` unless an instance holds it, and says whether it did.
///
/// The slot is claimed through its lock first: the flock an instance takes on
/// Unix, or an exclusive open that fails while an instance keeps the file open
/// on Windows. It is then renamed out of the way and removed, so an instance
/// starting under the same pid meanwhile finds no slot and creates a fresh
/// one instead of losing its files. On Windows a directory with an open file
/// cannot be renamed, so the claim is released first and a rename that fails
/// means an instance has opened the lock since: the slot is kept. An instance
/// could also open the lock, write its files and exit before the rename, so a
/// slot with a file written after `last_write`, the write the age check saw,
/// is put back rather than removed.
fn remove_unclaimed_slot(
    root: &Path,
    dir: &Path,
    pid: u32,
    last_write: SystemTime,
) -> io::Result<bool> {
    let Some(claim) = claim_slot(&dir.join("lock"))? else {
        return Ok(false);
    };
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let removing = root.join(format!("{REMOVING_PREFIX}{pid}-{nanos}"));
    #[cfg(windows)]
    drop(claim);
    let renamed = fs::rename(dir, &removing);
    #[cfg(not(windows))]
    drop(claim);
    if let Err(error) = renamed {
        // Expected while an instance holds the slot on Windows; anything else
        // (a root that refuses renames, say) is worth a line.
        tracing::debug!(path = %dir.display(), %error, "kept a stopped instance's runtime directory: rename failed");
        return Ok(false);
    }
    if newest_entry_write(&removing).is_some_and(|newest| newest > last_write) {
        // Back where its instance expects it, unless one has recreated it.
        let _ = fs::rename(&removing, dir);
        return Ok(false);
    }
    fs::remove_dir_all(&removing)?;
    Ok(true)
}

/// The lock of a slot no instance holds, held until dropped; `None` when an
/// instance holds it or it has gone.
#[cfg(unix)]
fn claim_slot(lock: &Path) -> io::Result<Option<fs::File>> {
    let file = match fs::OpenOptions::new().read(true).write(true).open(lock) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    claim_opened_lock(file, lock)
}

/// Takes the flock on `file`, opened from `lock`, and keeps it only while
/// `lock` still names that file. Another cleanup may have claimed the file
/// first, renamed its slot away and released it, and an instance created a
/// new slot at the same path since: the lock held is then not the slot's.
#[cfg(unix)]
fn claim_opened_lock(file: fs::File, lock: &Path) -> io::Result<Option<fs::File>> {
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::io::AsRawFd;
    // SAFETY: flock is safe to call with a valid fd.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Ok(None);
        }
        return Err(error);
    }
    let held = file.metadata()?;
    let still_here = fs::metadata(lock)
        .is_ok_and(|current| current.dev() == held.dev() && current.ino() == held.ino());
    Ok(still_here.then_some(file))
}

#[cfg(windows)]
fn claim_slot(lock: &Path) -> io::Result<Option<fs::File>> {
    use std::os::windows::fs::OpenOptionsExt;
    const ERROR_SHARING_VIOLATION: i32 = 32;
    match fs::OpenOptions::new().read(true).share_mode(0).open(lock) {
        Ok(file) => Ok(Some(file)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) if error.raw_os_error() == Some(ERROR_SHARING_VIOLATION) => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(not(any(unix, windows)))]
fn claim_slot(_lock: &Path) -> io::Result<Option<fs::File>> {
    Ok(None)
}

/// Whether the process that owns this slot may still be running. Only a
/// process that is gone, or an id now held by another process, answers no.
/// An `owner.json` that is present but unreadable or invalid establishes
/// nothing, so the slot is kept; only an absent one falls back to the pid.
fn owner_may_be_running(dir: &Path, dir_pid: u32) -> bool {
    let owner = match fs::read_to_string(dir.join("owner.json")) {
        Ok(json) => match serde_json::from_str::<SlotOwner>(&json) {
            Ok(owner) => Some(owner),
            Err(_) => return true,
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(_) => return true,
    };
    let (pid, comm, started_at) = match owner {
        Some(owner) => (
            owner.pid,
            owner
                .mesh_llm_binary
                .as_deref()
                .and_then(super::instance::binary_process_name)
                .unwrap_or_else(|| "mesh-llm".to_string()),
            // Builds that could not read their own start time recorded 0.
            owner.started_at_unix.filter(|started| *started > 0),
        ),
        // A run that ended before writing owner.json: only its id is known.
        None => (dir_pid, "mesh-llm".to_string(), None),
    };
    !matches!(
        validate::pid_identity(pid, &comm, started_at),
        PidIdentity::Gone | PidIdentity::Other
    )
}

/// The latest modification time of the slot directory and the files in it and
/// in its `logs/` directory, the places a running instance writes.
fn newest_write(dir: &Path) -> Option<SystemTime> {
    let own = fs::metadata(dir).and_then(|meta| meta.modified()).ok();
    match (own, newest_entry_write(dir)) {
        (Some(own), Some(entries)) => Some(own.max(entries)),
        (own, entries) => own.or(entries),
    }
}

/// The newest modification time of the files in `dir` and `dir/logs`, leaving
/// out `dir` itself, whose time a rename may change.
fn newest_entry_write(dir: &Path) -> Option<SystemTime> {
    let modified = |path: &Path| fs::metadata(path).and_then(|meta| meta.modified()).ok();
    let mut newest: Option<SystemTime> = None;
    for scanned in [dir.to_path_buf(), dir.join("logs")] {
        let Ok(entries) = fs::read_dir(&scanned) else {
            continue;
        };
        for entry in entries.flatten() {
            if let Some(time) = modified(&entry.path()) {
                newest = Some(newest.map_or(time, |current| current.max(time)));
            }
        }
    }
    newest
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Not a valid id on Windows (ids are multiples of 4) and far above the
    // usual Linux and macOS ranges, as in the instance scan tests.
    const DEAD_PID: u32 = 999_999;
    const WEEK_AND_A_DAY: Duration = Duration::from_secs(8 * 24 * 60 * 60);

    fn slot(root: &Path, pid: u32, owner: Option<serde_json::Value>) -> std::path::PathBuf {
        let dir = root.join(pid.to_string());
        fs::create_dir_all(dir.join("logs")).unwrap();
        fs::write(dir.join("lock"), "").unwrap();
        fs::write(dir.join("logs").join("skippy-native.log"), "native log").unwrap();
        if let Some(owner) = owner {
            fs::write(dir.join("owner.json"), owner.to_string()).unwrap();
        }
        dir
    }

    fn dead_owner(pid: u32) -> serde_json::Value {
        serde_json::json!({
            "pid": pid,
            "started_at_unix": 1_700_000_000,
            "mesh_llm_binary": "/usr/bin/mesh-llm",
        })
    }

    #[test]
    fn removes_a_stopped_instance_slot_after_the_retention() {
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, Some(dead_owner(DEAD_PID)));

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        let removed = prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION);

        assert_eq!(removed, 1);
        assert!(!dir.exists());
    }

    #[test]
    fn keeps_a_stopped_instance_slot_within_the_retention() {
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, Some(dead_owner(DEAD_PID)));

        let removed =
            prune_dead_runtime_slots(root.path(), 1, SystemTime::now(), DEAD_SLOT_RETENTION);

        assert_eq!(removed, 0);
        assert!(dir.join("logs").join("skippy-native.log").exists());
    }

    #[test]
    fn removes_an_old_slot_that_never_wrote_owner_json() {
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, None);

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            1
        );
        assert!(!dir.exists());
    }

    #[test]
    fn keeps_the_slot_of_a_running_owner_however_old() {
        let root = tempdir().unwrap();
        let my_pid = std::process::id();
        let running = serde_json::json!({
            "pid": my_pid,
            "started_at_unix": validate::current_process_start_time_unix().unwrap(),
            "mesh_llm_binary": std::env::current_exe().unwrap().to_string_lossy(),
        });
        // Under another directory name, so only the owner check keeps it.
        let dir = slot(root.path(), DEAD_PID, Some(running));

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(dir.exists());
    }

    #[test]
    fn keeps_the_callers_own_slot_and_other_entries() {
        let root = tempdir().unwrap();
        let own = slot(root.path(), 4242, Some(dead_owner(4242)));
        let stray = root.path().join("not-a-pid");
        fs::create_dir_all(&stray).unwrap();
        fs::write(root.path().join("12345"), "a file, not a slot").unwrap();

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 4242, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(own.exists());
        assert!(stray.exists());
        assert!(root.path().join("12345").exists());
    }

    #[test]
    fn keeps_a_numeric_directory_that_is_not_an_instance_slot() {
        // An archive named after a number, under a custom runtime root.
        let root = tempdir().unwrap();
        let archive = root.path().join(DEAD_PID.to_string());
        fs::create_dir_all(&archive).unwrap();
        fs::write(archive.join("notes.txt"), "kept").unwrap();

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(archive.join("notes.txt").exists());
    }

    #[test]
    fn keeps_a_numeric_directory_holding_files_no_instance_writes() {
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, Some(dead_owner(DEAD_PID)));
        fs::write(dir.join("report.pdf"), "kept").unwrap();

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(dir.join("report.pdf").exists());
    }

    #[test]
    fn keeps_a_slot_whose_owner_record_cannot_be_read() {
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, None);
        fs::write(dir.join("owner.json"), "{\"pid\": 9").unwrap();

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(dir.join("owner.json").exists());
        assert!(dir.join("logs").join("skippy-native.log").exists());
    }

    #[test]
    fn keeps_an_expired_slot_whose_lock_a_live_instance_holds() {
        // A new instance under a reused pid holds the lock of an old slot.
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, Some(dead_owner(DEAD_PID)));
        let held = hold_lock(&dir.join("lock"));

        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(dir.join("logs").join("skippy-native.log").exists());
        drop(held);
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            1
        );
        assert!(!dir.exists());
    }

    #[test]
    fn a_native_log_written_within_the_retention_keeps_the_slot() {
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, Some(dead_owner(DEAD_PID)));
        let created = SystemTime::now();
        // Eight days on, the slot directory, its owner record and logs/ are
        // past the retention, but the native log was appended two days ago.
        let later = created + WEEK_AND_A_DAY;
        let appended = created + Duration::from_secs(6 * 24 * 60 * 60);
        let log = fs::OpenOptions::new()
            .append(true)
            .open(dir.join("logs").join("skippy-native.log"))
            .unwrap();
        log.set_modified(appended).unwrap();
        drop(log);

        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(dir.join("logs").join("skippy-native.log").exists());
    }

    #[test]
    fn a_slot_written_after_the_age_check_is_put_back() {
        // On Windows an instance can open the lock, write its owner record
        // and exit between the release of the claim and the rename.
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, Some(dead_owner(DEAD_PID)));
        let checked = newest_write(&dir).unwrap() - Duration::from_secs(60);

        assert!(!remove_unclaimed_slot(root.path(), &dir, DEAD_PID, checked).unwrap());
        assert!(dir.join("logs").join("skippy-native.log").exists());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);

        let unchanged = newest_write(&dir).unwrap();
        assert!(remove_unclaimed_slot(root.path(), &dir, DEAD_PID, unchanged).unwrap());
        assert!(!dir.exists());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn keeps_a_numeric_directory_whose_entries_are_not_of_the_kind_an_instance_writes() {
        let root = tempdir().unwrap();
        let dir = root.path().join(DEAD_PID.to_string());
        fs::create_dir_all(dir.join("owner.json.tmp")).unwrap();
        fs::write(dir.join("lock"), "").unwrap();
        let later = SystemTime::now() + WEEK_AND_A_DAY;
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(dir.join("owner.json.tmp").is_dir());

        // `logs` as a file is not an instance's either.
        fs::remove_dir(dir.join("owner.json.tmp")).unwrap();
        fs::write(dir.join("logs"), "").unwrap();
        assert_eq!(
            prune_dead_runtime_slots(root.path(), 1, later, DEAD_SLOT_RETENTION),
            0
        );
        assert!(dir.join("logs").is_file());
    }

    #[test]
    fn finishes_removing_a_slot_an_earlier_cleanup_renamed() {
        let root = tempdir().unwrap();
        let renamed = root.path().join(format!("{REMOVING_PREFIX}{DEAD_PID}-1"));
        fs::create_dir_all(renamed.join("logs")).unwrap();
        fs::write(renamed.join("logs").join("skippy-native.log"), "left").unwrap();
        // Something else whose name only starts the same way is a file.
        let file = root.path().join(format!("{REMOVING_PREFIX}notes"));
        fs::write(&file, "kept").unwrap();

        prune_dead_runtime_slots(root.path(), 1, SystemTime::now(), DEAD_SLOT_RETENTION);
        assert!(!renamed.exists());
        assert!(file.exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_lock_replaced_after_it_was_opened_is_not_claimed() {
        let root = tempdir().unwrap();
        let dir = slot(root.path(), DEAD_PID, Some(dead_owner(DEAD_PID)));
        let lock = dir.join("lock");
        let open = || {
            fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&lock)
                .unwrap()
        };
        let opened = open();
        // Meanwhile the slot went and a new instance's lock took its path.
        fs::remove_file(&lock).unwrap();
        fs::write(&lock, "").unwrap();
        assert!(claim_opened_lock(opened, &lock).unwrap().is_none());
        assert!(claim_opened_lock(open(), &lock).unwrap().is_some());
    }

    /// Holds a slot lock the way a running instance does.
    fn hold_lock(lock: &Path) -> fs::File {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(lock)
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: flock is safe to call with a valid fd.
            assert_eq!(
                unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
                0
            );
        }
        file
    }

    #[test]
    fn a_missing_root_removes_nothing() {
        let root = tempdir().unwrap();
        let missing = root.path().join("absent");
        assert_eq!(
            prune_dead_runtime_slots(&missing, 1, SystemTime::now(), DEAD_SLOT_RETENTION),
            0
        );
    }
}
