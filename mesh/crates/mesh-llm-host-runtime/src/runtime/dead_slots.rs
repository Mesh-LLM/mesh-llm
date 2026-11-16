//! Removal of runtime slots left by stopped instances.
//!
//! Every non-client run creates `{runtime_root}/{pid}/` (see [`super::instance`])
//! and nothing removed it when the process ended, so the root grew by one
//! directory per run, and `mesh-llm stop` and `uninstall` printed one line for
//! each of them. A slot holds that run's native logs, which are what a person
//! reads after a crash, so a dead slot is kept for [`DEAD_SLOT_RETENTION`]
//! after its last write and removed after that.

use super::instance::validate::{self, PidIdentity};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

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
    let mut removed = 0;
    for entry in entries.flatten() {
        let dir = entry.path();
        let Some(dir_pid) = dir
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        if dir_pid == my_pid || !dir.is_dir() || owner_may_be_running(&dir, dir_pid) {
            continue;
        }
        let Some(last_write) = newest_write(&dir) else {
            continue;
        };
        let age = now.duration_since(last_write).unwrap_or_default();
        if age < retention {
            continue;
        }
        match fs::remove_dir_all(&dir) {
            Ok(()) => removed += 1,
            Err(error) => tracing::debug!(
                path = %dir.display(),
                %error,
                "could not remove a stopped instance's runtime directory"
            ),
        }
    }
    if removed > 0 {
        tracing::debug!(removed, root = %root.display(), "removed stopped instances' runtime directories");
    }
    removed
}

/// Whether the process that owns this slot may still be running. Only a
/// process that is gone, or an id now held by another process, answers no.
fn owner_may_be_running(dir: &Path, dir_pid: u32) -> bool {
    let owner = fs::read_to_string(dir.join("owner.json"))
        .ok()
        .and_then(|json| serde_json::from_str::<SlotOwner>(&json).ok());
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
    let modified = |path: &Path| fs::metadata(path).and_then(|meta| meta.modified()).ok();
    let mut newest = modified(dir);
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
    fn a_missing_root_removes_nothing() {
        let root = tempdir().unwrap();
        let missing = root.path().join("absent");
        assert_eq!(
            prune_dead_runtime_slots(&missing, 1, SystemTime::now(), DEAD_SLOT_RETENTION),
            0
        );
    }
}
