//! The last build version this install reported under.
//!
//! An upgrade is invisible to every other signal here. `install_first_run`
//! fires once and never again, and `serve_started` carries the current
//! version but says nothing about what came before it — so an install that
//! upgraded and one that never moved look identical unless it happens to
//! serve. Recording the last-seen version turns that into an explicit
//! [`Event::InstallUpdated`](crate::Event::InstallUpdated).
//!
//! Comparing against stored state rather than hooking the updater is what
//! makes this cover every route a new binary can arrive by: `--auto-update`,
//! `mesh-llm update`, a re-run of `install.sh`, a package manager, or a
//! hand-swapped binary. The updater does not have to cooperate, and a process
//! that `exec`s itself away mid-update does not have to deliver anything: the
//! next run observes the change and reports it with a normal lifetime and a
//! normal flush.
//!
//! # Reporting exactly once
//!
//! The transition is a read-compare-write across processes, and an upgrade
//! must produce one `install_updated`, not none and not several. Three things
//! hold that together:
//!
//! - **A lock.** Several mesh-llm processes can start at once after an
//!   upgrade. Without a lock each would read the old version and each would
//!   report the same transition. The read and the write happen under one
//!   exclusive lock on the state directory, and a process that cannot take it
//!   reports nothing rather than racing.
//! - **An atomic replace.** The record is written to a temporary file and
//!   renamed over the old one, so a crash mid-write leaves the previous
//!   version intact instead of a truncated file that reads as no record.
//! - **Commit before report.** [`PendingVersion::commit`] returns whether the
//!   record now actually reflects this build, and the caller only reports the
//!   upgrade when it does. A write that failed leaves the old version in
//!   place, so the next run sees the same transition again and can retry —
//!   the opposite of the old behaviour, which reported a change it had failed
//!   to persist and then reported it again on every subsequent run.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// File name inside the mesh-llm state directory.
pub const VERSION_FILE: &str = "analytics-version";

/// Lock held across the read-compare-write.
const VERSION_LOCK_FILE: &str = "analytics-version.lock";

/// Temporary file the new record is staged in before being renamed into place.
const VERSION_TEMP_FILE: &str = "analytics-version.tmp";

/// What changed about the build version since the last run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VersionTransition {
    /// No version was recorded before, so there is nothing to compare to.
    /// A genuine first run reports `install_first_run` instead.
    Fresh,
    /// The recorded version matches this build.
    Unchanged,
    /// This build differs from the recorded one.
    Changed {
        /// The version recorded by the previous run.
        from: String,
    },
}

/// A read transition that has not been written back yet.
///
/// Holds the state-directory lock until it is committed or dropped, so no
/// other process can observe the same transition in the meantime.
#[must_use = "the transition is not persisted until it is committed"]
pub struct PendingVersion {
    transition: VersionTransition,
    /// `None` when there is nothing to persist, or when the lock could not be
    /// taken and this process must stay out of the way.
    target: Option<Target>,
    /// Dropping the handle releases the lock. Also released by the OS if the
    /// process dies, which is why a crash cannot wedge later runs.
    _lock: Option<File>,
}

struct Target {
    dir: PathBuf,
    current: String,
}

impl PendingVersion {
    /// What the previous run recorded, relative to this build.
    #[must_use]
    pub fn transition(&self) -> &VersionTransition {
        &self.transition
    }

    /// Persist this build as the recorded version.
    ///
    /// Returns whether the record now reflects this build. `false` means the
    /// write failed or was skipped, the previous record still stands, and the
    /// caller must not report the transition — the next run will see it again.
    pub fn commit(self) -> bool {
        let Some(target) = self.target.as_ref() else {
            // Nothing to write (already current), or the lock was not ours.
            // Either way this process must not report a transition.
            return false;
        };
        write_atomically(&target.dir, &target.current).is_ok()
    }

    /// A transition this process must not act on.
    fn inert() -> Self {
        Self {
            transition: VersionTransition::Fresh,
            target: None,
            _lock: None,
        }
    }
}

/// Read the recorded version, holding the state lock until the caller commits.
///
/// Best effort by design: a state directory that cannot be locked or read
/// yields an inert [`PendingVersion`], which reports nothing. The alternative
/// — treating an unreadable directory as a version change — would report an
/// upgrade on every single run.
///
/// The version is stored verbatim rather than sanitized. It is compared for
/// equality here and passed through
/// [`Label::sanitize_or_redact`](crate::Label::sanitize_or_redact) before it
/// reaches the wire, so a build environment that stamped something strange
/// into the version cannot smuggle it into a property.
pub fn begin(dir: &Path, current: &str) -> PendingVersion {
    if fs::create_dir_all(dir).is_err() {
        return PendingVersion::inert();
    }

    // An exclusive lock, not a shared one: this is a read-compare-write, and
    // two readers that both see the old version both report the upgrade.
    let Ok(lock) = File::create(dir.join(VERSION_LOCK_FILE)) else {
        return PendingVersion::inert();
    };
    if lock.try_lock().is_err() {
        // Another mesh-llm process holds it and is resolving the same
        // transition. Exactly one of them should report it, and it is not us.
        return PendingVersion::inert();
    }

    let previous = fs::read_to_string(dir.join(VERSION_FILE))
        .ok()
        .map(|raw| raw.trim().to_owned())
        .filter(|recorded| !recorded.is_empty());

    let transition = match &previous {
        None => VersionTransition::Fresh,
        Some(recorded) if recorded == current => VersionTransition::Unchanged,
        Some(recorded) => VersionTransition::Changed {
            from: recorded.clone(),
        },
    };

    // Nothing to write when the record already names this build. Skipping it
    // keeps an ordinary run from touching the state directory at all.
    let target = (previous.as_deref() != Some(current)).then(|| Target {
        dir: dir.to_owned(),
        current: current.to_owned(),
    });

    PendingVersion {
        transition,
        target,
        _lock: Some(lock),
    }
}

/// Replace the record by renaming a fully written temporary file over it.
///
/// A direct write truncates first, so a crash between truncate and write
/// leaves an empty record that reads as "no version ever seen" — which would
/// silently swallow the next upgrade. `rename` within one directory is atomic,
/// so a reader sees either the old record or the new one.
///
/// The fixed temporary name is safe because the caller holds the exclusive
/// lock for the whole read-commit window.
fn write_atomically(dir: &Path, current: &str) -> std::io::Result<()> {
    let temp = dir.join(VERSION_TEMP_FILE);
    // Scoped so the handle is closed before the rename; Windows refuses to
    // rename a file that is still open.
    {
        let mut file = File::create(&temp)?;
        file.write_all(format!("{current}\n").as_bytes())?;
        // Without this the rename can be durable while the contents are not,
        // leaving an empty record after a crash.
        file.sync_all()?;
    }
    fs::rename(&temp, dir.join(VERSION_FILE)).inspect_err(|_| {
        let _ = fs::remove_file(&temp);
    })
}

#[cfg(test)]
#[path = "version_state/tests.rs"]
mod tests;
