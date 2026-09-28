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

use std::fs;
use std::path::Path;

/// File name inside the mesh-llm state directory.
pub const VERSION_FILE: &str = "analytics-version";

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

/// Compare `current` against the recorded version and record `current`.
///
/// Best effort by design: a state directory that cannot be written yields
/// [`VersionTransition::Fresh`] every run, which over-reports nothing. The
/// alternative — treating an unwritable directory as a version change —
/// would report an upgrade on every single run.
///
/// The version is stored verbatim rather than sanitized. It is compared for
/// equality here and passed through
/// [`Label::sanitize_or_redact`](crate::Label::sanitize_or_redact) before it
/// reaches the wire, so a build environment that stamped something strange
/// into the version cannot smuggle it into a property.
pub fn record(dir: &Path, current: &str) -> VersionTransition {
    let path = dir.join(VERSION_FILE);
    let previous = fs::read_to_string(&path)
        .ok()
        .map(|raw| raw.trim().to_owned())
        .filter(|recorded| !recorded.is_empty());

    // Written on every transition, including `Fresh`, so the *next* run has a
    // baseline. Without this a directory that never gets a version file would
    // never be able to observe an upgrade.
    if previous.as_deref() != Some(current) {
        let _ = fs::create_dir_all(dir);
        let _ = fs::write(&path, format!("{current}\n"));
    }

    match previous {
        None => VersionTransition::Fresh,
        Some(recorded) if recorded == current => VersionTransition::Unchanged,
        Some(recorded) => VersionTransition::Changed { from: recorded },
    }
}

#[cfg(test)]
#[path = "version_state/tests.rs"]
mod tests;
