//! The per-machine record of which plugin sets this machine has been offered.
//!
//! The recommended set installs once per machine: at setup, or at the first
//! update on a machine that was never offered it. This record is what makes
//! "once" true. After a set is recorded, update and setup never install or
//! prompt for it again, a member the operator removed is never reinstalled,
//! and members the catalog adds later surface as an attended notice rather
//! than an install. The record lives in the plugin store root as
//! `set-offers.json` (a map of set name to the member names already offered)
//! and the store listing ignores it because it is a file, not a plugin
//! directory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const RECORD_FILE: &str = "set-offers.json";

/// Set name to the member names this machine has already been offered.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SetOffers {
    sets: BTreeMap<String, Vec<String>>,
}

impl SetOffers {
    /// Read the record under `store_root`. A missing file is an empty record.
    /// An unreadable or malformed file is an error: callers treat it as "do
    /// not install", never as "never offered", so a damaged record cannot
    /// cause a reinstall.
    pub fn load(store_root: &Path) -> Result<Self> {
        let path = record_path(store_root);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => {
                return Err(error).with_context(|| format!("read {}", path.display()));
            }
        };
        let sets =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        Ok(Self { sets })
    }

    /// The members already offered for `set`, or `None` when the machine was
    /// never offered it.
    pub fn offered(&self, set: &str) -> Option<&[String]> {
        self.sets.get(set).map(Vec::as_slice)
    }

    /// Record `members` as offered for `set` and write the file. The write
    /// holds a cross-process lock, reloads what other commands recorded since
    /// this record was loaded, and merges it in, so an overlapping setup can
    /// never erase a recorded refusal.
    pub fn record(&mut self, store_root: &Path, set: &str, members: &[String]) -> Result<()> {
        std::fs::create_dir_all(store_root)
            .with_context(|| format!("create {}", store_root.display()))?;
        let _lock = RecordLock::acquire(store_root)?;
        // A damaged file under the lock is replaced: callers that must not
        // reinstall already declined to get here (see `load`).
        let latest = Self::load(store_root).unwrap_or_default();
        for (name, offered) in latest.sets {
            let entry = self.sets.entry(name).or_default();
            for member in offered {
                if !entry.contains(&member) {
                    entry.push(member);
                }
            }
        }
        let entry = self.sets.entry(set.to_string()).or_default();
        for member in members {
            if !entry.contains(member) {
                entry.push(member.clone());
            }
        }
        let path = record_path(store_root);
        let bytes = serde_json::to_vec_pretty(&self.sets).context("serialize set offers")?;
        // Write beside the record and rename over it, so an interrupted write
        // never leaves a torn file that would block later offers.
        let temp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&temp, bytes).with_context(|| format!("write {}", temp.display()))?;
        std::fs::rename(&temp, &path).with_context(|| format!("replace {}", path.display()))
    }

    /// Where the record lives under `store_root`.
    pub fn path(store_root: &Path) -> PathBuf {
        record_path(store_root)
    }
}

fn record_path(store_root: &Path) -> PathBuf {
    store_root.join(RECORD_FILE)
}

/// An exclusive advisory lock on `set-offers.lock`, released on drop. The
/// lock file is a file in the store root, so the store listing ignores it.
struct RecordLock {
    file: std::fs::File,
}

impl RecordLock {
    fn acquire(store_root: &Path) -> Result<Self> {
        let path = store_root.join("set-offers.lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        fs2::FileExt::lock_exclusive(&file).with_context(|| format!("lock {}", path.display()))?;
        Ok(Self { file })
    }
}

impl Drop for RecordLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

/// Members of `current` the record has not offered yet, in catalog order.
pub fn members_added_since(offered: &[String], current: &[String]) -> Vec<String> {
    current
        .iter()
        .filter(|member| !offered.contains(member))
        .cloned()
        .collect()
}

/// The attended notice for members added to an already-offered set. It names
/// the explicit command, which installs members not yet installed.
pub fn added_members_notice(set: &str, added: &[String]) -> Vec<String> {
    vec![
        format!(
            "ℹ️  The {set} plugin set has new members: {}.",
            added.join(", ")
        ),
        format!("ℹ️  Install them when ready: mesh-llm plugins sets install {set}"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn a_missing_record_means_never_offered() {
        let temp = tempfile::tempdir().unwrap();
        let offers = SetOffers::load(temp.path()).unwrap();
        assert!(offers.offered("recommended").is_none());
    }

    #[test]
    fn a_recorded_offer_survives_a_reload_and_keeps_earlier_members() {
        let temp = tempfile::tempdir().unwrap();
        let mut offers = SetOffers::load(temp.path()).unwrap();
        offers
            .record(temp.path(), "recommended", &names(&["capsules"]))
            .unwrap();
        let mut offers = SetOffers::load(temp.path()).unwrap();
        assert_eq!(
            offers.offered("recommended"),
            Some(&names(&["capsules"])[..])
        );
        offers
            .record(temp.path(), "recommended", &names(&["capsules", "notes"]))
            .unwrap();
        let offers = SetOffers::load(temp.path()).unwrap();
        assert_eq!(
            offers.offered("recommended"),
            Some(&names(&["capsules", "notes"])[..])
        );
    }

    #[test]
    fn a_stale_record_merges_instead_of_erasing_a_recorded_refusal() {
        let temp = tempfile::tempdir().unwrap();
        // Two commands load the record before either writes.
        let mut first = SetOffers::load(temp.path()).unwrap();
        let mut second = SetOffers::load(temp.path()).unwrap();
        second
            .record(temp.path(), "recommended", &names(&["capsules"]))
            .unwrap();
        first
            .record(temp.path(), "extra", &names(&["notes"]))
            .unwrap();
        let reloaded = SetOffers::load(temp.path()).unwrap();
        assert_eq!(
            reloaded.offered("recommended"),
            Some(&names(&["capsules"])[..]),
            "the older writer must not drop the later refusal"
        );
        assert_eq!(reloaded.offered("extra"), Some(&names(&["notes"])[..]));
    }

    #[test]
    fn a_damaged_record_is_an_error_not_an_empty_record() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join(RECORD_FILE), b"not json").unwrap();
        assert!(SetOffers::load(temp.path()).is_err());
    }

    #[test]
    fn only_members_added_since_the_offer_are_new() {
        assert_eq!(
            members_added_since(&names(&["capsules"]), &names(&["capsules", "notes"])),
            names(&["notes"])
        );
        assert!(members_added_since(&names(&["capsules"]), &names(&["capsules"])).is_empty());
    }

    #[test]
    fn the_notice_names_the_new_members_and_the_command() {
        let lines = added_members_notice("recommended", &names(&["notes"]));
        assert!(lines[0].contains("notes"), "{lines:?}");
        assert!(lines[1].ends_with("mesh-llm plugins sets install recommended"));
    }
}
