//! The recorded opt-out, for the standalone CLI.
//!
//! Skippy has no global configuration file, so `skippy analytics disable`
//! and `enable` record the choice in the analytics state directory. It is
//! read on every start and survives upgrades.

use crate::consent::ConfigPreference;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// File name inside the analytics state directory.
pub const PREFERENCE_FILE: &str = "analytics-preference";

/// What `dir` records, failing closed on anything unreadable.
#[must_use]
pub fn read_preference(dir: &Path) -> ConfigPreference {
    match fs::read_to_string(dir.join(PREFERENCE_FILE)) {
        Ok(raw) => match raw.trim() {
            "enabled" => ConfigPreference::Stated(true),
            "disabled" => ConfigPreference::Stated(false),
            _ => ConfigPreference::Unreadable,
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ConfigPreference::Unstated,
        Err(_) => ConfigPreference::Unreadable,
    }
}

/// Record `enabled` in `dir`.
pub fn write_preference(dir: &Path, enabled: bool) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let value = if enabled { "enabled\n" } else { "disabled\n" };
    let path = dir.join(PREFERENCE_FILE);
    fs::write(&path, value).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn an_absent_preference_is_unstated() {
        let dir = TempDir::new().unwrap();
        assert_eq!(read_preference(dir.path()), ConfigPreference::Unstated);
    }

    #[test]
    fn a_recorded_choice_round_trips() {
        let dir = TempDir::new().unwrap();
        write_preference(dir.path(), false).unwrap();
        assert_eq!(read_preference(dir.path()), ConfigPreference::Stated(false));
        write_preference(dir.path(), true).unwrap();
        assert_eq!(read_preference(dir.path()), ConfigPreference::Stated(true));
    }

    #[test]
    fn a_garbled_preference_fails_closed() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(PREFERENCE_FILE), "maybe").unwrap();
        assert_eq!(read_preference(dir.path()), ConfigPreference::Unreadable);
    }
}
