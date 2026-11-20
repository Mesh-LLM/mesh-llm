//! The anonymous install identifier.
//!
//! This is a random v4 UUID with no derivation from anything about the
//! machine: not the hostname, not a MAC address, not a disk serial, and not
//! any identity an embedding host publishes. Deleting the file resets the
//! identifier, and the user is told so in `skippy analytics status`.
//!
//! An embedding host that governs consent supplies its own identifier
//! instead, and none is created here.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// File name inside the Skippy analytics state directory.
pub const INSTALL_ID_FILE: &str = "analytics-id";

/// An install identifier, and whether this process created it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstallId {
    id: String,
    first_run: bool,
}

impl InstallId {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.id
    }

    /// True when this process generated the identifier, i.e. a first run.
    #[must_use]
    pub const fn is_first_run(&self) -> bool {
        self.first_run
    }
}

/// The analytics state directory.
///
/// `SKIPPY_DATA_DIR/analytics` when the operator set one, so a relocated
/// install keeps one identity; otherwise the platform's local data directory,
/// then `~/.skippy`.
pub fn state_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("SKIPPY_DATA_DIR").filter(|dir| !dir.is_empty()) {
        return Ok(PathBuf::from(dir).join("analytics"));
    }
    if let Some(dir) = dirs::data_local_dir() {
        return Ok(dir.join("skippy").join("analytics"));
    }
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".skippy").join("analytics"))
}

/// Read the install identifier without creating one.
///
/// `skippy analytics status` uses this: asking what is collected must not
/// itself write state, and creating the identifier here would consume the
/// first-run signal so the real first run never reports as one.
pub fn load(dir: &Path) -> Option<InstallId> {
    let raw = fs::read_to_string(dir.join(INSTALL_ID_FILE)).ok()?;
    let trimmed = raw.trim();
    Uuid::parse_str(trimmed).ok()?;
    Some(InstallId {
        id: trimmed.to_owned(),
        first_run: false,
    })
}

/// Load the install identifier from `dir`, generating it on first use.
///
/// A malformed or empty file is replaced rather than treated as an error: a
/// corrupted identifier should not be able to break command dispatch.
pub fn load_or_create(dir: &Path) -> Result<InstallId> {
    let path = dir.join(INSTALL_ID_FILE);

    if let Ok(existing) = fs::read_to_string(&path) {
        let trimmed = existing.trim();
        if Uuid::parse_str(trimmed).is_ok() {
            return Ok(InstallId {
                id: trimmed.to_owned(),
                first_run: false,
            });
        }
    }

    fs::create_dir_all(dir)
        .with_context(|| format!("failed to create state directory {}", dir.display()))?;
    let id = Uuid::new_v4().to_string();
    fs::write(&path, format!("{id}\n"))
        .with_context(|| format!("failed to write install identifier {}", path.display()))?;
    restrict_permissions(&path);

    Ok(InstallId {
        id,
        first_run: true,
    })
}

/// Best-effort owner-only permissions. A failure here is not worth failing on.
#[cfg(unix)]
fn restrict_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) {}

#[cfg(test)]
#[path = "install_id/tests.rs"]
mod tests;
