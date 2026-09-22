//! The anonymous install identifier.
//!
//! This is a random v4 UUID with no derivation from anything about the
//! machine: not the hostname, not a MAC address, not a disk serial. It is
//! deliberately **not** the mesh node identity from `mesh-llm-identity` —
//! that key is published to the public mesh, and reusing it would tie every
//! analytics event to a publicly visible node. Deleting the file resets the
//! identifier, and the user is told so in `mesh-llm analytics status`.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// File name inside the mesh-llm state directory.
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

/// The default state directory, `~/.mesh-llm`.
pub fn state_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    Ok(home.join(".mesh-llm"))
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
