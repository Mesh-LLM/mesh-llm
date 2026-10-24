//! Host-owned launch inputs; never synchronized by a member mesh's config API.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

#[derive(Clone, Debug)]
pub struct ServingMembershipConfig {
    pub name: String,
    pub token_file: PathBuf,
    pub trust_store: Option<PathBuf>,
    pub owner_key: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct SharedServingConfig {
    pub model_path: PathBuf,
    pub model_id: String,
    pub state_dir: PathBuf,
    pub context_size: u32,
    pub gpu_layers: i32,
    pub concurrency: usize,
    pub queue_capacity: usize,
    pub membership_queue_capacity: usize,
    pub lan_only: bool,
    pub bind_ip: Option<std::net::IpAddr>,
    pub memberships: Vec<ServingMembershipConfig>,
}

impl SharedServingConfig {
    /// Validate before allocating native memory, binding endpoints or creating identity files.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=16).contains(&self.memberships.len()),
            "configure between one and sixteen independent memberships"
        );
        ensure!(
            (1..=64).contains(&self.concurrency),
            "shared concurrency must be between 1 and 64"
        );
        ensure!(
            self.queue_capacity <= 4096 && self.membership_queue_capacity <= 4096,
            "shared queue capacities must not exceed 4096"
        );
        ensure!(
            self.context_size >= 512,
            "shared context size must be at least 512"
        );
        ensure!(
            self.gpu_layers >= -1,
            "GPU layers must be -1 or nonnegative"
        );
        ensure!(
            !self.model_id.trim().is_empty()
                && self.model_id.len() <= 256
                && !self.model_id.chars().any(char::is_control),
            "invalid shared model ID"
        );
        ensure!(
            self.model_path.is_file(),
            "shared model must be an existing local GGUF file"
        );
        ensure!(
            !self.state_dir.as_os_str().is_empty(),
            "shared state directory must be explicit"
        );
        let mut names = BTreeSet::new();
        for membership in &self.memberships {
            ensure!(
                valid_membership_name(&membership.name),
                "invalid membership name"
            );
            ensure!(
                names.insert(membership.name.to_ascii_lowercase()),
                "membership names must be unique ignoring ASCII case: {}",
                membership.name
            );
            read_join_token(&membership.token_file)?;
            if let Some(path) = &membership.trust_store {
                read_explicit_trust_store(path)?;
            }
            if let Some(path) = &membership.owner_key {
                ensure!(path.is_file(), "explicit owner keystore does not exist");
            }
        }
        Ok(())
    }
}

/// Unlike the ordinary optional trust store, an explicitly selected file must
/// exist on every reconnect. Read it directly so removal cannot fall back to
/// the default `off` policy between an existence check and the actual read.
pub(super) fn read_explicit_trust_store(path: &Path) -> Result<crate::crypto::TrustStore> {
    let bytes = std::fs::read(path).context("read explicit membership trust store")?;
    let store: crate::crypto::TrustStore =
        serde_json::from_slice(&bytes).context("parse explicit membership trust store")?;
    ensure!(
        store.version == mesh_llm_identity::ownership::TRUST_STORE_VERSION,
        "unsupported membership trust store version"
    );
    Ok(store)
}

fn valid_membership_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

/// Reread on every reconnect. Never fall back to an old token after a bad file.
pub(super) fn read_join_token(path: &Path) -> Result<String> {
    use std::io::Read;
    let file = std::fs::File::open(path).context("open membership token file")?;
    let mut token = String::new();
    file.take(65537)
        .read_to_string(&mut token)
        .context("read membership token file")?;
    ensure!(token.len() <= 65536, "membership token exceeds 64 KiB");
    let token = token.trim();
    ensure!(
        !token.is_empty() && !token.chars().any(char::is_whitespace),
        "membership token file must contain exactly one token"
    );
    crate::mesh::parse_invite_token(token)
        .map_err(|_| anyhow::anyhow!("invalid membership join token"))?;
    Ok(token.to_owned())
}

pub(super) fn private_directory(path: &Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)?;
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "shared state must be a real directory, not a symlink"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "shared state directory must be private (mode 0700)"
        );
    }
    Ok(())
}

pub(super) fn private_file(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    ensure!(
        !std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()),
        "shared state file must not be a symlink"
    );
    Ok(options.open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_trust_store_requires_valid_supported_contents() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trust.json");
        assert!(read_explicit_trust_store(&path).is_err());
        let mut trust = crate::crypto::TrustStore {
            policy: crate::crypto::TrustPolicy::Allowlist,
            ..Default::default()
        };
        crate::crypto::save_trust_store(&path, &trust).unwrap();
        assert!(matches!(
            read_explicit_trust_store(&path).unwrap().policy,
            crate::crypto::TrustPolicy::Allowlist
        ));
        trust.version += 1;
        crate::crypto::save_trust_store(&path, &trust).unwrap();
        assert!(read_explicit_trust_store(&path).is_err());
        std::fs::write(&path, "not JSON").unwrap();
        assert!(read_explicit_trust_store(&path).is_err());
    }

    #[test]
    fn labels_are_bounded_path_components() {
        for name in ["", ".", "..", "work/lab", "work\\lab", "é", "a\nb"] {
            assert!(!valid_membership_name(name), "accepted {name:?}");
        }
        assert!(valid_membership_name("work-2_lan"));
        assert!(!valid_membership_name(&"x".repeat(65)));
    }

    #[test]
    fn malformed_or_oversized_token_never_appears_in_diagnostics() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("token");
        for token in [
            "not a token".to_owned(),
            "secret-invalid-token".to_owned(),
            "x".repeat(65537),
        ] {
            std::fs::write(&path, &token).unwrap();
            let message = read_join_token(&path).unwrap_err().to_string();
            assert!(!message.contains(&token));
        }
    }

    #[cfg(unix)]
    #[test]
    fn state_paths_reject_symlinks_and_public_existing_directories() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let directory = tempfile::tempdir().unwrap();
        let state = directory.path().join("state");
        private_directory(&state).unwrap();
        assert_eq!(
            std::fs::metadata(&state).unwrap().permissions().mode() & 0o777,
            0o700
        );
        let linked = directory.path().join("linked");
        symlink(&state, &linked).unwrap();
        assert!(private_directory(&linked).is_err());
        let target = state.join("target");
        std::fs::write(&target, "unchanged").unwrap();
        let link = state.join("management.token");
        symlink(&target, &link).unwrap();
        assert!(private_file(&link).is_err());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "unchanged");
        std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(private_directory(&state).is_err());
        assert_eq!(
            std::fs::metadata(&state).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }
}
