//! Local-only launch configuration for one model serving independent meshes.
//!
//! This is deliberately separate from `MeshConfig`: mesh config synchronization
//! must never change host capacity, membership credentials or identity paths.

use std::collections::BTreeSet;
use std::io::Read;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

const MAX_CONFIG_BYTES: u64 = 65_536;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MembershipsConfig {
    pub version: u32,
    pub model_id: String,
    pub state_dir: PathBuf,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
    #[serde(default = "default_membership_queue_capacity")]
    pub membership_queue_capacity: usize,
    #[serde(default = "default_context_size")]
    pub context_size: u32,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,
    #[serde(default)]
    pub management_port: u16,
    #[serde(default)]
    pub lan_only: bool,
    pub bind_ip: Option<IpAddr>,
    pub memberships: Vec<MembershipConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MembershipConfig {
    pub name: String,
    pub join_token_file: PathBuf,
    pub trust_store: Option<PathBuf>,
    pub owner_key: Option<PathBuf>,
}

impl MembershipsConfig {
    /// Read a bounded file and resolve paths relative to it, without creating
    /// state, reading credentials, opening listeners or loading a model.
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).context("open memberships config")?;
        let mut raw = String::new();
        file.take(MAX_CONFIG_BYTES + 1)
            .read_to_string(&mut raw)
            .context("read memberships config as UTF-8")?;
        ensure!(
            raw.len() as u64 <= MAX_CONFIG_BYTES,
            "memberships config exceeds 64 KiB"
        );
        // TOML errors can include source snippets and unknown values. Never
        // echo them: a mistaken inline token must not become a log credential.
        let mut config: Self = toml::from_str(&raw).map_err(|_| {
            anyhow::anyhow!(
                "invalid memberships TOML: check required fields, types and unknown keys"
            )
        })?;
        config.validate()?;
        let absolute = std::path::absolute(path).context("resolve memberships config path")?;
        let directory = absolute
            .parent()
            .context("memberships config needs a parent directory")?;
        config.resolve_paths(directory);
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported memberships config version (expected 1)"
        );
        ensure!(
            !self.model_id.trim().is_empty()
                && self.model_id.len() <= 256
                && !self.model_id.chars().any(char::is_control),
            "invalid shared model ID"
        );
        ensure!(
            !self.state_dir.as_os_str().is_empty(),
            "state_dir must be explicit"
        );
        ensure!(
            (1..=64).contains(&self.concurrency),
            "concurrency must be between 1 and 64"
        );
        ensure!(
            self.queue_capacity <= 4096 && self.membership_queue_capacity <= 4096,
            "queue capacities must not exceed 4096"
        );
        ensure!(
            self.context_size >= 512,
            "context_size must be at least 512"
        );
        ensure!(
            self.gpu_layers >= -1,
            "gpu_layers must be -1 or nonnegative"
        );
        ensure!(
            (1..=16).contains(&self.memberships.len()),
            "configure between one and sixteen memberships"
        );
        let mut names = BTreeSet::new();
        for member in &self.memberships {
            ensure!(
                valid_membership_directory_name(&member.name),
                "invalid membership directory name: use 1–64 lowercase ASCII letters, digits, '.', '_' or '-', without leading/trailing '.' or reserved state/device names"
            );
            ensure!(
                names.insert(&member.name),
                "membership names must be unique"
            );
            ensure!(
                !member.join_token_file.as_os_str().is_empty(),
                "join_token_file must not be empty"
            );
            for path in [member.trust_store.as_ref(), member.owner_key.as_ref()]
                .into_iter()
                .flatten()
            {
                ensure!(
                    !path.as_os_str().is_empty(),
                    "membership credential paths must not be empty"
                );
            }
        }
        Ok(())
    }

    fn resolve_paths(&mut self, directory: &Path) {
        self.state_dir = directory.join(&self.state_dir);
        for member in &mut self.memberships {
            member.join_token_file = directory.join(&member.join_token_file);
            member.trust_store = member.trust_store.as_ref().map(|path| directory.join(path));
            member.owner_key = member.owner_key.as_ref().map(|path| directory.join(path));
        }
    }
}

/// Common label syntax for configured and dynamically added memberships.
/// Lowercase names remain distinct on case-insensitive filesystems.
pub fn valid_membership_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && !name.starts_with('.')
        && name.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// Configured memberships use their names directly as state directories.
/// Reject host state filenames and Windows-normalized/device names on every
/// OS so a portable manifest cannot alias another identity or host control file.
pub fn valid_membership_directory_name(name: &str) -> bool {
    if !valid_membership_name(name)
        || name.ends_with('.')
        || matches!(name, "host.lock" | "management.token")
    {
        return false;
    }
    let stem = name.split('.').next().unwrap_or_default();
    if matches!(stem, "con" | "prn" | "aux" | "nul") {
        return false;
    }
    !(stem.len() == 4
        && (stem.starts_with("com") || stem.starts_with("lpt"))
        && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}

const fn default_concurrency() -> usize {
    1
}
const fn default_queue_capacity() -> usize {
    32
}
const fn default_membership_queue_capacity() -> usize {
    16
}
const fn default_context_size() -> u32 {
    4096
}
const fn default_gpu_layers() -> i32 {
    -1
}

#[cfg(test)]
mod tests;
