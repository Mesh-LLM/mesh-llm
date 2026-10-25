//! Pinned request bytes on an existing readonly immutable model mount, never a large secret.
use super::{ModelMount, admission};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
pub const MAX_REQUEST_BYTES: usize = 8 * 1048576;
pub const LOCATOR_ENVIRONMENT: &str = "MESH_HF_JOB_REQUEST";
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MountedRequest {
    pub schema_version: u32,
    pub path: String,
    pub sha256: String,
    pub byte_size: u64,
    pub repo: String,
    pub revision: String,
}
impl MountedRequest {
    pub(super) fn admit(&self, bytes: &[u8], mounts: &[ModelMount]) -> Result<()> {
        admission::mounts_admitted(mounts)?;
        let path = Path::new(&self.path);
        if self.schema_version != 1
            || !admission::absolute(&self.path)
            || !(1..=MAX_REQUEST_BYTES as u64).contains(&self.byte_size)
            || self.byte_size != bytes.len() as u64
            || self.sha256 != admission::digest(bytes)
            || !mounts.iter().any(|m| {
                m.repo == self.repo
                    && m.revision == self.revision
                    && path
                        .strip_prefix(&m.mount_path)
                        .is_ok_and(|p| !p.as_os_str().is_empty())
            })
        {
            bail!("mounted Jobs request pin/immutable mount refused");
        }
        Ok(())
    }
}
#[cfg(all(test, unix))]
#[path = "request_transport/tests.rs"]
mod tests;
