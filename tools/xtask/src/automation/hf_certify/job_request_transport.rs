//! Small locator to a bounded pinned regular request; provider mount attribution stays declared.
use super::admission;
use crate::command::DynResult;
use serde::Deserialize;
use std::path::Path;
pub(super) const MAX_REQUEST_BYTES: usize = 8 * 1048576;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Locator {
    schema_version: u32,
    path: String,
    sha256: String,
    byte_size: u64,
    repo: String,
    revision: String,
}
fn read(locator: &Locator) -> DynResult<Vec<u8>> {
    let path = Path::new(&locator.path);
    let parts = locator.repo.split('/').collect::<Vec<_>>();
    if locator.schema_version != 1
        || !path.is_absolute()
        || locator.path.len() > 4096
        || locator.path.chars().any(char::is_control)
        || path.components().any(|p| {
            matches!(
                p,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
        || !(1..=MAX_REQUEST_BYTES as u64).contains(&locator.byte_size)
        || !super::bootstrap::contract::hex(&locator.sha256, 64)
        || !super::bootstrap::contract::hex(&locator.revision, 40)
        || parts.len() != 2
        || !parts.iter().all(|p| {
            !p.is_empty()
                && *p != "."
                && *p != ".."
                && p.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
        })
    {
        return Err("mounted Jobs request locator refused".into());
    }
    // The existing owner checks regular/no-follow/nonblocking opened FD and bounded length.
    let bytes = admission::read(path, MAX_REQUEST_BYTES as u64)?;
    if bytes.len() as u64 != locator.byte_size || admission::digest(&bytes) != locator.sha256 {
        return Err("mounted Jobs request bytes changed".into());
    }
    Ok(bytes)
}
pub(super) fn environment(name: &str) -> DynResult<Vec<u8>> {
    if name != "MESH_HF_JOB_REQUEST" {
        return Err("mounted Jobs request environment refused".into());
    }
    let value = std::env::var_os(name).ok_or("mounted Jobs request locator absent")?;
    let value = value
        .to_str()
        .ok_or("mounted Jobs request locator Unicode")?;
    if value.is_empty() || value.len() > 8192 {
        return Err("mounted Jobs request locator byte bound".into());
    }
    read(&serde_json::from_str(value).map_err(|_| "mounted Jobs request typed locator refused")?)
}
#[cfg(test)]
#[path = "job_request_transport/tests.rs"]
mod tests;
