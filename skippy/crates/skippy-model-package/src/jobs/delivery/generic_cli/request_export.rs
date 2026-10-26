//! Canonical offline request bytes and an immutable mount locator, validated before output.
use super::{Cli, Input, hash, prepare, read, root, write};
use crate::jobs::delivery::request_transport::{MAX_REQUEST_BYTES, MountedRequest};
use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{io::Write as _, path::Path, time::Instant};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Destination {
    path: String,
    repo: String,
    revision: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    delivery: Input,
    request_destination: Destination,
}
pub(super) fn worker_file(root: &Path, worker: &Value) -> Result<(String, u64)> {
    let bytes = serde_json::to_vec(worker)?;
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        bail!("canonical worker request byte bound");
    }
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    file.write_all(&bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(root.join("worker-input.json"))?;
    Ok((super::super::admission::digest(&bytes), bytes.len() as u64))
}
fn admitted(request: &mut Request) -> Result<MountedRequest> {
    if request.schema_version != 1 || request.delivery.mounted_request.is_some() {
        bail!("offline request export requires a delivery without a prior locator");
    }
    let bytes = serde_json::to_vec(&request.delivery.worker_input)?;
    let locator = MountedRequest {
        schema_version: 1,
        path: request.request_destination.path.clone(),
        sha256: super::super::admission::digest(&bytes),
        byte_size: bytes.len() as u64,
        repo: request.request_destination.repo.clone(),
        revision: request.request_destination.revision.clone(),
    };
    locator.admit(&bytes, &request.delivery.mounts)?;
    request.delivery.mounted_request = Some(locator.clone());
    // Existing typed workflow/resource/source admission precedes every output write.
    prepare(&request.delivery)?;
    Ok(locator)
}
pub(super) fn run(cli: &Cli, until: Instant) -> Result<bool> {
    if cli.credential_file.is_some() || cli.submitted_file.is_some() || cli.confirm_submission {
        bail!("offline export cannot consume credentials or submission authority");
    }
    let input = cli
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("export input absent"))?;
    let bytes = read(input, 16 * 1048576)?;
    let mut request: Request = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("offline export typed request refused"))?;
    let locator = admitted(&mut request)?;
    let output = cli
        .output_directory
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("export output absent"))?;
    if Instant::now() >= until {
        bail!("offline export deadline expired before output");
    }
    let root = root(output)?;
    let (sha, size) = worker_file(&root, &request.delivery.worker_input)?;
    if sha != locator.sha256 || size != locator.byte_size {
        bail!("canonical worker export correlation refused");
    }
    write(&root, "mounted-request.json", &locator)?;
    write(&root, "delivery.json", &request.delivery)?;
    write(
        &root,
        "result.json",
        &json!({"schema_version":1,"status":"REQUEST_EXPORTED_OFFLINE","input_sha256":super::super::admission::digest(&bytes),"delivery_input_sha256":hash(&request.delivery)?,"worker_input_sha256":sha,"worker_input_byte_size":size,"submitted":false,"mount_bytes_observed":false}),
    )?;
    Ok(Instant::now() < until)
}
#[cfg(test)]
#[path = "request_export/tests.rs"]
mod tests;
