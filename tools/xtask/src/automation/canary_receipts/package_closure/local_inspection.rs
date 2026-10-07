//! Inspection authority for an uncommitted normal local repair, never workflow artifacts.
use super::{manifest_policy, parity_inventory, process, source, split_roster};
use crate::{automation::canary_receipts::Digest, command::DynResult};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Controller {
    root: PathBuf,
    revision: String,
    executable_sha256: Digest,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalRepairSource {
    controller: Controller,
    root: PathBuf,
    base: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inspection {
    authority: LocalRepairSource,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Roster {
    authority: LocalRepairSource,
    check: bool,
}
impl LocalRepairSource {
    fn validate(&self) -> DynResult<PathBuf> {
        source::revision(&self.controller.revision)?;
        source::revision(&self.base)?;
        if !self.controller.root.is_absolute() || !self.root.is_absolute() {
            return Err("local repair requires absolute source/controller roots".into());
        }
        let root = self.root.canonicalize()?;
        if root != self.controller.root.canonicalize()? || self.base != self.controller.revision {
            return Err("local repair cannot substitute selected/historical source".into());
        }
        if process::text(&root, &["rev-parse", "HEAD"])? != self.base {
            return Err("local repair checkout changed from frozen base".into());
        }
        if executable_digest(&std::env::current_exe()?)? != self.controller.executable_sha256 {
            return Err("local repair running controller differs from frozen executable".into());
        }
        process::check()?;
        Ok(root)
    }
}
fn executable_digest(path: &Path) -> DynResult<Digest> {
    let before = fs::symlink_metadata(path)?;
    if !before.is_file() || before.len() > 512 * 1024 * 1024 {
        return Err("local repair controller must be a bounded regular executable".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != before.len() {
        return Err("local repair controller changed during admission".into());
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    let mut read = 0u64;
    loop {
        process::check()?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        read += count as u64;
        if read > before.len() {
            return Err("local repair controller grew during admission".into());
        }
        digest.update(&buffer[..count]);
    }
    if read != before.len() || file.metadata()?.modified()? != metadata.modified()? {
        return Err("local repair controller changed during hashing".into());
    }
    Ok(Digest::try_from(hex::encode(digest.finalize()))?)
}
pub(super) fn execute(bytes: &[u8], verb: &str) -> DynResult<Value> {
    let (authority, check) = if verb == "local-split-roster" {
        let input: Roster = serde_json::from_slice(bytes)?;
        (input.authority, Some(input.check))
    } else {
        let input: Inspection = serde_json::from_slice(bytes)?;
        (input.authority, None)
    };
    let root = authority.validate()?;
    let output = match verb {
        "local-manifest-policy" => manifest_policy::admit(&root, &authority.base)?,
        "local-parity-inventory" => parity_inventory::admit(&root, &authority.base)?,
        "local-split-roster" => {
            split_roster::admit(&root, check.ok_or("local roster requires check mode")?)?
        }
        _ => return Err("unknown local repair inspection".into()),
    };
    authority.validate()?;
    process::check()?;
    Ok(output)
}
#[cfg(test)]
#[path = "local_inspection/tests.rs"]
mod tests;
// Append to the existing local_inspection owner, retaining its actual authority.
pub(super) fn with_parity<T>(
    bytes: &[u8],
    body: impl FnOnce(&Path, &Value, &Value, &Value, &crate::process::Cancellation) -> DynResult<T>,
) -> DynResult<T> {
    process::operation(|| {
        let input: Inspection = serde_json::from_slice(bytes)?;
        let root = input.authority.validate()?;
        let prepared = source::prepared(&root)?;
        let admitted = parity_inventory::admit(&root, &input.authority.base)?;
        let manifest = super::policy_document::json(&root, source::parity_manifest_path(&root)?)?;
        let registry = if admitted["classifications"].as_array().is_some_and(|rows| {
            rows.iter()
                .any(|row| row["artifact_id"].is_string() && !row["model_pin"].is_object())
        }) {
            super::policy_document::json(&root, "ci/model-artifacts/manifests/skippy-parity.json")?
        } else {
            Value::Null
        };
        let output = body(
            &root,
            &admitted,
            &manifest,
            &registry,
            &process::cancellation(),
        );
        if super::policy_document::json(&root, source::parity_manifest_path(&root)?)? != manifest {
            return Err("parity manifest changed during manual operation".into());
        }
        if !registry.is_null()
            && super::policy_document::json(
                &root,
                "ci/model-artifacts/manifests/skippy-parity.json",
            )? != registry
        {
            return Err("manual registry changed during operation".into());
        }
        let after = source::prepared(&root)?;
        if prepared.head != after.head || prepared.markers != after.markers {
            return Err("prepared native identity changed during manual parity operation".into());
        }
        input.authority.validate()?;
        process::check()?;
        output
    })
}
