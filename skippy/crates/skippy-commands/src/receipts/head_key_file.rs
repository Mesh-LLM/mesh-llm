//! Where the client keeps its secret head key for a claimed model, and its
//! first build. The key lives only on this machine, readable by its owner
//! alone: a node that learned it could change logits the check would miss.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use skippy_receipts::head::key::HeadKey;
use skippy_receipts::weights::WeightFile;

/// The key file for the claimed model `identity` names.
pub fn path_for(identity: &str) -> PathBuf {
    let name = hex::encode(&Sha256::digest(identity.as_bytes())[..16]);
    skippy_model_hf::application_cache_dir()
        .join("receipts")
        .join("head-keys")
        .join(format!("{name}.json"))
}

/// The key at `path`, or a fresh one built from `weights` and kept there,
/// with the seconds the build took.
pub fn load_or_build(path: &Path, weights: &WeightFile) -> Result<(HeadKey, Option<f64>)> {
    if let Some(key) = std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<HeadKey>(&bytes).ok())
        .filter(|key| key.fits(weights))
    {
        return Ok((key, None));
    }
    let started = Instant::now();
    let key = HeadKey::build(weights, rand::random())?;
    let seconds = started.elapsed().as_secs_f64();
    save(path, &key)?;
    Ok((key, Some(seconds)))
}

fn save(path: &Path, key: &HeadKey) -> Result<()> {
    let parent = path.parent().context("a key path has a directory")?;
    std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let temp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temp)
        .with_context(|| format!("write {}", temp.display()))?;
    file.write_all(&serde_json::to_vec(key)?)?;
    file.sync_all()?;
    std::fs::rename(&temp, path).with_context(|| format!("write {}", path.display()))
}
