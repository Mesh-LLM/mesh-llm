//! Where the client keeps its secret head key for a claimed model, and its
//! first build. The key lives only on this machine, readable by its owner
//! alone: a node that learned it could change logits the check would miss.
//!
//! One key checks every node that serves the model, and whether a check
//! with it passes is a bit a node learns about it. So a key that refused a
//! node's logits is dropped, and the next audit builds a fresh one.

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
/// with the seconds the build took. A file that is no key, or a key that
/// does not fit the model's output matrix or cannot check with it, is
/// built again: kept, it would leave the head unchecked for good.
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

/// Drops the key at `path`, after a check with it refused a node's logits.
/// A file that is already gone is fine.
pub fn discard(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove {}", path.display())),
    }
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

#[cfg(test)]
mod tests {
    use skippy_receipts::toy_turn::weights;

    use super::*;

    #[test]
    fn a_key_is_built_once_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys").join("toy.json");
        let weights = weights();
        let (built, seconds) = load_or_build(&path, &weights).unwrap();
        assert!(seconds.is_some());
        let (read, seconds) = load_or_build(&path, &weights).unwrap();
        assert!(seconds.is_none());
        assert_eq!(read, built);
    }

    #[test]
    fn a_file_that_cannot_check_is_built_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("toy.json");
        let weights = weights();
        let (key, _) = load_or_build(&path, &weights).unwrap();
        let mut json = serde_json::to_value(&key).unwrap();
        // a key that still parses and names the right matrix, with a
        // vector missing: every check with it would stop with an error
        json["vectors"].as_array_mut().unwrap().pop();
        for broken in [json.to_string(), "not a key".to_owned(), String::new()] {
            std::fs::write(&path, broken).unwrap();
            let (rebuilt, seconds) = load_or_build(&path, &weights).unwrap();
            assert!(seconds.is_some(), "the broken key was kept");
            assert!(rebuilt.fits(&weights));
            assert_ne!(rebuilt, key);
            // and the file now holds the key that was built
            assert_eq!(load_or_build(&path, &weights).unwrap().0, rebuilt);
        }
    }

    #[test]
    fn a_discarded_key_is_replaced_by_a_fresh_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("toy.json");
        let weights = weights();
        let (first, _) = load_or_build(&path, &weights).unwrap();
        discard(&path).unwrap();
        assert!(!path.exists());
        // dropping it twice is fine
        discard(&path).unwrap();
        let (second, seconds) = load_or_build(&path, &weights).unwrap();
        assert!(seconds.is_some());
        assert_ne!(second, first);
    }
}
