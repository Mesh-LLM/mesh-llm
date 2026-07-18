//! Nostr key persistence and rotation.

use anyhow::Result;
use nostr_sdk::prelude::*;

// ---------------------------------------------------------------------------
// Keys — stored in ~/.mesh-llm/nostr.nsec
// ---------------------------------------------------------------------------

fn nostr_key_path() -> Result<std::path::PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".mesh-llm").join("nostr.nsec"))
}

/// Load or generate a Nostr keypair for publishing.
pub fn load_or_create_keys() -> Result<Keys> {
    load_or_create_keys_at(&nostr_key_path()?)
}

fn load_or_create_keys_at(path: &std::path::Path) -> Result<Keys> {
    if let Some(parent) = path.parent() {
        ensure_private_nostr_dir(parent)?;
    }

    if path.exists() {
        ensure_private_nostr_key_file(path)?;
        let nsec = std::fs::read_to_string(path)?;
        let sk = SecretKey::from_bech32(nsec.trim())?;
        Ok(Keys::new(sk))
    } else {
        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32()?;
        crate::crypto::write_keystore_bytes_atomically(path, nsec.as_bytes())?;
        tracing::info!("Generated new Nostr key, saved to {}", path.display());
        Ok(keys)
    }
}

#[cfg(unix)]
fn ensure_private_nostr_dir(dir: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(dir)?;
    let metadata = std::fs::metadata(dir)?;
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o700);
        std::fs::set_permissions(dir, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_nostr_dir(dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_nostr_key_file(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        anyhow::bail!("Nostr key path is not a regular file");
    }
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_nostr_key_file(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

/// Delete the Nostr key and node identity key.  After rotation the
/// node gets a fresh identity on next start.
pub fn rotate_keys() -> Result<()> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    let mesh_dir = home.join(".mesh-llm");

    let nostr_path = nostr_key_path()?;
    if nostr_path.exists() {
        std::fs::remove_file(&nostr_path)?;
        eprintln!("🔑 Deleted {}", nostr_path.display());
    } else {
        eprintln!("No Nostr key to rotate (none exists yet).");
    }

    let node_key_path = mesh_dir.join("key");
    if node_key_path.exists() {
        std::fs::remove_file(&node_key_path)?;
        eprintln!("🔑 Deleted {}", node_key_path.display());
    } else {
        eprintln!("No node key to rotate (none exists yet).");
    }

    eprintln!();
    eprintln!("✅ Keys rotated. New identities will be generated on next start.");
    Ok(())
}

#[cfg(test)]
mod rotate_key_tests {
    use super::*;
    use serial_test::serial;
    use std::fs;

    // rotate_keys uses hardcoded paths (~/.mesh-llm/), so we test the logic
    // by verifying files are created/deleted in the real location.
    // This is safe because rotate_keys only deletes key and nostr.nsec.
    //
    // Both scenarios (keys present + keys missing) run in a single test to
    // avoid a race — Rust runs tests in parallel and both would touch the
    // same files.

    #[test]
    #[serial]
    fn rotate_deletes_both_keys_and_handles_missing() {
        let dir = dirs::home_dir().unwrap().join(".mesh-llm");
        fs::create_dir_all(&dir).ok();

        let key_path = dir.join("key");
        let nsec_path = dir.join("nostr.nsec");

        // Save originals so we can restore after the test.
        let orig_key = if key_path.exists() {
            Some(fs::read(&key_path).unwrap())
        } else {
            None
        };
        let orig_nsec = if nsec_path.exists() {
            Some(fs::read(&nsec_path).unwrap())
        } else {
            None
        };

        // --- Scenario 1: both keys exist → rotate deletes them ---
        fs::write(&key_path, b"test-node-key").unwrap();
        fs::write(&nsec_path, b"test-nostr-nsec").unwrap();

        let result = rotate_keys();
        assert!(result.is_ok(), "rotate should succeed when keys exist");
        assert!(!key_path.exists(), "node key should be deleted");
        assert!(!nsec_path.exists(), "nostr key should be deleted");

        // --- Scenario 2: no keys on disk → rotate still succeeds ---
        // (files were just deleted above, so the directory is clean)
        let result = rotate_keys();
        assert!(result.is_ok(), "rotate should succeed even with no keys");

        // Restore originals.
        if let Some(k) = orig_key {
            fs::write(&key_path, k).ok();
        }
        if let Some(n) = orig_nsec {
            fs::write(&nsec_path, n).ok();
        }
    }
}

#[cfg(test)]
mod key_file_tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_key_path(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("{prefix}-{unique}"))
            .join("nostr.nsec")
    }

    #[test]
    fn load_or_create_keys_at_round_trips() {
        let path = temp_key_path("mesh-llm-nostr-key");
        let first = load_or_create_keys_at(&path).unwrap();
        let second = load_or_create_keys_at(&path).unwrap();
        assert_eq!(
            first.secret_key().to_bech32().unwrap(),
            second.secret_key().to_bech32().unwrap()
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_keys_at_hardens_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_key_path("mesh-llm-nostr-key-perms");
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755)).unwrap();

        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32().unwrap();
        std::fs::write(&path, &nsec).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        let loaded = load_or_create_keys_at(&path).unwrap();
        assert_eq!(
            loaded.secret_key().to_bech32().unwrap(),
            keys.secret_key().to_bech32().unwrap()
        );
        assert_eq!(
            std::fs::metadata(dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_keys_at_rejects_symlink_key() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_key_path("mesh-llm-nostr-key-symlink");
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let real_file = dir.join("nostr.real");
        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32().unwrap();
        std::fs::write(&real_file, &nsec).unwrap();
        std::fs::set_permissions(&real_file, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&real_file, &path).unwrap();

        let result = load_or_create_keys_at(&path);
        assert!(result.is_err(), "expected error for symlinked nostr key");

        let _ = std::fs::remove_dir_all(dir);
    }
}
