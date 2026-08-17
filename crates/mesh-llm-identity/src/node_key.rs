use std::io::Write;
use std::path::{Path, PathBuf};

use crate::CryptoError;

pub const NODE_KEY_BYTES: usize = 32;

pub fn default_node_key_path() -> Result<PathBuf, CryptoError> {
    let home = dirs::home_dir().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine home directory",
        ))
    })?;
    Ok(home.join(".mesh-llm").join("key"))
}

pub fn load_node_key_bytes_from_path(path: &Path) -> Result<[u8; NODE_KEY_BYTES], CryptoError> {
    ensure_private_node_key_file(path)?;

    let hex = std::fs::read_to_string(path)?;
    let bytes = hex::decode(hex.trim()).map_err(|err| CryptoError::InvalidKeyMaterial {
        reason: format!("bad node key hex in {}: {err}", path.display()),
    })?;
    bytes
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: format!(
                "node key in {} must be {NODE_KEY_BYTES} bytes",
                path.display()
            ),
        })
}

pub fn save_node_key_bytes_to_path(
    path: &Path,
    key_bytes: &[u8; NODE_KEY_BYTES],
) -> Result<(), CryptoError> {
    let parent = path.parent().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("node key path {} has no parent directory", path.display()),
        ))
    })?;
    ensure_private_node_key_dir(parent)?;
    if path.exists() {
        ensure_private_node_key_file(path)?;
    }
    write_bytes_atomically(path, hex::encode(key_bytes).as_bytes())?;
    ensure_private_node_key_file(path)?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_node_key_dir(dir: &Path) -> Result<(), CryptoError> {
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
fn ensure_private_node_key_dir(dir: &Path) -> Result<(), CryptoError> {
    std::fs::create_dir_all(dir)?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_node_key_file(path: &Path) -> Result<(), CryptoError> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(CryptoError::InvalidKeyMaterial {
            reason: format!("node key path {} is not a regular file", path.display()),
        });
    }
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_node_key_file(path: &Path) -> Result<(), CryptoError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(CryptoError::InvalidKeyMaterial {
            reason: format!("node key path {} is not a regular file", path.display()),
        });
    }
    Ok(())
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), CryptoError> {
    let parent = path.parent().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("node key path {} has no parent directory", path.display()),
        ))
    })?;
    let file_name = path.file_name().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("node key path {} has no file name", path.display()),
        ))
    })?;
    let tmp_path = parent.join(format!(
        ".{}.tmp-{}-{}",
        file_name.to_string_lossy(),
        std::process::id(),
        rand::random::<u64>()
    ));

    let write_result = (|| -> Result<(), CryptoError> {
        let mut options = std::fs::OpenOptions::new();
        options.create_new(true).write(true);

        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;

            options.mode(0o600);
        }

        let mut file = options.open(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);

        #[cfg(windows)]
        if path.exists() {
            std::fs::remove_file(path)?;
        }

        std::fs::rename(&tmp_path, path)?;

        #[cfg(unix)]
        {
            let dir = std::fs::File::open(parent)?;
            dir.sync_all()?;
        }

        Ok(())
    })();

    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }

    write_result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_node_key_path() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mesh-node-key-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("key")
    }

    #[test]
    fn node_key_bytes_round_trip() {
        let path = temp_node_key_path();
        let key = [7u8; NODE_KEY_BYTES];

        save_node_key_bytes_to_path(&path, &key).unwrap();

        assert_eq!(load_node_key_bytes_from_path(&path).unwrap(), key);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn rejects_wrong_length_node_key() {
        let path = temp_node_key_path();
        std::fs::write(&path, "abcd").unwrap();

        let error = load_node_key_bytes_from_path(&path).unwrap_err();

        assert!(matches!(error, CryptoError::InvalidKeyMaterial { .. }));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
