//! Private backups and same-directory replacement; never follow a config symlink.
use super::*;
use std::fs::{self, OpenOptions};

pub(super) fn read(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::symlink_metadata(path) {
        Ok(meta) if !meta.file_type().is_file() => {
            bail!("Config must be a regular file, not a symlink")
        }
        Ok(_) => Ok(Some(fs::read(path)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn create_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn save(path: &Path, original: Option<&[u8]>, updated: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if read(path)?.as_deref() != original {
        bail!("Config changed during setup; retry");
    }
    if original == Some(updated) {
        return Ok(());
    }
    let suffix = format!(
        "mesh-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    );
    let name = path
        .file_name()
        .context("Config path has no filename")?
        .to_string_lossy();
    if let Some(bytes) = original {
        create_private(&parent.join(format!("{name}.{suffix}.bak")), bytes)?;
    }
    let staged = parent.join(format!(".{name}.{suffix}.tmp"));
    let result = (|| {
        create_private(&staged, updated)?;
        if read(path)?.as_deref() != original {
            bail!("Config changed during setup; refusing replacement");
        }
        fs::rename(&staged, path)?;
        Ok(())
    })();
    if staged.exists() {
        fs::remove_file(&staged)?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backup_is_exact_and_stale_write_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        fs::write(&path, b"# comment\nmodel: old\n").unwrap();
        let old = read(&path).unwrap().unwrap();
        save(&path, Some(&old), b"model: new\n").unwrap();
        let backup = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "bak"))
            .unwrap();
        assert_eq!(fs::read(backup).unwrap(), old);
        assert!(save(&path, Some(&old), b"stale").is_err());
        assert_eq!(fs::read(path).unwrap(), b"model: new\n");
    }
}
