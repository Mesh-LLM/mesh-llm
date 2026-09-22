//! Private backups and same-directory replacement; never follow a config symlink.
use super::*;
use std::fs::{self, File, OpenOptions};

/// What a successful [`save`] did to the config file.
#[derive(Debug)]
pub(super) enum SaveOutcome {
    /// The requested contents already matched the file; nothing was written.
    Unchanged,
    /// No previous file existed, so the config was created without a backup.
    Created,
    /// The previous config was replaced after its exact bytes were backed up.
    Replaced,
}

/// Read the config, refusing anything that is not a regular file.
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

/// Create `path` with mode 0600 on Unix, failing if it already exists.
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

/// Exclusive advisory lock that serializes this tool's own config writers.
///
/// The lock file lives beside the config so the final stale check and the
/// replacement cannot interleave between two `mesh-llm` invocations. Editors
/// that do not take this lock are outside the guarantee, and the kernel
/// releases the lock when the process exits, so a crash cannot strand it.
struct WriteLock {
    file: File,
}

impl WriteLock {
    fn acquire(path: &Path) -> Result<Self> {
        let name = path
            .file_name()
            .context("Config path has no filename")?
            .to_string_lossy();
        let path = path.with_file_name(format!(".{name}.mesh-write.lock"));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("Cannot open config lock {}", path.display()))?;
        fs2::FileExt::try_lock_exclusive(&file).map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                anyhow::anyhow!(
                    "Another mesh-llm config write holds {}; retry",
                    path.display()
                )
            } else {
                anyhow::Error::new(error).context(format!("Cannot lock {}", path.display()))
            }
        })?;
        Ok(Self { file })
    }
}

impl Drop for WriteLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

/// Replace the config with `updated`, refusing if the file no longer matches
/// `original`, and report whether a backup was made.
pub(super) fn save(path: &Path, original: Option<&[u8]>, updated: &[u8]) -> Result<SaveOutcome> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let _lock = WriteLock::acquire(path)?;
    if read(path)?.as_deref() != original {
        bail!("Config changed during setup; retry");
    }
    if original == Some(updated) {
        return Ok(SaveOutcome::Unchanged);
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
    let outcome = if let Some(bytes) = original {
        create_private(&parent.join(format!("{name}.{suffix}.bak")), bytes)?;
        SaveOutcome::Replaced
    } else {
        SaveOutcome::Created
    };
    let staged = parent.join(format!(".{name}.{suffix}.tmp"));
    let result = (|| {
        create_private(&staged, updated)?;
        if read(path)?.as_deref() != original {
            bail!("Config changed during setup; refusing replacement");
        }
        fs::rename(&staged, path)?;
        // A rename updates the file's directory entry, so sync the directory
        // too or a crash can leave the replacement non-durable.
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if staged.exists() {
        fs::remove_file(&staged)?;
    }
    result?;
    Ok(outcome)
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
        let outcome = save(&path, Some(&old), b"model: new\n").unwrap();
        assert!(matches!(outcome, SaveOutcome::Replaced));
        let backup = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "bak"))
            .unwrap();
        assert_eq!(fs::read(backup).unwrap(), old);
        assert!(save(&path, Some(&old), b"stale").is_err());
        assert_eq!(fs::read(path).unwrap(), b"model: new\n");
    }

    #[test]
    fn creation_and_unchanged_writes_report_no_backup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("openclaw.json");
        let outcome = save(&path, None, b"{}\n").unwrap();
        assert!(matches!(outcome, SaveOutcome::Created));
        let current = read(&path).unwrap().unwrap();
        let outcome = save(&path, Some(&current), b"{}\n").unwrap();
        assert!(matches!(outcome, SaveOutcome::Unchanged));
        assert!(
            !fs::read_dir(dir.path()).unwrap().any(|entry| entry
                .unwrap()
                .path()
                .extension()
                .is_some_and(|extension| extension == "bak")),
            "neither write should have created a backup"
        );
    }

    #[test]
    fn a_held_write_lock_refuses_a_second_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        let _held = WriteLock::acquire(&path).unwrap();
        let error = save(&path, None, b"model: new\n").unwrap_err();
        assert!(
            error.to_string().contains("Another mesh-llm config write"),
            "{error}"
        );
        assert!(
            !path.exists(),
            "the refused writer must not create a config"
        );
    }
}
