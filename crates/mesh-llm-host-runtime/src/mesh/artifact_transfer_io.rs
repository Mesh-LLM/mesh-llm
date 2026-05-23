use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tokio::io::{AsyncRead, AsyncWriteExt};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PartialArtifactSelection {
    pub(super) path: PathBuf,
    pub(super) offset: u64,
}

pub(super) struct PartialArtifactGuard {
    path: PathBuf,
    cleanup_on_drop: bool,
}

impl PartialArtifactGuard {
    #[cfg(test)]
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            path,
            cleanup_on_drop: true,
        }
    }

    pub(super) fn preserve_on_error(path: PathBuf) -> Self {
        Self {
            path,
            cleanup_on_drop: false,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.cleanup_on_drop = false;
    }

    pub(super) fn remove_now(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        self.disarm();
    }
}

impl Drop for PartialArtifactGuard {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub(super) fn partial_artifact_path(destination: &Path) -> PathBuf {
    let file_name = artifact_file_name(destination);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    destination.with_file_name(format!(
        ".{file_name}.{}.{}.part",
        std::process::id(),
        unique
    ))
}

pub(super) fn select_partial_artifact(
    destination: &Path,
    max_resume_size: u64,
) -> Result<PartialArtifactSelection> {
    if let Some((path, offset)) = largest_resumable_partial(destination, max_resume_size)? {
        return Ok(PartialArtifactSelection { path, offset });
    }
    Ok(PartialArtifactSelection {
        path: partial_artifact_path(destination),
        offset: 0,
    })
}

pub(super) async fn read_artifact_transfer_chunk<R>(
    reader: &mut R,
    buffer: &mut [u8],
    idle_timeout: std::time::Duration,
) -> Result<usize>
where
    R: AsyncRead + Unpin,
{
    let read = tokio::time::timeout(idle_timeout, tokio::io::AsyncReadExt::read(reader, buffer))
        .await
        .map_err(|_| {
            anyhow::anyhow!("artifact transfer body read idle timeout after {idle_timeout:?}")
        })?
        .context("read artifact transfer bytes")?;
    anyhow::ensure!(
        read > 0,
        "artifact transfer ended before expected byte count"
    );
    Ok(read)
}

pub(super) async fn append_artifact_transfer_body<R>(
    reader: &mut R,
    partial_path: &Path,
    offset: u64,
    total_size: u64,
    buffer_bytes: usize,
    idle_timeout: std::time::Duration,
) -> Result<()>
where
    R: AsyncRead + Unpin,
{
    let actual_offset = match tokio::fs::metadata(partial_path).await {
        Ok(metadata) => {
            anyhow::ensure!(metadata.is_file(), "partial artifact is not a file");
            metadata.len()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error).context("stat partial artifact"),
    };
    anyhow::ensure!(
        actual_offset == offset,
        "partial artifact changed while opening transfer"
    );

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(partial_path)
        .await
        .context("open partial artifact")?;
    let mut remaining = total_size.saturating_sub(offset);
    let mut buffer = vec![0u8; buffer_bytes];
    while remaining > 0 {
        let limit = buffer.len().min(remaining as usize);
        let read = read_artifact_transfer_chunk(reader, &mut buffer[..limit], idle_timeout).await?;
        file.write_all(&buffer[..read])
            .await
            .context("write partial artifact")?;
        remaining -= read as u64;
    }
    file.flush().await.context("flush partial artifact")?;
    Ok(())
}

fn largest_resumable_partial(
    destination: &Path,
    max_resume_size: u64,
) -> Result<Option<(PathBuf, u64)>> {
    let Some(parent) = destination.parent() else {
        return Ok(None);
    };
    let entries = match std::fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("read package artifact directory"),
    };
    let prefix = partial_file_prefix(destination);
    let mut best: Option<(PathBuf, u64)> = None;
    for entry in entries {
        let entry = entry.context("read package artifact directory entry")?;
        let path = entry.path();
        if !is_partial_for_destination(&path, &prefix) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if !metadata.is_file() {
            continue;
        }
        let size = metadata.len();
        if size > max_resume_size {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        let replace = best.as_ref().is_none_or(|(_, best_size)| size > *best_size);
        if replace {
            best = Some((path, size));
        }
    }
    Ok(best)
}

fn is_partial_for_destination(path: &Path, prefix: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".part"))
}

fn partial_file_prefix(destination: &Path) -> String {
    format!(".{}.", artifact_file_name(destination))
}

fn artifact_file_name(destination: &Path) -> String {
    destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn append_artifact_transfer_body_resumes_existing_partial() {
        let temp = tempfile::tempdir().unwrap();
        let partial = temp.path().join(".artifact.gguf.123.part");
        std::fs::write(&partial, b"layer").unwrap();
        let (mut writer, mut reader) = tokio::io::duplex(3);
        tokio::spawn(async move {
            writer.write_all(b"000").await.unwrap();
        });

        append_artifact_transfer_body(
            &mut reader,
            &partial,
            5,
            8,
            2,
            std::time::Duration::from_secs(1),
        )
        .await
        .unwrap();

        assert_eq!(std::fs::read(partial).unwrap(), b"layer000");
    }

    #[test]
    fn partial_artifact_guard_removes_armed_partial_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(".artifact.part");
        std::fs::write(&path, b"partial").unwrap();

        {
            let _guard = PartialArtifactGuard::new(path.clone());
        }

        assert!(!path.exists());
    }

    #[test]
    fn partial_artifact_guard_preserves_disarmed_installed_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(".artifact.part");
        std::fs::write(&path, b"partial").unwrap();

        {
            let mut guard = PartialArtifactGuard::new(path.clone());
            guard.disarm();
        }

        assert!(path.exists());
    }

    #[test]
    fn partial_artifact_guard_can_preserve_partial_after_transfer_error() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(".artifact.part");
        std::fs::write(&path, b"partial").unwrap();

        {
            let _guard = PartialArtifactGuard::preserve_on_error(path.clone());
        }

        assert!(path.exists());
    }

    #[test]
    fn select_partial_artifact_reuses_largest_valid_partial() {
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("layer-000.gguf");
        let small = temp.path().join(".layer-000.gguf.small.part");
        let large = temp.path().join(".layer-000.gguf.large.part");
        let oversized = temp.path().join(".layer-000.gguf.oversized.part");
        let unrelated = temp.path().join(".layer-001.gguf.large.part");
        std::fs::write(&small, b"la").unwrap();
        std::fs::write(&large, b"layer").unwrap();
        std::fs::write(&oversized, b"layer0000").unwrap();
        std::fs::write(&unrelated, b"layer000").unwrap();

        let selected = select_partial_artifact(&destination, 8).unwrap();

        assert_eq!(
            selected,
            PartialArtifactSelection {
                path: large,
                offset: 5
            }
        );
        assert!(!oversized.exists());
        assert!(unrelated.exists());
    }
}
