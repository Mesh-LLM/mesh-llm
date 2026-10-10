//! Stage both history outputs before replacement; rollback a partial publication.
use super::input::Input;
use crate::command::DynResult;
use std::{
    io::Write,
    path::{Path, PathBuf},
};
struct Stage {
    destination: PathBuf,
    prepared: Option<tempfile::NamedTempFile>,
    restore: Option<tempfile::NamedTempFile>,
    original: Option<Vec<u8>>,
    committed: bool,
}
pub(super) fn publish(input: &mut Input<'_>, outputs: [(&Path, &[u8]); 2]) -> DynResult<()> {
    let mut stages = Vec::new();
    for (path, bytes) in outputs {
        stages.push(stage(input, path, bytes)?);
    }
    if stages[0].destination == stages[1].destination {
        return Err("history output and report must be distinct".into());
    }
    for item in &stages {
        input.check()?;
        let current = existing(input, &item.destination)?;
        if current != item.original {
            return Err("history output changed before publication".into());
        }
    }
    let result = (|| {
        for item in &mut stages {
            input.check()?;
            let prepared = item
                .prepared
                .take()
                .ok_or("missing staged history output")?;
            prepared
                .persist(&item.destination)
                .map_err(|error| error.error)?;
            item.committed = true;
        }
        input.check()
    })();
    if let Err(error) = result {
        for item in stages.iter_mut().rev().filter(|item| item.committed) {
            if let Some(previous) = item.restore.take() {
                previous
                    .persist(&item.destination)
                    .map_err(|error| error.error)?;
            } else {
                std::fs::remove_file(&item.destination)?;
            }
        }
        return Err(error);
    }
    Ok(())
}
fn stage(input: &mut Input<'_>, path: &Path, bytes: &[u8]) -> DynResult<Stage> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let name = path.file_name().ok_or("invalid history output name")?;
    let destination = parent.canonicalize()?.join(name);
    if input.sources.contains(&destination) {
        return Err("history output aliases consumed evidence".into());
    }
    let original = existing(input, &destination)?;
    let permissions = std::fs::symlink_metadata(&destination)
        .ok()
        .map(|meta| meta.permissions());
    let mut prepared = tempfile::NamedTempFile::new_in(parent)?;
    prepared.write_all(bytes)?;
    prepared.as_file().sync_all()?;
    if let Some(permissions) = &permissions {
        prepared.as_file().set_permissions(permissions.clone())?;
    }
    let restore = if let Some(old) = &original {
        let mut restore = tempfile::NamedTempFile::new_in(parent)?;
        restore.write_all(old)?;
        restore.as_file().sync_all()?;
        if let Some(permissions) = permissions {
            restore.as_file().set_permissions(permissions)?;
        }
        Some(restore)
    } else {
        None
    };
    Ok(Stage {
        destination,
        prepared: Some(prepared),
        restore,
        original,
        committed: false,
    })
}
fn existing(input: &mut Input<'_>, path: &Path) -> DynResult<Option<Vec<u8>>> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
        Ok(_) => Ok(Some(input.read(path)?)),
    }
}
