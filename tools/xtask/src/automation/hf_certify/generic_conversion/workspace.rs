//! Admit existing resume paths before the native converter can mutate its workspace.
use super::contract::Input;
use crate::command::DynResult;
use std::path::Path;
fn existing(path: &Path, directory: bool) -> DynResult<()> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(m) if (directory && m.is_dir()) || (!directory && m.is_file()) => Ok(()),
        _ => Err("generic resume workspace refuses links/nonregular paths".into()),
    }
}
pub(super) fn admit(input: &Input) -> DynResult<()> {
    if !std::fs::symlink_metadata(&input.work_directory)?.is_dir() {
        return Err("generic work root must be regular directory".into());
    }
    for path in [
        input.work_directory.join("target"),
        input.artifact_directory(),
        input.work_directory.join("spool"),
        input.work_directory.join("records"),
    ] {
        existing(&path, true)?;
    }
    for path in [
        input.work_directory.join("convert-manifest.json"),
        input.work_directory.join("status.json"),
    ] {
        existing(&path, false)?;
    }
    Ok(())
}
