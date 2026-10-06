//! Preparation candidate publication retains actual signal ownership and revokes only its file.
use super::*;
use std::io::Write as _;
#[cfg(test)]
#[path = "preparation_publication_tests.rs"]
pub(super) mod tests;
fn owned(path: &Path, file: &std::fs::File) -> DynResult<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() {
        return Err("preparation publication ownership changed".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let original = file.metadata()?;
        if original.dev() != metadata.dev() || original.ino() != metadata.ino() {
            return Err("preparation publication ownership changed".into());
        }
    }
    Ok(())
}
pub(super) fn finish(
    path: &Path,
    bytes: &[u8],
    deadline: std::time::Instant,
    interrupt: crate::automation::command_interrupt::Interrupt,
    observed_write: &mut impl FnMut() -> DynResult<()>,
) -> DynResult<()> {
    let cancel = interrupt.cancellation();
    Budget {
        until: deadline,
        cancel: &cancel,
    }
    .guard()?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err("cache preparation projection exceeds64MiB".into());
    }
    let mut temporary =
        tempfile::NamedTempFile::new_in(path.parent().ok_or("preparation parent absent")?)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    Budget {
        until: deadline,
        cancel: &cancel,
    }
    .guard()?;
    let file = temporary.persist_noclobber(path)?;
    let observation = observed_write();
    let guard = Budget {
        until: deadline,
        cancel: &cancel,
    }
    .guard();
    // Handlers stay installed through the fresh candidate's durable write.
    let finished = interrupt.finish();
    let final_guard = Budget {
        until: deadline,
        cancel: &cancel,
    }
    .guard();
    if observation.is_err() || guard.is_err() || finished.is_err() || final_guard.is_err() {
        owned(path, &file)?;
        std::fs::remove_file(path)?;
    }
    observation?;
    guard?;
    finished?;
    final_guard?;
    Ok(())
}
