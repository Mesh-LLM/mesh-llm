//! Cache matrix final receipt signal ownership and owned-file terminal downgrade.
use super::*;
use std::io::{Seek as _, SeekFrom, Write as _};
use std::time::Instant;
#[cfg(test)]
#[path = "final_publication_tests.rs"]
mod tests;
struct ReceiptFile {
    path: std::path::PathBuf,
    file: std::fs::File,
}
impl ReceiptFile {
    fn new(path: &Path) -> DynResult<Self> {
        let temporary =
            tempfile::NamedTempFile::new_in(path.parent().ok_or("receipt parent absent")?)?;
        Ok(Self {
            path: path.into(),
            file: temporary.persist_noclobber(path)?,
        })
    }
    fn write(&mut self, value: &Value) -> DynResult<()> {
        self.file.set_len(0)?;
        let bytes = serde_json::to_vec_pretty(value)?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err("cache matrix projection exceeds64MiB".into());
        }
        let metadata = std::fs::symlink_metadata(&self.path)?;
        if !metadata.is_file() {
            return Err("final receipt ownership changed".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let own = self.file.metadata()?;
            if own.dev() != metadata.dev() || own.ino() != metadata.ino() {
                return Err("final receipt ownership changed".into());
            }
        }
        // The retained descriptor is already truncated before replacement encoding.
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&bytes)?;
        self.file.set_len(bytes.len() as u64)?;
        self.file.sync_all()?;
        Ok(())
    }
}
/// Keep actual handlers installed through receipt writes and bounded output emission.
pub(super) fn finish(
    receipt: &mut Value,
    path: &Path,
    deadline: Instant,
    interrupt: crate::automation::command_interrupt::Interrupt,
    emit: &mut impl FnMut(&Value) -> DynResult<()>,
) -> DynResult<()> {
    let cancel = interrupt.cancellation();
    super::finalize(
        receipt,
        cancel.is_cancelled(),
        Instant::now() >= deadline,
        true,
    );
    let mut owned = ReceiptFile::new(path)?;
    owned.write(receipt)?;
    let emission = emit(receipt);
    if emission.is_err() {
        receipt["status"] = serde_json::json!("incomplete");
        receipt["output_publication_failed"] = serde_json::json!(true);
    }
    let cancelled = cancel.is_cancelled();
    let expired = Instant::now() >= deadline;
    super::finalize(receipt, cancelled, expired, true);
    if emission.is_err() || cancelled || expired {
        owned.write(receipt)?;
    }
    // No successful receipt serialization or write occurs after unregistering handlers.
    let finished = interrupt.finish();
    let cancelled = cancel.is_cancelled();
    let expired = Instant::now() >= deadline;
    super::finalize(receipt, cancelled, expired, finished.is_ok());
    if cancelled || expired || finished.is_err() {
        owned.write(receipt)?;
    }
    emission?;
    finished?;
    // Failure-only rewrites cannot lead to a successful return after their I/O.
    let final_expired = Instant::now() >= deadline;
    if final_expired && !expired {
        super::finalize(receipt, cancelled, true, true);
        owned.write(receipt)?;
    }
    if cancelled || final_expired {
        return Err("final receipt terminal admission refused".into());
    }
    Ok(())
}
