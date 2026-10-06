//! Scheduler receipt publication with a live command signal scope and retained file custody.
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Cancellation};
use serde_json::Value;
use std::{
    io::{Seek as _, SeekFrom, Write as _},
    path::{Path, PathBuf},
    time::Instant,
};
#[cfg(test)]
#[path = "final_publication_tests.rs"]
mod tests;
pub(super) struct ReceiptFile {
    path: PathBuf,
    file: std::fs::File,
}
impl ReceiptFile {
    pub(super) fn new(path: &Path) -> DynResult<Self> {
        let temporary =
            tempfile::NamedTempFile::new_in(path.parent().ok_or("receipt parent absent")?)?;
        Ok(Self {
            path: path.into(),
            file: temporary.persist_noclobber(path)?,
        })
    }
    pub(super) fn write(&mut self, value: &Value) -> DynResult<()> {
        self.file.set_len(0)?;
        let bytes = serde_json::to_vec_pretty(value)?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err("MTP receipt exceeds bounded publication".into());
        }
        let metadata = std::fs::symlink_metadata(&self.path)?;
        if !metadata.is_file() {
            return Err("MTP receipt ownership changed".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let own = self.file.metadata()?;
            if own.dev() != metadata.dev() || own.ino() != metadata.ino() {
                return Err("MTP receipt ownership changed".into());
            }
        }
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&bytes)?;
        self.file.set_len(bytes.len() as u64)?;
        self.file.sync_all()?;
        Ok(())
    }
    /// Healthy final bytes are written while handlers live. After finish, only refusal downgrades write.
    pub(super) fn finish(
        &mut self,
        receipt: &mut Value,
        mut result: DynResult<()>,
        interrupt: Interrupt,
        deadline: Instant,
        emit: &mut impl FnMut(&Value) -> DynResult<()>,
    ) -> DynResult<()> {
        let cancellation = interrupt.cancellation();
        terminal(receipt, &mut result, &cancellation, deadline);
        status(receipt, &result);
        if let Err(error) = self.write(receipt) {
            refusal(receipt, &mut result, error.to_string());
        }
        terminal(receipt, &mut result, &cancellation, deadline);
        if result.is_ok()
            && let Err(error) = emit(receipt)
        {
            refusal(receipt, &mut result, error.to_string());
        }
        terminal(receipt, &mut result, &cancellation, deadline);
        let finished = interrupt.finish();
        if let Err(error) = finished {
            refusal(receipt, &mut result, error.to_string());
        }
        terminal(receipt, &mut result, &cancellation, deadline);
        if result.is_err() {
            status(receipt, &result);
            self.write(receipt)?;
            // Only an already-refused result reaches this post-write check.
            // Healthy admission is final before the downgrade branch.
            terminal(receipt, &mut result, &cancellation, deadline);
        }

        result
    }
}
fn status(receipt: &mut Value, result: &DynResult<()>) {
    receipt["status"] = if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .into();
    receipt["orchestration_complete"] = result.is_ok().into();
    if let Err(error) = result
        && receipt.get("failure").is_none()
    {
        receipt["failure"] = error.to_string().into();
    }
}
fn refusal(receipt: &mut Value, result: &mut DynResult<()>, reason: String) {
    receipt["terminal_refusal"] = reason.clone().into();
    if result.is_ok() {
        *result = Err(reason.into());
    }
}
fn terminal(
    receipt: &mut Value,
    result: &mut DynResult<()>,
    cancel: &Cancellation,
    deadline: Instant,
) {
    if cancel.is_cancelled() {
        receipt["cancelled"] = true.into();
        refusal(receipt, result, "MTP final publication cancelled".into());
    }
    if Instant::now() >= deadline {
        receipt["deadline_exceeded"] = true.into();
        refusal(
            receipt,
            result,
            "MTP final publication deadline exceeded".into(),
        );
    }
}
