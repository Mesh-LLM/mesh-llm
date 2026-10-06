//! Probe observation publication retains actual signal custody and owned-file failure evidence.
use super::*;
use std::io::{Seek as _, SeekFrom, Write as _};
struct ObservationFile {
    path: std::path::PathBuf,
    file: std::fs::File,
}
impl ObservationFile {
    fn new(path: &Path) -> DynResult<Self> {
        let temporary =
            tempfile::NamedTempFile::new_in(path.parent().ok_or("observation parent")?)?;
        Ok(Self {
            path: path.into(),
            file: temporary.persist_noclobber(path)?,
        })
    }
    fn write(&mut self, evidence: &Value) -> DynResult<()> {
        let bytes = serde_json::to_vec_pretty(evidence)?;
        if bytes.len() > 1048576 {
            return Err("probe observations exceed 1MiB".into());
        }
        let metadata = std::fs::symlink_metadata(&self.path)?;
        if !metadata.is_file() {
            return Err("probe observation ownership changed".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let own = self.file.metadata()?;
            if own.dev() != metadata.dev() || own.ino() != metadata.ino() {
                return Err("probe observation ownership changed".into());
            }
        }
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&bytes)?;
        self.file.set_len(bytes.len() as u64)?;
        self.file.sync_all()?;
        Ok(())
    }
}
fn terminal(
    evidence: &mut Value,
    until: Instant,
    cancel: &Cancellation,
    finish_failed: bool,
) -> bool {
    let cancelled = cancel.is_cancelled();
    let expired = Instant::now() >= until;
    evidence["terminal_cancelled"] = json!(cancelled);
    evidence["terminal_deadline_expired"] = json!(expired);
    let finish_failed = finish_failed || evidence["interrupt_finish_failed"] == true;
    evidence["interrupt_finish_failed"] = json!(finish_failed);
    let failed = cancelled || expired || finish_failed;
    if failed {
        evidence["phase_error"] = json!(true);
    }
    failed
}
/// The callback is the local output seam, invoked only while actual handlers remain installed.
/// All receipts remain observations only, even on a clean component return.
pub(super) fn finish(
    evidence: &mut Value,
    path: &Path,
    until: Instant,
    interrupt: Interrupt,
    phase: DynResult<()>,
    after_write: &mut impl FnMut() -> DynResult<()>,
) -> DynResult<()> {
    finish_with(
        evidence,
        path,
        until,
        interrupt,
        phase,
        after_write,
        |scope| scope.finish().map_err(Into::into),
    )
}
pub(super) fn finish_with(
    evidence: &mut Value,
    path: &Path,
    until: Instant,
    interrupt: Interrupt,
    phase: DynResult<()>,
    after_write: &mut impl FnMut() -> DynResult<()>,
    finish_scope: impl FnOnce(Interrupt) -> DynResult<()>,
) -> DynResult<()> {
    let cancel = interrupt.cancellation();
    evidence["phase_error"] = json!(phase.is_err());
    terminal(evidence, until, &cancel, false);
    let mut owned = ObservationFile::new(path)?;
    let written = owned.write(evidence);
    let emitted = if written.is_ok() {
        after_write()
    } else {
        Ok(())
    };
    let publication_failed = written.is_err() || emitted.is_err();
    evidence["observation_publication_failed"] = json!(publication_failed);
    if publication_failed {
        evidence["phase_error"] = json!(true);
    }
    let refused = terminal(evidence, until, &cancel, false);
    if publication_failed || refused {
        owned.write(evidence)?;
    }
    let finished = finish_scope(interrupt);
    let refused = terminal(evidence, until, &cancel, finished.is_err());
    // No healthy observation serialization/write occurs after unregistering actual handlers.
    if refused {
        owned.write(evidence)?;
    }
    let result = phase.and(written).and(emitted).and(finished);
    if result.is_err() || refused {
        // Failure-only I/O never changes a refused component into success.
        terminal(evidence, until, &cancel, false);
        evidence["phase_error"] = json!(true);
        owned.write(evidence)?;
        return result.and(Err(
            "quant probe terminal/publication refused; observations retained".into(),
        ));
    }
    Ok(())
}
