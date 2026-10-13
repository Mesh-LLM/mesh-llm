//! Final report bundle publication; late refusal retains rows and disables promotion.
use super::*;
use std::io::{Seek as _, SeekFrom, Write as _};
const NAMES: [&str; 3] = [
    "cache-correctness-table.json",
    "cache-correctness-table.md",
    "batch-summary.json",
];
pub(super) struct Files {
    paths: [PathBuf; 3],
    owned: [Option<std::fs::File>; 3],
}
impl Files {
    pub(super) fn new(root: &Path) -> Self {
        Self {
            paths: NAMES.map(|n| root.join(n)),
            owned: std::array::from_fn(|_| None),
        }
    }
    pub(super) fn write(&mut self, index: usize, bytes: &[u8]) -> DynResult<()> {
        if bytes.len() > 1024 * 1024 {
            return Err("correctness table/summary exceeds1MiB".into());
        }
        let path = &self.paths[index];
        if self.owned[index].is_none() {
            self.owned[index] = Some(
                std::fs::OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create_new(true)
                    .open(path)?,
            );
        }
        let file = self.owned[index]
            .as_mut()
            .ok_or("owned correctness output")?;
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.is_file() {
            return Err("correctness output ownership changed".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let original = file.metadata()?;
            if original.dev() != metadata.dev() || original.ino() != metadata.ino() {
                return Err("correctness output ownership changed".into());
            }
        }
        // Rewrites use only the retained owned FD, never reopen or clobber a foreign path.
        file.seek(SeekFrom::Start(0))?;
        file.write_all(bytes)?;
        file.set_len(bytes.len() as u64)?;
        file.sync_all()?;
        Ok(())
    }
}
fn bytes(index: usize, rows: &[Value], summary: &Value) -> DynResult<Vec<u8>> {
    let bytes = match index {
        0 => serde_json::to_vec_pretty(rows)?,
        1 => markdown(rows).into_bytes(),
        _ => serde_json::to_vec_pretty(summary)?,
    };
    if bytes.len() > 1024 * 1024 {
        return Err("correctness table/summary exceeds1MiB".into());
    }
    Ok(bytes)
}
fn downgrade(rows: &mut [Value], summary: &mut Value, flags: (bool, bool, bool)) {
    terminal(rows, flags.0, flags.1, flags.2);
    summary["completed"] = json!(false);
    summary["terminal_refusal"] =
        json!({"cancelled":flags.0,"deadline_expired":flags.1,"interrupt_finish_failed":!flags.2});
}
pub(super) fn finish(
    rows: &mut [Value],
    summary: &mut Value,
    guard: &mut impl FnMut() -> (bool, bool, bool),
    writer: &mut impl FnMut(usize, &[u8]) -> DynResult<()>,
) -> DynResult<bool> {
    let flags = guard();
    let completed = terminal(rows, flags.0, flags.1, flags.2);
    summary["completed"] = json!(completed);
    for index in 0..3 {
        let value = bytes(index, rows, summary)?;
        let flags = guard();
        if flags.0 || flags.1 || !flags.2 {
            downgrade(rows, summary, flags);
            return retain_failed(rows, summary, writer);
        }
        writer(index, &value)?;
        let flags = guard();
        if flags.0 || flags.1 || !flags.2 {
            downgrade(rows, summary, flags);
            return retain_failed(rows, summary, writer);
        }
    }
    Ok(completed)
}
fn retain_failed(
    rows: &[Value],
    summary: &Value,
    writer: &mut impl FnMut(usize, &[u8]) -> DynResult<()>,
) -> DynResult<bool> {
    for index in 0..3 {
        writer(index, &bytes(index, rows, summary)?)?;
    }
    Ok(false)
}

/// Handler lifetime includes all final writes; finish refusal downgrades owned outputs.
pub(super) fn finish_owned(
    rows: &mut [Value],
    summary: &mut Value,
    interrupt: crate::automation::command_interrupt::Interrupt,
    deadline: Instant,
    writer: &mut impl FnMut(usize, &[u8]) -> DynResult<()>,
) -> DynResult<bool> {
    let cancellation = interrupt.cancellation();
    let observed = finish(
        rows,
        summary,
        &mut || {
            (
                cancellation.is_cancelled(),
                Instant::now() >= deadline,
                true,
            )
        },
        writer,
    );
    let finalization = interrupt.finish();
    let completed = observed?;
    let flags = (
        cancellation.is_cancelled(),
        Instant::now() >= deadline,
        finalization.is_ok(),
    );
    if flags.0 || flags.1 || !flags.2 {
        downgrade(rows, summary, flags);
        retain_failed(rows, summary, writer)?;
    }
    finalization?;
    Ok(completed && !flags.0 && !flags.1)
}
