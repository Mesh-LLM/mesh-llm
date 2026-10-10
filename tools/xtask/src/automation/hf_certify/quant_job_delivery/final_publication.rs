//! Keep signal handlers through final quant delivery writes and locator emission.
use super::*;
use std::{fs::File, path::PathBuf};
pub(super) struct OwnedReceipt {
    path: PathBuf,
    file: Option<File>,
}
impl OwnedReceipt {
    pub(super) fn new(path: PathBuf) -> Self {
        Self { path, file: None }
    }
    pub(super) fn write(&mut self, value: &Value) -> DynResult<()> {
        use std::io::Write as _;
        let bytes = serde_json::to_vec_pretty(value)?;
        if bytes.len() > 1048576 {
            return Err("quant receipt exceeds 1MiB bounded export".into());
        }
        if let Some(file) = &self.file {
            let current = std::fs::symlink_metadata(&self.path)?;
            if !current.is_file() {
                return Err("quant receipt foreign file type".into());
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt as _;
                let original = file.metadata()?;
                if original.dev() != current.dev() || original.ino() != current.ino() {
                    return Err("quant receipt foreign inode".into());
                }
            }
            #[cfg(not(unix))]
            {
                let _ = file;
                return Err("quant receipt downgrade requires Unix identity".into());
            }
        }
        let mut next =
            tempfile::NamedTempFile::new_in(self.path.parent().ok_or("quant receipt parent")?)?;
        next.write_all(&bytes)?;
        next.as_file().sync_all()?;
        let file = if self.file.is_some() {
            next.persist(&self.path)?
        } else {
            next.persist_noclobber(&self.path)?
        };
        self.file = Some(file);
        Ok(())
    }
}
fn mark(delivery: &mut Value, completed: bool) {
    delivery["status"] = json!(if completed { "DELIVERED" } else { "FAILED" });
    delivery["native_completed"] = json!(completed);
    if !delivery["locator"].is_null() {
        delivery["locator"]["delivery_complete"] = json!(completed);
    }
}
fn emit(delivery: &Value) -> DynResult<()> {
    if !delivery["locator"].is_null() {
        use std::io::Write as _;
        writeln!(
            std::io::stdout().lock(),
            "MESH_NATIVE_DELIVERY {}",
            serde_json::to_string(&delivery["locator"])?
        )?;
    }
    Ok(())
}
struct Observations<'a> {
    native: &'a mut Value,
    native_file: &'a mut OwnedReceipt,
    delivery: &'a mut Value,
}
fn downgrade(observed: &mut Observations<'_>, writer: &mut OwnedReceipt) -> DynResult<()> {
    observed.native["status"] = json!("FAILED");
    if observed.native["error"].is_null() {
        observed.native["error"] =
            json!("quant Jobs final delivery admission refused; observations retained");
    }
    if observed.native["operator"].is_object() {
        observed.native["operator"]["completed_job"] = json!(false);
    }
    mark(observed.delivery, false);
    observed.native_file.write(observed.native)?;
    writer.write(observed.delivery)?;
    emit(observed.delivery)
}
pub(super) fn finish(
    root: &Path,
    native: &mut Value,
    native_file: &mut OwnedReceipt,
    delivery: &mut Value,
    candidate: bool,
    until: Instant,
    interrupt: Interrupt,
) -> DynResult<()> {
    finish_owned(
        root,
        Observations {
            native,
            native_file,
            delivery,
        },
        candidate,
        until,
        interrupt,
        |_| Ok(()),
    )
}
fn finish_owned(
    root: &Path,
    mut observed: Observations<'_>,
    candidate: bool,
    until: Instant,
    interrupt: Interrupt,
    mut after: impl FnMut(&str) -> DynResult<()>,
) -> DynResult<()> {
    let cancel = interrupt.cancellation();
    let mut writer = OwnedReceipt::new(root.join("native-job-delivery.json"));
    let mut completed = candidate && guard(until, &cancel).is_ok();
    mark(observed.delivery, completed);
    let writes = (|| {
        writer.write(observed.delivery)?;
        after("write")?;
        guard(until, &cancel)?;
        emit(observed.delivery)?;
        after("emit")?;
        guard(until, &cancel)
    })();
    let finished = interrupt.finish();
    completed &= writes.is_ok() && finished.is_ok() && guard(until, &cancel).is_ok();
    if !completed {
        downgrade(&mut observed, &mut writer)?;
        return Err("quant Jobs final admission refused; inspect retained observations".into());
    }
    // No successful serialization or I/O after unregistering handlers.
    Ok(())
}

#[cfg(test)]
#[path = "final_publication/tests.rs"]
mod tests;
