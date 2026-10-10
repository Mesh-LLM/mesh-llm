//! Upload receipt admission while the actual caller's signal latch remains installed.
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::{
    io::{Seek as _, SeekFrom, Write as _},
    path::Path,
};
pub(super) fn publish(
    root: &Path,
    name: &str,
    mut value: Value,
    mut check: impl FnMut() -> Result<()>,
    after_write: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let initial = check();
    if let Err(error) = &initial {
        refuse(&mut value, error.to_string());
    }
    let temporary = tempfile::NamedTempFile::new_in(root)?;
    let mut held = temporary.persist_noclobber(root.join(name))?;
    write(&mut held, &value)?;
    let result = initial.and(after_write()).and_then(|()| check());
    if let Err(error) = result {
        refuse(&mut value, error.to_string());
        write(&mut held, &value)?;
        return Err(error);
    }
    Ok(())
}
fn refuse(value: &mut Value, error: String) {
    value["status"] = json!("FAILED");
    value["terminal_refused"] = json!(true);
    value["publication"]["completed"] = json!(false);
    if value["publication"]["error"].is_null() {
        value["publication"]["error"] = json!(error);
    }
}
fn write(file: &mut std::fs::File, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > 1048576 {
        bail!("upload terminal receipt bound");
    }
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&bytes)?;
    file.set_len(bytes.len() as u64)?;
    file.sync_all()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upload_actual_final_receipt_downgrades_late_terminal_refusal_preserving_uploads() {
        for mode in ["success", "cancel", "deadline", "prior-error"] {
            let root = tempfile::tempdir().unwrap();
            let refused = std::cell::Cell::new(false);
            let prior = mode == "prior-error";
            let value = json!({"status":if prior{"FAILED"}else{"PUBLISHED"},"publication":{"completed":!prior,"error":if prior{json!("earlier upload failure")}else{Value::Null},"unlinked":true,"attempts":[{"commit_oid":"c".repeat(40),"remote_verified":true}]}});
            let result = publish(
                root.path(),
                "upload.json",
                value,
                || {
                    if refused.get() {
                        bail!("late {mode}");
                    }
                    Ok(())
                },
                || {
                    let before: Value =
                        serde_json::from_slice(&std::fs::read(root.path().join("upload.json"))?)
                            .unwrap();
                    assert_eq!(before["publication"]["completed"], !prior);
                    refused.set(mode != "success");
                    Ok(())
                },
            );
            assert_eq!(result.is_ok(), mode == "success");
            let observed: Value =
                serde_json::from_slice(&std::fs::read(root.path().join("upload.json")).unwrap())
                    .unwrap();
            assert_eq!(observed["publication"]["completed"], mode == "success");
            assert_eq!(observed["publication"]["unlinked"], true);
            assert_eq!(
                observed["publication"]["attempts"][0]["remote_verified"],
                true
            );
            if prior {
                assert_eq!(observed["publication"]["error"], "earlier upload failure");
            }
        }
    }
}
