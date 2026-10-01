use super::{ResultRow, execution::Session};
use crate::command::DynResult;
use std::path::Path;

pub(super) fn finish(
    directory: &Path,
    report: &Session,
    joined: bool,
    mut rows: Vec<ResultRow>,
) -> DynResult<bool> {
    let success = joined && report.as_ref().is_ok_and(|report| report.success());
    match report {
        Ok(report) => {
            if !success {
                rows.push(ResultRow {
                    status: "FAIL",
                    name: "session",
                    message: format!("outcome={:?}; failure={:?}", report.outcome, report.failure),
                });
            }
            if let Some(reason) = &report.rejection {
                rows.push(ResultRow {
                    status: "FAIL",
                    name: "narrative",
                    message: reason.clone(),
                });
            }
            let receipts: Vec<_> = report.members.iter().map(|member| serde_json::json!({
                "name":String::from_utf8_lossy(member.member.name()),"generation":member.member.generation(),
                "pid":member.process.pid,"disposition":member.disposition.label(),"exit_code":member.process.status.and_then(|status|status.code()),
                "cleanup_complete":member.process.cleanup.complete,"cleanup_forced":member.process.cleanup.forced,
                "completion":member.completion.as_ref().map(|receipt|serde_json::json!({"status":receipt.status,"deadline_ms":receipt.deadline.as_millis(),"elapsed_ms":receipt.elapsed.as_millis()}))
            })).collect();
            write_json(
                &directory.join("processes.json"),
                &serde_json::json!(receipts),
            )?;
        }
        Err(error) => rows.push(ResultRow {
            status: "FAIL",
            name: "session",
            message: error.to_string(),
        }),
    }
    rows.push(ResultRow {
        status: if success { "PASS" } else { "FAIL" },
        name: "cleanup",
        message: "retained session finalization".into(),
    });
    let mut jsonl = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut jsonl, row)?;
        jsonl.push(b'\n');
    }
    std::fs::write(directory.join("results.jsonl"), jsonl)?;
    write_json(
        &directory.join("summary.json"),
        &serde_json::json!({
            "overall":if success {"pass"}else{"fail"},"evidence_dir":directory,
            "counts":{"pass":rows.iter().filter(|row|row.status=="PASS").count(),"fail":rows.iter().filter(|row|row.status=="FAIL").count()},"results":rows
        }),
    )?;
    Ok(success)
}

pub(super) fn write_json(path: &Path, value: &serde_json::Value) -> DynResult<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    std::fs::write(path, bytes)?;
    Ok(())
}
