//! Keep owned report handles through terminal admission; never reopen foreign paths.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::Path,
};

const NAMES: [&str; 4] = [
    "moe-expert-smoke.json",
    "moe-expert-smoke.md",
    "moe-expert-smoke-table.json",
    "moe-expert-smoke-table.md",
];
pub(super) struct Reports {
    files: Vec<File>,
    publication_failed: bool,
}
fn encode(receipt: &Value) -> DynResult<[Vec<u8>; 4]> {
    let rows: Vec<Value> = receipt["cases"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|case| {
            case["observation"]["rows"]
                .as_array()
                .into_iter()
                .flatten()
                .cloned()
        })
        .collect();
    let aggregate = serde_json::to_vec_pretty(receipt)?;
    let table = serde_json::to_vec_pretty(
        &json!({"status":receipt["status"],"rows":rows,"scope":"observations; admission follows completed parent receipt"}),
    )?;
    let markdown = super::render::markdown(receipt).into_bytes();
    let bytes = [aggregate, markdown.clone(), table, markdown];
    if bytes.iter().any(|b| b.len() > 32 * 1024 * 1024 + 4096) {
        return Err("MoE report exceeds bounded observation plus terminal metadata".into());
    }
    Ok(bytes)
}
fn rewrite(reports: &mut Reports, receipt: &Value) -> DynResult<()> {
    // Revoke the eligible aggregate before encoding the downgraded projection.
    // If encoding or I/O fails, the old completed JSON cannot remain admitted.
    for file in &mut reports.files {
        file.set_len(0)?;
    }
    let bytes = encode(receipt)?;
    for (index, file) in reports.files.iter_mut().enumerate() {
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&bytes[index])?;
        file.sync_all()?;
    }
    Ok(())
}
fn write_fresh(
    output: &Path,
    bytes: &[Vec<u8>; 4],
    reports: &mut Reports,
    after: &mut impl FnMut(usize),
) -> DynResult<()> {
    for (index, name) in NAMES.iter().enumerate() {
        let mut temporary = tempfile::NamedTempFile::new_in(output)?;
        temporary.write_all(&bytes[index])?;
        temporary.as_file().sync_all()?;
        let file = temporary.persist_noclobber(output.join(name))?;
        reports.files.push(file);
        after(index);
    }
    Ok(())
}
pub(super) fn publish(
    output: &Path,
    receipt: &mut Value,
    guard: &mut impl FnMut() -> (bool, bool),
    after: &mut impl FnMut(usize),
) -> DynResult<Reports> {
    let candidate = encode(receipt)?;
    let (cancelled, expired) = guard();
    super::finalize(receipt, cancelled, expired, true);
    let bytes = if cancelled || expired {
        encode(receipt)?
    } else {
        candidate
    };
    let mut reports = Reports {
        files: Vec::new(),
        publication_failed: false,
    };
    if write_fresh(output, &bytes, &mut reports, after).is_err() {
        reports.publication_failed = true;
        receipt["status"] = json!("incomplete");
        receipt["publication_refused"] = json!(true);
    }
    let (cancelled, expired) = guard();
    super::finalize(receipt, cancelled, expired, true);
    if cancelled || expired || reports.publication_failed {
        rewrite(&mut reports, receipt)?;
    }
    Ok(reports)
}
pub(super) fn finish(
    reports: &mut Reports,
    receipt: &mut Value,
    cancelled: bool,
    expired: bool,
    finish_ok: bool,
) -> DynResult<()> {
    super::finalize(receipt, cancelled, expired, finish_ok);
    if cancelled || expired || !finish_ok {
        rewrite(reports, receipt)?;
    }
    if cancelled || expired || !finish_ok || reports.publication_failed {
        return Err("MoE terminal publication refused; owned observations retained".into());
    }
    Ok(())
}
