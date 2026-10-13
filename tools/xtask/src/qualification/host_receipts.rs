//! `qualify host-receipts`: verify and record the standalone Skippy receipts a
//! Mesh host was built against. The host must not be produced from an
//! unqualified Skippy product: this consumes the per-row qualification
//! receipts for the same source and plan, requires the platform's mandatory
//! row to be qualified, and records every receipt digest so the host product
//! carries its provenance.
//!
//! [`validate_record`] is the structural re-check `product compose
//! --qualification-receipts` applies to the record file embedded in a bundle;
//! the record's provenance is established here.

use super::contract::{self, Json};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

const USAGE: &str = "qualify host-receipts --receipts <dir> --source-sha <sha> --plan-digest <digest> --platform <linux|macos|windows> --output <file>";

/// Row the protected hosted graph always executes for each platform.
const REQUIRED_ROW: [(&str, &str); 3] = [
    ("linux", "linux-cpu"),
    ("macos", "macos-metal"),
    ("windows", "windows-cpu"),
];

pub(crate) const RECORD_NAME: &str = "skippy-qualification-receipts.json";

/// A validated host qualification record.
pub(crate) struct HostRecord {
    pub(crate) source_sha: String,
    pub(crate) plan_digest: String,
    pub(crate) platform: String,
    pub(crate) required_row: String,
    /// `(row_id, status, receipt_sha256)` sorted by row.
    pub(crate) receipts: Vec<(String, String, String)>,
}

/// Structural re-check of a record bound into a composed product: schema,
/// digest formats, platform, row prefixes, statuses, uniqueness, and the
/// required row's qualification.
pub(crate) fn validate_record(record: &Json) -> Result<HostRecord, String> {
    contract::require(
        record.get("schema_version").and_then(Json::as_int) == Some(1),
        "qualification receipt record has an unsupported schema",
    )?;
    let source_sha = record
        .get("source_sha")
        .and_then(Json::as_str)
        .filter(|value| contract::is_git_sha(value))
        .ok_or("qualification receipt record has an invalid source revision")?;
    let plan_digest = record
        .get("plan_digest")
        .and_then(Json::as_str)
        .filter(|value| contract::is_sha256(value))
        .ok_or("qualification receipt record has an invalid plan digest")?;
    let platform = record
        .get("platform")
        .and_then(Json::as_str)
        .filter(|value| REQUIRED_ROW.iter().any(|(name, _)| name == value))
        .ok_or("qualification receipt record names an unknown platform")?;
    let required_row = record
        .get("required_row")
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("qualification receipt record names no required row")?;
    let receipts = record
        .get("receipts")
        .and_then(Json::as_array)
        .filter(|receipts| !receipts.is_empty())
        .ok_or("qualification receipt record carries no receipts")?;
    let mut bound = Vec::new();
    for entry in receipts {
        contract::require(
            entry.as_object().is_some(),
            "qualification receipt record carries a malformed receipt",
        )?;
        let row_id = entry.get("row_id").and_then(Json::as_str).unwrap_or("");
        contract::require(
            !row_id.is_empty() && row_id.starts_with(&format!("{platform}-")),
            "qualification receipt record carries a foreign row",
        )?;
        let status = entry.get("status").and_then(Json::as_str).unwrap_or("");
        contract::require(
            matches!(status, "qualified" | "hardware-unavailable"),
            &format!("qualification receipt for {row_id} has an unknown status"),
        )?;
        let digest = entry
            .get("receipt_sha256")
            .and_then(Json::as_str)
            .unwrap_or("");
        contract::require(
            contract::is_sha256(digest),
            &format!("qualification receipt for {row_id} has an invalid digest"),
        )?;
        contract::require(
            !bound
                .iter()
                .any(|(row, _, _): &(String, String, String)| row == row_id),
            &format!("qualification receipt record repeats {row_id}"),
        )?;
        bound.push((row_id.to_owned(), status.to_owned(), digest.to_owned()));
    }
    contract::require(
        bound
            .iter()
            .find(|(row, _, _)| row == required_row)
            .map(|(_, status, _)| status.as_str())
            == Some("qualified"),
        &format!("required row {required_row} is not qualified in the receipt record"),
    )?;
    bound.sort();
    Ok(HostRecord {
        source_sha: source_sha.to_owned(),
        plan_digest: plan_digest.to_owned(),
        platform: platform.to_owned(),
        required_row: required_row.to_owned(),
        receipts: bound,
    })
}

fn receipt_files(receipts_dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(receipts_dir)
        .map_err(|error| format!("{}: {error}", receipts_dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            receipt_files(&path, files)?;
        } else if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            files.push(path);
        }
    }
    Ok(())
}

fn verify(
    receipts_dir: &Path,
    source_sha: &str,
    plan_digest: &str,
    platform: &str,
) -> Result<Value, String> {
    let required = REQUIRED_ROW
        .iter()
        .find(|(name, _)| *name == platform)
        .map(|(_, row)| *row)
        .ok_or_else(|| format!("unknown platform: {platform}"))?;
    contract::require(contract::is_git_sha(source_sha), "invalid source SHA")?;
    contract::require(contract::is_sha256(plan_digest), "invalid plan digest")?;
    let mut files = Vec::new();
    receipt_files(receipts_dir, &mut files)?;
    files.sort();
    contract::require(
        !files.is_empty(),
        "no standalone qualification receipts were downloaded",
    )?;
    let mut rows: Vec<(String, String, String)> = Vec::new();
    for path in &files {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("receipt.json");
        let receipt = contract::load_json(path)?;
        contract::require(
            receipt.get("schema_version").and_then(Json::as_int) == Some(1),
            &format!("{name}: unknown receipt schema"),
        )?;
        contract::require(
            receipt.get("source_sha").and_then(Json::as_str) == Some(source_sha),
            &format!("{name}: receipt belongs to another source revision"),
        )?;
        contract::require(
            receipt.get("plan_digest").and_then(Json::as_str) == Some(plan_digest),
            &format!("{name}: receipt belongs to another CI plan"),
        )?;
        let row_id = receipt.get("row_id").and_then(Json::as_str).unwrap_or("");
        contract::require(
            contract::CORE_ROWS.contains(&row_id),
            &format!("{name}: receipt names an unknown core row"),
        )?;
        contract::require(
            row_id.starts_with(&format!("{platform}-")),
            &format!("{name}: receipt belongs to another platform"),
        )?;
        contract::require(
            !rows.iter().any(|(row, _, _)| row == row_id),
            &format!("{name}: duplicate receipt for {row_id}"),
        )?;
        let status = receipt.get("status").and_then(Json::as_str).unwrap_or("");
        contract::require(
            matches!(status, "qualified" | "hardware-unavailable"),
            &format!("{name}: invalid qualification status"),
        )?;
        rows.push((
            row_id.to_owned(),
            status.to_owned(),
            contract::digest(path)?,
        ));
    }
    contract::require(
        rows.iter().any(|(row, _, _)| row == required),
        &format!("required row {required} produced no receipt"),
    )?;
    contract::require(
        rows.iter()
            .find(|(row, _, _)| row == required)
            .map(|(_, status, _)| status.as_str())
            == Some("qualified"),
        &format!("required row {required} is not qualified"),
    )?;
    rows.sort();
    Ok(json!({
        "schema_version": 1,
        "source_sha": source_sha,
        "plan_digest": plan_digest,
        "platform": platform,
        "required_row": required,
        "receipts": rows
            .iter()
            .map(|(row_id, status, digest)| json!({
                "row_id": row_id,
                "status": status,
                "receipt_sha256": digest,
            }))
            .collect::<Vec<_>>(),
    }))
}

pub(super) fn run(args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--receipts",
            "--source-sha",
            "--plan-digest",
            "--platform",
            "--output",
        ],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let missing: Vec<&str> = GRAMMAR
        .values
        .iter()
        .filter(|name| parsed.last(name).is_none())
        .copied()
        .collect();
    if !missing.is_empty() || !parsed.positionals.is_empty() {
        return GRAMMAR.error(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    let value = |name: &str| parsed.last(name).unwrap_or_default();
    let result = verify(
        Path::new(value("--receipts")),
        value("--source-sha"),
        value("--plan-digest"),
        value("--platform"),
    )
    .and_then(|record| {
        contract::write_atomic(
            Path::new(value("--output")),
            contract::render(&record)?.as_bytes(),
        )
    });
    match result {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("standalone qualification receipts rejected: {error}\n"),
        ),
    }
}
