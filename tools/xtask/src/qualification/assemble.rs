//! `qualify assemble`: assemble a Skippy qualification receipt from executed
//! suite evidence and self-validate it before publishing.

use super::contract::{self, Json};
use super::validate;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use serde_json::{Map, Value, json};
use std::path::Path;

const USAGE: &str = "qualify assemble --product-dir <dir> --availability <file> --evidence-dir <dir> [--hardware <file>] --source-sha <sha> --plan-digest <digest> --row-id <row> --output <file>";

#[allow(clippy::too_many_arguments)]
fn assemble(
    root: &Path,
    product_dir: &Path,
    availability_path: &Path,
    evidence_dir: &Path,
    hardware_path: Option<&Path>,
    source_sha: &str,
    plan_digest: &str,
    row_id: &str,
) -> Result<Json, String> {
    let product_manifest = product_dir.join("product-manifest.json");
    let product = contract::load_json(&product_manifest)?;
    let availability = contract::load_json(availability_path)?;
    let state = availability
        .get("state")
        .and_then(Json::as_str)
        .unwrap_or("");
    contract::require(
        matches!(state, "available" | "hardware-unavailable"),
        "unknown hardware availability state",
    )?;
    contract::require(
        state != "available" || hardware_path.is_some(),
        "hardware evidence is required when execution is available",
    )?;
    // An unavailable row cannot claim hardware use, so any supplied path is
    // ignored rather than trusted.
    let hardware_path = if state == "available" {
        hardware_path
    } else {
        None
    };

    let mut suites = Map::new();
    for name in contract::suite_names() {
        if state == "hardware-unavailable" && name != "packaging-runtime" {
            suites.insert(name.to_owned(), json!({"status": "not-executed"}));
            continue;
        }
        let evidence_path = evidence_dir.join(format!("{name}.json"));
        let evidence = contract::load_json(&evidence_path)?;
        contract::require(
            evidence.get("status").and_then(Json::as_str) == Some("passed"),
            &format!("{name} evidence did not pass"),
        )?;
        suites.insert(
            name.to_owned(),
            json!({
                "status": "passed",
                "cases": evidence.get("executed_cases").cloned().unwrap_or(Json::Null).to_value(),
                "models": evidence.get("models").cloned().unwrap_or(Json::Null).to_value(),
                "evidence_file": format!("{name}.json"),
                "evidence_sha256": contract::digest(&evidence_path)?,
            }),
        );
    }

    let hardware = match hardware_path {
        Some(path) => contract::load_json(path)?.to_value(),
        None => Value::Null,
    };
    let receipt = json!({
        "schema_version": 1,
        "source_sha": source_sha,
        "plan_digest": plan_digest,
        "row_id": row_id,
        "product_manifest_sha256": contract::digest(&product_manifest)?,
        "availability_sha256": contract::digest(availability_path)?,
        "status": if state == "available" { "qualified" } else { "hardware-unavailable" },
        "hardware": hardware,
        "suites": Value::Object(suites),
    });
    let receipt_json = contract::parse_strict(receipt.to_string().as_bytes())?;
    validate::validate_receipt(
        root,
        &receipt_json,
        &product,
        &product_manifest,
        product_dir,
        &availability,
        availability_path,
        evidence_dir,
        source_sha,
        plan_digest,
        row_id,
    )?;
    Ok(receipt_json)
}

pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--product-dir",
            "--availability",
            "--evidence-dir",
            "--hardware",
            "--source-sha",
            "--plan-digest",
            "--row-id",
            "--output",
        ],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let required: Vec<&str> = GRAMMAR
        .values
        .iter()
        .filter(|name| **name != "--hardware" && parsed.last(name).is_none())
        .copied()
        .collect();
    if !required.is_empty() || !parsed.positionals.is_empty() {
        return GRAMMAR.error(&format!(
            "the following arguments are required: {}",
            required.join(", ")
        ));
    }
    let value = |name: &str| parsed.last(name).unwrap_or_default();
    let output = std::path::PathBuf::from(value("--output"));
    let result = assemble(
        root,
        Path::new(value("--product-dir")),
        Path::new(value("--availability")),
        Path::new(value("--evidence-dir")),
        parsed.last("--hardware").map(Path::new),
        value("--source-sha"),
        value("--plan-digest"),
        value("--row-id"),
    )
    .and_then(|receipt| {
        contract::write_atomic(&output, contract::render(&receipt.to_value())?.as_bytes())?;
        contract::digest(&output)
    });
    match result {
        Ok(digest) => CheckReport::success(format!("{digest}\n")),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("Skippy qualification assembly rejected: {error}\n"),
        ),
    }
}
