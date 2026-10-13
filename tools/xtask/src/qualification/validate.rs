//! `qualify validate`: validate a standalone Skippy execution receipt against
//! its exact CI product. Every rule fails closed.

use super::contract::{self, Json};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::Path;

const USAGE: &str = "qualify validate --receipt <file> --product-manifest <file> --product-dir <dir> --availability <file> --evidence-dir <dir> --source-sha <sha> --plan-digest <digest> --row-id <row>";

fn field<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
    value.get(key)
}

fn text(value: Option<&Json>) -> Option<&str> {
    value.and_then(Json::as_str)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_receipt(
    root: &Path,
    receipt: &Json,
    product: &Json,
    product_manifest: &Path,
    product_dir: &Path,
    availability: &Json,
    availability_path: &Path,
    evidence_dir: &Path,
    source_sha: &str,
    plan_digest: &str,
    row_id: &str,
) -> Result<(), String> {
    let row = contract::catalog_row(root, row_id)?;
    contract::require(contract::is_git_sha(source_sha), "invalid source SHA")?;
    contract::require(contract::is_sha256(plan_digest), "invalid plan digest")?;
    contract::require(
        product.get("schema_version").and_then(Json::as_int) == Some(1)
            && text(field(product, "contract")) == Some("skippy-product-v1"),
        "invalid Skippy product manifest",
    )?;
    contract::validate_product_bytes(product, product_dir)?;
    for (key, expected) in [
        ("source_sha", source_sha.to_owned()),
        ("target", contract::row_field(&row, "target")?.to_owned()),
        ("backend", contract::row_field(&row, "backend")?.to_owned()),
    ] {
        contract::require(
            text(field(product, key)) == Some(expected.as_str()),
            &format!("product {key} differs from selected row"),
        )?;
    }
    contract::require(
        receipt.get("schema_version").and_then(Json::as_int) == Some(1),
        "unknown qualification receipt schema",
    )?;
    for (key, expected) in [
        ("source_sha", source_sha.to_owned()),
        ("plan_digest", plan_digest.to_owned()),
        ("row_id", row_id.to_owned()),
        (
            "product_manifest_sha256",
            contract::digest(product_manifest)?,
        ),
        ("availability_sha256", contract::digest(availability_path)?),
    ] {
        contract::require(
            text(field(receipt, key)) == Some(expected.as_str()),
            &format!("receipt {key} differs from selected product or plan"),
        )?;
    }
    validate_availability(receipt, availability, source_sha, plan_digest, row_id)?;
    validate_suites(
        root,
        receipt,
        availability,
        evidence_dir,
        product_manifest,
        source_sha,
        row_id,
    )?;
    validate_hardware(root, receipt, availability, &row)
}

fn validate_availability(
    receipt: &Json,
    availability: &Json,
    source_sha: &str,
    plan_digest: &str,
    row_id: &str,
) -> Result<(), String> {
    contract::require(
        availability.get("schema_version").and_then(Json::as_int) == Some(1),
        "unknown hardware availability schema",
    )?;
    contract::require(
        text(field(availability, "source_sha")) == Some(source_sha)
            && text(field(availability, "plan_digest")) == Some(plan_digest),
        "availability belongs to another source or plan",
    )?;
    contract::require(
        text(field(availability, "row_id")) == Some(row_id),
        "availability belongs to another row",
    )?;
    contract::require(
        text(field(availability, "policy_source")) == Some("protected-ci"),
        "hardware availability is not from the protected policy",
    )?;
    let state = text(field(availability, "state")).unwrap_or("");
    contract::require(
        matches!(state, "available" | "hardware-unavailable"),
        "invalid hardware availability state",
    )?;
    if contract::REQUIRED_AVAILABLE_ROWS.contains(&row_id) {
        contract::require(
            state == "available",
            "required available row cannot be hardware-unavailable",
        )?;
    }
    if state == "hardware-unavailable" {
        contract::require(
            text(field(availability, "reason")).is_some_and(|reason| !reason.trim().is_empty()),
            "unavailable hardware needs a reason",
        )?;
        contract::require(
            text(field(receipt, "status")) == Some("hardware-unavailable"),
            "unavailable row cannot claim qualification",
        )?;
    } else {
        contract::require(
            text(field(availability, "runner")).is_some_and(|runner| !runner.trim().is_empty()),
            "available hardware needs an approved runner",
        )?;
        contract::require(
            text(field(receipt, "status")) == Some("qualified"),
            "available row must be qualified",
        )?;
    }
    Ok(())
}

fn validate_suites(
    root: &Path,
    receipt: &Json,
    availability: &Json,
    evidence_dir: &Path,
    product_manifest: &Path,
    source_sha: &str,
    row_id: &str,
) -> Result<(), String> {
    let state = text(field(availability, "state")).unwrap_or("");
    let suites = receipt
        .get("suites")
        .and_then(Json::as_object)
        .ok_or("missing, duplicate, or unknown qualification suite")?;
    let names: std::collections::BTreeSet<&str> =
        suites.iter().map(|(name, _)| name.as_str()).collect();
    let expected = contract::suite_names();
    contract::require(
        names == expected,
        "missing, duplicate, or unknown qualification suite",
    )?;
    for (name, result) in suites {
        contract::require(
            result.as_object().is_some(),
            &format!("{name} result must be an object"),
        )?;
        if state == "hardware-unavailable" && name != "packaging-runtime" {
            let not_executed = Json::Object(vec![(
                "status".to_owned(),
                Json::String("not-executed".to_owned()),
            )]);
            contract::require(
                *result == not_executed,
                &format!("{name} claims execution on unavailable hardware"),
            )?;
            continue;
        }
        validate_suite(
            root,
            name,
            result,
            evidence_dir,
            product_manifest,
            source_sha,
            row_id,
        )?;
    }
    Ok(())
}

fn validate_suite(
    root: &Path,
    name: &str,
    result: &Json,
    evidence_dir: &Path,
    product_manifest: &Path,
    source_sha: &str,
    row_id: &str,
) -> Result<(), String> {
    contract::require(
        text(field(result, "status")) == Some("passed"),
        &format!("{name} did not pass"),
    )?;
    let cases = result
        .get("cases")
        .and_then(Json::as_array)
        .filter(|cases| !cases.is_empty())
        .ok_or_else(|| format!("{name} lacks executed cases"))?;
    contract::require(
        cases
            .iter()
            .all(|case| case.as_str().is_some_and(|case| !case.trim().is_empty())),
        &format!("{name} has an invalid case"),
    )?;
    let unique: std::collections::BTreeSet<&str> = cases.iter().filter_map(Json::as_str).collect();
    contract::require(
        unique.len() == cases.len(),
        &format!("{name} repeats an executed case"),
    )?;
    let required: std::collections::BTreeSet<&str> = contract::suite_cases(name)
        .unwrap_or(&[])
        .iter()
        .copied()
        .collect();
    contract::require(
        required.is_subset(&unique),
        &format!("{name} is missing required cases"),
    )?;
    let evidence_name = text(field(result, "evidence_file"))
        .filter(|evidence| *evidence == format!("{name}.json"))
        .ok_or_else(|| format!("{name} has an invalid evidence path"))?;
    let evidence_path = evidence_dir.join(evidence_name);
    contract::require(
        evidence_path.is_file() && !evidence_path.is_symlink(),
        &format!("{name} evidence is missing"),
    )?;
    contract::require(
        text(field(result, "evidence_sha256")) == Some(contract::digest(&evidence_path)?.as_str()),
        &format!("{name} evidence digest differs from bytes"),
    )?;
    let evidence = contract::load_json(&evidence_path)?;
    contract::require(
        evidence.get("schema_version").and_then(Json::as_int) == Some(1),
        &format!("{name} evidence has unknown schema"),
    )?;
    contract::require(
        text(field(&evidence, "status")) == Some("passed"),
        &format!("{name} evidence did not pass"),
    )?;
    contract::require(
        text(field(&evidence, "source_sha")) == Some(source_sha)
            && text(field(&evidence, "row_id")) == Some(row_id),
        &format!("{name} evidence belongs to another source or row"),
    )?;
    contract::require(
        text(field(&evidence, "suite")) == Some(name),
        &format!("{name} evidence belongs to another suite"),
    )?;
    contract::require(
        text(field(&evidence, "product_manifest_sha256"))
            == Some(contract::digest(product_manifest)?.as_str()),
        &format!("{name} evidence belongs to another product"),
    )?;
    contract::require(
        field(&evidence, "executed_cases") == field(result, "cases"),
        &format!("{name} evidence cases differ from receipt"),
    )?;
    let models = field(result, "models").ok_or_else(|| format!("{name} lacks model identities"))?;
    contract::validate_models(root, name, models)?;
    let evidence_models =
        field(&evidence, "models").ok_or_else(|| format!("{name} lacks model identities"))?;
    contract::require(
        evidence_models.to_value() == models.to_value(),
        &format!("{name} evidence models differ from receipt"),
    )?;
    Ok(())
}

fn validate_hardware(
    _root: &Path,
    receipt: &Json,
    availability: &Json,
    row: &Json,
) -> Result<(), String> {
    let state = text(field(availability, "state")).unwrap_or("");
    let hardware = receipt.get("hardware");
    if state == "hardware-unavailable" {
        return contract::require(
            hardware.is_none() || hardware == Some(&Json::Null),
            "unavailable row cannot claim hardware use",
        );
    }
    let hardware = hardware
        .filter(|hardware| hardware.as_object().is_some())
        .ok_or("qualified row lacks hardware evidence")?;
    let backend = contract::row_field(row, "backend")?;
    contract::require(
        text(field(hardware, "actual_backend")) == Some(backend),
        "actual backend differs from selected backend",
    )?;
    contract::require(
        text(field(hardware, "runner")) == text(field(availability, "runner")),
        "execution used a different runner from the protected availability plan",
    )?;
    let device = text(field(hardware, "device"))
        .filter(|device| !device.trim().is_empty())
        .ok_or("missing actual device")?;
    contract::require(
        text(field(hardware, "selected_device")) == contract::expected_device(backend),
        "qualified row did not select the requested backend device",
    )?;
    if backend == "cpu" {
        contract::require(device == "CPU", "CPU row has an unexpected device")?;
        contract::require(
            field(hardware, "offloaded_layers")
                .and_then(Json::as_int)
                .unwrap_or(0)
                == 0,
            "CPU row reports GPU offload",
        )?;
    } else {
        for key in ["driver", "runtime_version", "device_architecture"] {
            contract::require(
                text(field(hardware, key)).is_some_and(|value| !value.trim().is_empty()),
                &format!("GPU row lacks {key} identity"),
            )?;
        }
        contract::require(
            !device.eq_ignore_ascii_case("cpu"),
            "GPU row reports a CPU device",
        )?;
        let lowered = device.to_lowercase();
        contract::require(
            !["lavapipe", "llvmpipe", "software"]
                .iter()
                .any(|word| lowered.contains(word)),
            "software renderer cannot qualify GPU row",
        )?;
    }
    Ok(())
}

pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--receipt",
            "--product-manifest",
            "--product-dir",
            "--availability",
            "--evidence-dir",
            "--source-sha",
            "--plan-digest",
            "--row-id",
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
    let result = (|| {
        let receipt = contract::load_json(Path::new(value("--receipt")))?;
        let product = contract::load_json(Path::new(value("--product-manifest")))?;
        let availability = contract::load_json(Path::new(value("--availability")))?;
        validate_receipt(
            root,
            &receipt,
            &product,
            Path::new(value("--product-manifest")),
            Path::new(value("--product-dir")),
            &availability,
            Path::new(value("--availability")),
            Path::new(value("--evidence-dir")),
            value("--source-sha"),
            value("--plan-digest"),
            value("--row-id"),
        )?;
        contract::digest(Path::new(value("--receipt")))
    })();
    match result {
        Ok(digest) => CheckReport::success(format!("{digest}\n")),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("Skippy qualification rejected: {error}\n"),
        ),
    }
}
