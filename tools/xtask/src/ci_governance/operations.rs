//! Read-only admission of operator-normalized evidence, never execution authority.
mod environment_readback;
mod intent;
mod preparation;
mod record;
mod record_source;
mod state;

use super::policy::Policy;
use crate::command::DynResult;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(crate) fn prepare_administration(root: &Path, args: &[String]) -> DynResult<Value> {
    preparation::run(root, args)
}

pub(crate) fn inspect_administration_approval(root: &Path, args: &[String]) -> DynResult<Value> {
    let mut result = preparation::run(root, args)?;
    result["status"] = json!("AUTHORIZATION_BLOCKED");
    Ok(result)
}

#[cfg(test)]
mod preparation_fixtures;

pub(crate) fn validate_change(
    policy: &Policy,
    expected_path: &Path,
    record_path: &Path,
    observed_path: &Path,
) -> DynResult<Value> {
    let expected: state::ExpectedState = read(expected_path)?;
    let observed: state::Observation = read(observed_path)?;
    let record: record::ChangeRecord = read(record_path)?;
    validate_record(policy, &expected, &record, &observed, now()?)
}

pub(crate) enum DriftOutcome {
    Match(Value),
    Mismatch {
        report: Value,
        differences: Vec<&'static str>,
    },
}

pub(crate) fn check_drift(
    policy: &Policy,
    expected_path: &Path,
    observed_path: &Path,
) -> DynResult<DriftOutcome> {
    let expected: state::ExpectedState = read(expected_path)?;
    let observed: state::Observation = read(observed_path)?;
    drift_outcome(policy, &expected, &observed, now()?)
}

fn validate_record(
    policy: &Policy,
    expected: &state::ExpectedState,
    record: &record::ChangeRecord,
    observed: &state::Observation,
    now: u64,
) -> DynResult<Value> {
    expected.validate(policy)?;
    observed.validate(policy, now, expected.drift_target_seconds)?;
    record.validate(policy, expected, observed, now)?;
    Ok(json!({
        "schema_version": 1,
        "status": if record.manual_readback.is_some() { "RECORDED_MANUAL_READBACK" } else { "VALIDATED_PLAN" },
        "operation_id": record.request.operation_id,
        "request_sha256": record.request.canonical_digest()?,
        "identity_binding_status": "logical-only-unverified",
        "execution_authorized": false,
        "evidence_authenticated": false,
        "limitation": "Local evidence consistency only; a protected human-operated collector must authenticate policy, approval, run, replay ledger and live readbacks."
    }))
}

fn drift_outcome(
    policy: &Policy,
    expected: &state::ExpectedState,
    observed: &state::Observation,
    now: u64,
) -> DynResult<DriftOutcome> {
    expected.validate(policy)?;
    observed.validate(policy, now, expected.drift_target_seconds)?;
    let differences = expected.state.differences(&observed.state)?;
    let report = json!({
        "schema_version": 1,
        "status": if differences.is_empty() { "MATCH" } else { "DRIFT" },
        "report_scope": "local-snapshot-comparison",
        "differences": &differences,
        "observed_at_unix": observed.observed_at_unix,
        "proposed_detection_target_seconds": expected.drift_target_seconds,
        "execution_authorized": false,
        "evidence_authenticated": false,
        "identity_binding_status": "logical-only-unverified",
        "live_detection_latency_measured": false
    });
    if differences.is_empty() {
        Ok(DriftOutcome::Match(report))
    } else {
        Ok(DriftOutcome::Mismatch {
            report,
            differences,
        })
    }
}

#[cfg(test)]
fn drift(
    policy: &Policy,
    expected: &state::ExpectedState,
    observed: &state::Observation,
    now: u64,
) -> DynResult<Value> {
    match drift_outcome(policy, expected, observed, now)? {
        DriftOutcome::Match(report) => Ok(report),
        DriftOutcome::Mismatch { differences, .. } => {
            Err(format!("CI governance drift: {}", differences.join(", ")).into())
        }
    }
}

/// Canonical digest is SHA-256 of compact JSON with recursively sorted object
/// keys and preserved array order. It is not a pretty-printed input file hash.
fn digest(value: &impl Serialize) -> DynResult<String> {
    let mut canonical = serde_json::to_value(value)?;
    canonical.sort_all_objects();
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&canonical)?)))
}

fn read<T: DeserializeOwned>(path: &Path) -> DynResult<T> {
    const MAX_INPUT_BYTES: u64 = 1024 * 1024;
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err("governance input exceeds 1 MiB".into());
    }
    serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "invalid governance JSON schema at line {}, column {}: unknown, missing or malformed fields",
            error.line(), error.column()
        ).into()
    })
}

fn now() -> DynResult<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs())
}

fn require(condition: bool, reason: &str) -> DynResult<()> {
    if !condition {
        return Err(reason.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
