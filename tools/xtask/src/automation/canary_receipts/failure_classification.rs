//! Failed evidence distinguishes candidate repair from a same-candidate infrastructure retry.
//! Classification never admits a receipt or replaces certification gates.
use super::storage::{RECEIPT_LIMIT, RESULTS_LIMIT, read_bounded};
use super::{Digest, Family, WorkerOutcome, WorkerReceipt};
use serde_json::Value;
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureClass {
    Candidate,
    Infrastructure,
    Contract,
}

pub(super) fn classify(receipt: &WorkerReceipt, directory: &Path) -> FailureClass {
    // The caller has already checked package, family, identity and attempt bounds.
    if matches!(
        receipt.outcome,
        WorkerOutcome::Cancelled | WorkerOutcome::Skipped
    ) {
        return FailureClass::Infrastructure;
    }
    let Ok(results) = read_bounded(&directory.join("results.jsonl"), RESULTS_LIMIT) else {
        return FailureClass::Contract;
    };
    if receipt.results_sha256.as_ref() != Some(&Digest::of_bytes(&results)) {
        return FailureClass::Contract;
    }
    let memory = directory.join("memory-admission.json");
    match fs::symlink_metadata(&memory) {
        Ok(_) => match read_bounded(&memory, RECEIPT_LIMIT)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        {
            Some(value) if value.is_object() && value["status"] == "passed" => (),
            _ => return FailureClass::Infrastructure,
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(_) => return FailureClass::Infrastructure,
    }
    classify_results(&results, &receipt.provenance.family)
}

fn classify_results(bytes: &[u8], family: &Family) -> FailureClass {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return FailureClass::Contract;
    };
    let mut remaining = text;
    let mut has_family = false;
    let mut infrastructure = false;
    loop {
        remaining = remaining.trim_start_matches([' ', '\t', '\n', '\r']);
        if remaining.is_empty() {
            break;
        }
        if !remaining.starts_with('{') {
            return FailureClass::Contract;
        }
        let mut stream = serde_json::Deserializer::from_str(remaining).into_iter::<Value>();
        let Some(Ok(row)) = stream.next() else {
            return FailureClass::Contract;
        };
        if row["family"] == family.as_str() {
            has_family = true;
        }
        if row["family"] == "battery" {
            let Some(outcomes) = row.get("outcomes").and_then(Value::as_array) else {
                return FailureClass::Contract;
            };
            for outcome in outcomes {
                if outcome["name"] == "environment-preflight"
                    && (outcome["status"] != "pass" || !zero_exit(&outcome["exit_code"]))
                {
                    infrastructure = true;
                }
            }
        }
        remaining = &remaining[stream.byte_offset()..];
    }
    if infrastructure || !has_family {
        FailureClass::Infrastructure
    } else {
        FailureClass::Candidate
    }
}

fn zero_exit(value: &Value) -> bool {
    value.as_i64() == Some(0)
}

#[cfg(test)]
#[path = "failure_classification_tests.rs"]
mod tests;
