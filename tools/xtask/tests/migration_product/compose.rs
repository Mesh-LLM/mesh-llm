use crate::support::{CAPTURE_ENV, LEGACY_ENV, Runner, TestResult, execute, fixture, read_json};
use serde_json::{Map, Value};
use std::path::Path;

/// The case file's shared input tree, merged under each case's own files
/// unless the case sets `"base": false`.
fn with_base(case: &Value, base: &Value) -> Value {
    let mut case = case.clone();
    if case.get("base") == Some(&Value::Bool(false)) {
        return case;
    }
    let mut files = base.as_object().cloned().unwrap_or_default();
    if let Some(own) = case.get("files").and_then(Value::as_object) {
        for (path, body) in own {
            if body.is_null() {
                files.remove(path);
            } else {
                files.insert(path.clone(), body.clone());
            }
        }
    }
    if let Some(object) = case.as_object_mut() {
        object.insert("files".into(), Value::Object(files));
    }
    case
}

/// Runs every case of one group against its goldens and, when requested,
/// against the legacy script.
fn run_group(group: &str) -> TestResult {
    let cases = read_json(&fixture("compose_cases.json"))?;
    let base = cases.get("base_files").cloned().unwrap_or(Value::Null);
    let group_cases = cases
        .get("groups")
        .and_then(|groups| groups.get(group))
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing case group {group}"))?;
    assert!(!group_cases.is_empty(), "empty case group {group}");
    let golden_path = fixture(&format!("compose_goldens_{group}.json"));
    let legacy = std::env::var_os(LEGACY_ENV);
    let capture = legacy.is_some() && std::env::var_os(CAPTURE_ENV).is_some();
    let goldens = if capture {
        Value::Object(Map::new())
    } else {
        read_json(&golden_path)?
    };
    let mut captured = Map::new();
    for (name, case) in group_cases {
        let case = with_base(case, &base);
        let ported = execute(Runner::Port, &case)?;
        if let Some(python) = &legacy {
            let observed = execute(Runner::Legacy(Path::new(python)), &case)?;
            assert_eq!(observed, ported, "legacy parity for {group}/{name}");
            captured.insert(name.clone(), observed);
        }
        if !capture {
            let golden = goldens
                .get(name)
                .ok_or_else(|| format!("missing golden {group}/{name}"))?;
            assert_eq!(&ported, golden, "golden for {group}/{name}");
        }
    }
    if capture {
        let text = serde_json::to_string_pretty(&Value::Object(captured))?;
        std::fs::write(golden_path, format!("{text}\n"))?;
    }
    Ok(())
}

#[test]
fn migration_product_compose_writes_manifests() -> TestResult {
    run_group("compose")
}

#[test]
fn migration_product_compose_check_verifies_existing_manifests() -> TestResult {
    run_group("check")
}

#[test]
fn migration_product_compose_rejects_mismatched_or_missing_inputs() -> TestResult {
    run_group("reject")
}

#[test]
fn migration_product_compose_reports_malformed_runtime_manifests() -> TestResult {
    run_group("malformed")
}

#[test]
fn migration_product_compose_argv_matches_argparse() -> TestResult {
    run_group("argv")
}
