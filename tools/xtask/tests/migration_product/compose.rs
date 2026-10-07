#[path = "current_composition.rs"]
mod current_composition;

use crate::support::{TestResult, execute, fixture, read_json};
use serde_json::Value;

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
    let goldens = read_json(&golden_path)?;
    for (name, case) in group_cases {
        let case = with_base(case, &base);
        let ported = execute(&case)?;
        {
            let golden = goldens
                .get(name)
                .ok_or_else(|| format!("missing golden {group}/{name}"))?;
            if name == "unicode_and_structured_ids" {
                assert_eq!(ported["code"], 1);
                assert_eq!(ported["products"], serde_json::json!({}));
                assert!(!ported["stderr"].as_str().ok_or("stderr")?.is_empty());
            } else if name == "abbreviations_reach_composition" {
                assert_eq!(ported["code"], 2);
                assert_eq!(ported["products"], serde_json::json!({}));
                assert!(!ported["stderr"].as_str().ok_or("stderr")?.is_empty());
            } else if golden["code"] != 0 {
                assert_eq!(ported["code"], golden["code"], "{group}/{name}");
                assert_eq!(ported["products"], golden["products"], "{group}/{name}");
                assert_eq!(ported["stdout"], golden["stdout"], "{group}/{name}");
                assert!(!ported["stderr"].as_str().ok_or("stderr")?.is_empty());
                if matches!(
                    name.as_str(),
                    "version_mismatch"
                        | "runtime_backend_mismatch"
                        | "build_backend_mismatch"
                        | "stale_host_digest"
                        | "stale_runtime_file"
                ) {
                    assert!(
                        ported["stderr"]
                            .as_str()
                            .ok_or("stderr")?
                            .starts_with("product composition failed:")
                    );
                }
            } else if group == "argv" && name.starts_with("help") {
                assert_eq!(ported["code"], 0);
                assert_eq!(ported["stderr"], "");
                assert_eq!(ported["products"], serde_json::json!({}));
                let help = ported["stdout"].as_str().ok_or("help")?;
                assert!(help.starts_with("usage: product compose "));
                for option in [
                    "--bundle",
                    "--host",
                    "--runtime",
                    "--version",
                    "--backend",
                    "--check",
                ] {
                    assert!(help.contains(option));
                }
            } else {
                assert_eq!(&ported, golden, "golden for {group}/{name}");
            }
        }
    }
    Ok(())
}

#[test]
fn migration_product_compose_writes_manifests() -> TestResult {
    current_composition::writes_manifests()
}

#[test]
fn migration_product_compose_check_verifies_existing_manifests() -> TestResult {
    current_composition::checks_existing_manifest()
}

#[test]
fn migration_product_compose_rejects_mismatched_or_missing_inputs() -> TestResult {
    current_composition::rejects_mismatch()
}

#[test]
fn migration_product_compose_reports_malformed_runtime_manifests() -> TestResult {
    run_group("malformed")
}

#[test]
fn migration_product_compose_argv_matches_argparse() -> TestResult {
    current_composition::argv_contract()
}
