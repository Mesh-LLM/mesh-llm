use super::admit;
use serde_json::{Value, json};
use std::{fs, path::PathBuf};

fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf, Value) {
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let manifest = root.join("ci/llama-canary/family-certified.json");
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    fs::copy(
        source_root.join("ci/llama-canary/family-certified.json"),
        &manifest,
    )
    .unwrap();
    let plan = temporary.path().join("plan.json");
    let bytes = crate::ci_plan::family::controller_plan(&root, &manifest).unwrap();
    fs::write(&plan, &bytes).unwrap();
    let value = serde_json::from_slice(&bytes).unwrap();
    (temporary, root, manifest, plan, value)
}

#[test]
fn controller_plan_and_selected_shard_are_admitted() {
    let (_temporary, root, manifest, plan, _) = fixture();
    admit(&root, &manifest, &plan, "").unwrap();
    admit(&root, &manifest, &plan, "000").unwrap();
    assert!(admit(&root, &manifest, &plan, "18446744073709551615").is_err());
    assert!(admit(&root, &manifest, &plan, "-1").is_err());
}

#[test]
fn canonical_verifier_rejects_each_battery_contract_mutation() {
    let (_temporary, root, manifest, plan, original) = fixture();
    let mut mutations = Vec::new();
    for (field, replacement) in [
        ("schema_version", json!(2)),
        ("manifest_sha256", json!("0".repeat(64))),
        ("required_certification_lanes", json!(["single-step"])),
        ("selected_models", json!([])),
    ] {
        let mut changed = original.clone();
        changed[field] = replacement;
        mutations.push(changed);
    }
    for class in original["model_class_lanes"].as_object().unwrap().keys() {
        let mut changed = original.clone();
        changed["model_class_lanes"][class] = json!(["single-step"]);
        mutations.push(changed);
    }
    let mut changed = original.clone();
    changed["shards"][0]["families"] = json!(["undeclared-family"]);
    mutations.push(changed);
    for changed in mutations {
        fs::write(&plan, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(admit(&root, &manifest, &plan, "0").is_err());
    }
}

#[test]
fn raw_manifest_byte_change_rejects_previously_admitted_plan() {
    let (_temporary, root, manifest, plan, _) = fixture();

    let mut bytes = fs::read(&manifest).unwrap();
    bytes.push(b'\n');
    fs::write(&manifest, bytes).unwrap();
    assert!(admit(&root, &manifest, &plan, "").is_err());
}
