use super::fixture::{invoke, repository};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};
fn resolve(manifest: &std::path::Path, cadence: &str) -> crate::process::RawProcessReport {
    invoke(
        env!("CARGO_BIN_EXE_xtask").into(),
        repository(),
        vec![
            "models".into(),
            "resolve".into(),
            manifest.to_str().unwrap().into(),
            "--artifact-id".into(),
            "family-qwen3-dense".into(),
            "--cadence".into(),
            cadence.into(),
            "--require-single-file".into(),
        ],
        BTreeMap::new(),
    )
}
#[test]
fn actual_current_gate_artifact_resolves_at_every_cadence_and_missing_authorization_rejects() {
    let source = repository().join("ci/model-artifacts/manifests/skippy-ci-smoke.json");
    let original = fs::read(&source).unwrap();
    let document: Value = serde_json::from_slice(&original).unwrap();
    for cadence in ["pull-request", "main", "manual"] {
        let report = resolve(&source, cadence);
        assert!(report.process.success(), "{:?}", report.process);
        let selected: Value = serde_json::from_slice(report.stdout.unwrap().as_bytes()).unwrap();
        assert_eq!(selected["artifact_id"], "family-qwen3-dense");
        let mut denied = document.clone();
        let artifact = denied["artifacts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|artifact| artifact["id"] == "family-qwen3-dense")
            .unwrap();
        artifact["cadences"]
            .as_array_mut()
            .unwrap()
            .retain(|value| value != cadence);
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("denied.json");
        fs::write(&path, serde_json::to_vec(&denied).unwrap()).unwrap();
        let rejected = resolve(&path, cadence);
        assert_eq!(rejected.process.status.unwrap().code(), Some(2));
        assert!(rejected.stdout.unwrap().as_bytes().is_empty());
        assert!(
            String::from_utf8_lossy(rejected.stderr.unwrap().as_bytes())
                .contains("is not allowed at cadence")
        );
    }
    assert_eq!(fs::read(source).unwrap(), original);
}
fn logical_files(artifact: &Value, registry: bool) -> Vec<Value> {
    let mut files = artifact["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| {
            if registry {
                serde_json::json!([file["path"], file["size_bytes"], file["sha256"]])
            } else {
                let path = file.as_str().unwrap();
                let integrity = &artifact["file_integrity"][path];
                serde_json::json!([path, integrity["size_bytes"], integrity["blob_id"]])
            }
        })
        .collect::<Vec<_>>();
    files.sort_by_key(Value::to_string);
    files
}
fn roster(document: &Value, registry: bool) -> Vec<String> {
    let mut rows = Vec::new();
    let list = if registry { "artifacts" } else { "models" };
    for row in document[list].as_array().unwrap() {
        if registry
            && !row["suites"]
                .as_array()
                .unwrap()
                .iter()
                .any(|suite| suite == "llama-family-certification")
        {
            continue;
        }
        if !registry {
            assert!(row.get("cadences").is_none());
        }
        rows.push(
            serde_json::json!([
                row["family"],
                row["artifact"]["repo"],
                row["artifact"]["revision"],
                logical_files(&row["artifact"], registry),
                row["artifact"]["selector"]
            ])
            .to_string(),
        );
    }
    rows.sort();
    rows
}
#[test]
fn family_certification_projection_is_complete_and_independent_of_ci_cadence_authorization() {
    let registry: Value = serde_json::from_slice(
        &fs::read(repository().join("ci/model-artifacts/registry.json")).unwrap(),
    )
    .unwrap();
    let family: Value = serde_json::from_slice(
        &fs::read(repository().join("ci/llama-canary/family-certified.json")).unwrap(),
    )
    .unwrap();
    let expected = roster(&registry, true);
    let actual = roster(&family, false);
    assert!(!actual.is_empty());
    assert_eq!(actual.iter().collect::<BTreeSet<_>>().len(), actual.len());
    assert_eq!(actual, expected);
    let mut missing = family.clone();
    missing["models"].as_array_mut().unwrap().pop();
    assert_ne!(roster(&missing, false), expected);
    let mut wrong_integrity = family.clone();
    let artifact = &mut wrong_integrity["models"][0]["artifact"];
    let path = artifact["files"][0].as_str().unwrap().to_owned();
    artifact["file_integrity"][&path]["blob_id"] = Value::String("0".repeat(64));
    assert_ne!(roster(&wrong_integrity, false), expected);
    let mut wrong_size = family.clone();
    wrong_size["models"][0]["artifact"]["file_integrity"][&path]["size_bytes"] =
        serde_json::json!(0);
    assert_ne!(roster(&wrong_size, false), expected);
}
