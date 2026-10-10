use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{fixture_catalog, fixture_materialization, selection::Trajectory};

fn checked_in() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../skippy/evals/skippy-scheduler-fixtures.json"
    )))
    .unwrap()
}

fn canned() -> (Value, Trajectory, String) {
    let mut catalog = checked_in();
    let row = Trajectory {
        session_id: "a".into(),
        source_dataset: "source".into(),
        messages_json: json!([{"role":"user","content":"é"}]).to_string(),
        n_turns: 20,
        max_isl: 9000,
        total_tokens: 9100,
    };
    let revision = "a".repeat(40);
    catalog["datasets"]["agentic-coding-trajectories"]["revision"] = revision.clone().into();
    let expected = include_str!("fixture_expected.json").replace("abc", &revision);
    let profile = &mut catalog["profiles"]["agentic-eviction-pressure"];
    profile["workload"]["families"] = 1.into();
    profile["workload"]["requests_per_family"] = 1.into();
    profile["workload"]["admission_concurrency"] = 1.into();
    profile["ci_trace"]["family_order"] = json!([0]);
    profile["corpus"]["selection"] = json!({"sources":["source"],"families":1,"min_isl":8192,"max_isl_exclusive":12000,"min_turns":20,"order":"md5(session_id)"});
    profile["corpus"]["rows"] = serde_json::to_value([&row]).unwrap();
    profile["corpus"]["prompt_manifest_sha256"] =
        hex::encode(Sha256::digest(expected.as_bytes())).into();
    (catalog, row, expected)
}

#[test]
fn checked_in_scheduler_catalog_pins_valid_context_model_and_trace() {
    fixture_catalog::validate(&checked_in()).unwrap();
}

#[test]
fn scheduler_catalog_preserves_measured_pressure_rows_and_immutable_inputs() {
    let catalog = checked_in();
    fixture_catalog::validate(&catalog).unwrap();
    let profile = &catalog["profiles"]["agentic-eviction-pressure"];
    assert_eq!(profile["workload"]["families"], 8);
    assert_eq!(profile["workload"]["ctx_size"], 131072);
    assert_eq!(profile["corpus"]["rows"].as_array().unwrap().len(), 8);
    assert_eq!(
        profile["model"]["sha256"],
        "603bd3f8a0281d16571da7c08bd661ee17ff0d1be6fcbd1b42242da257ef0bb8"
    );
    assert_eq!(
        profile["corpus"]["prompt_manifest_sha256"],
        "f1ddbe3d5974f3f4bd06f5d70fa45d0e10305bbafa4eb7399a0f972458d1beef"
    );
    assert_eq!(
        catalog["datasets"]["agentic-coding-trajectories"]["revision"],
        "cef72d1f4d0caabf85937adf8337a14b7522c782"
    );
}

#[test]
fn scheduler_catalog_requires_complete_context_and_every_model_identity_field() {
    for context in [65536, 131071] {
        let mut catalog = checked_in();
        catalog["profiles"]["agentic-eviction-pressure"]["workload"]["ctx_size"] = context.into();
        let error = fixture_catalog::validate(&catalog).unwrap_err().to_string();
        assert!(error.contains("pinned row totals"), "{error}");
        assert!(error.contains("131072"), "{error}");
    }
    for field in ["id", "repo", "filename", "revision", "sha256"] {
        let mut catalog = checked_in();
        catalog["profiles"]["warm-affinity"]["model"]
            .as_object_mut()
            .unwrap()
            .remove(field)
            .unwrap();
        assert!(
            fixture_catalog::validate(&catalog).is_err(),
            "field={field}"
        );
    }
}

#[test]
fn scheduler_catalog_rejects_context_identity_and_incomplete_trace() {
    for change in ["context", "identity", "trace", "source", "boolean"] {
        let mut catalog = checked_in();
        let profile = &mut catalog["profiles"]["agentic-eviction-pressure"];
        match change {
            "context" => profile["workload"]["ctx_size"] = 1.into(),
            "identity" => profile["model"]
                .as_object_mut()
                .unwrap()
                .remove("revision")
                .map(|_| ())
                .unwrap(),
            "trace" => profile["ci_trace"]["family_order"] = json!([0]),
            "source" => profile["corpus"]["selection"]["sources"] = json!([]),
            "boolean" => profile["workload"]["families"] = true.into(),
            _ => unreachable!(),
        }
        assert!(fixture_catalog::validate(&catalog).is_err(), "{change}");
    }
}

#[test]
fn scheduler_materialization_verifies_pinned_rows_and_exact_manifest_before_publish() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("manifest.json");
    let (catalog, row, expected) = canned();
    let digest = fixture_materialization::publish_selected(
        &catalog,
        "agentic-eviction-pressure",
        std::slice::from_ref(&row),
        &output,
    )
    .unwrap();
    assert_eq!(
        digest,
        catalog["profiles"]["agentic-eviction-pressure"]["corpus"]["prompt_manifest_sha256"]
            .as_str()
            .unwrap()
    );
    assert_eq!(std::fs::read(&output).unwrap(), expected.as_bytes());
}

#[test]
fn scheduler_row_or_hash_drift_preserves_existing_output_without_temporary_leaks() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("manifest.json");
    let (catalog, row, _) = canned();
    std::fs::write(&output, b"keep-me").unwrap();
    for field in ["row", "hash"] {
        let mut changed = catalog.clone();
        let corpus = &mut changed["profiles"]["agentic-eviction-pressure"]["corpus"];
        if field == "row" {
            corpus["rows"][0]["session_id"] = "unexpected".into();
        } else {
            corpus["prompt_manifest_sha256"] = "0".repeat(64).into();
        }
        let error = fixture_materialization::publish_selected(
            &changed,
            "agentic-eviction-pressure",
            std::slice::from_ref(&row),
            &output,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(if field == "row" {
            "pinned fixture provenance"
        } else {
            "SHA-256 mismatch"
        }));
        assert_eq!(std::fs::read(&output).unwrap(), b"keep-me");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
