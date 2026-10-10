use super::{Fixture, Value, fs, json};
use sha2::{Digest, Sha256};

#[test]
fn production_preflight_requires_full_cache_metadata_policy_before_any_output() {
    for mode in ["not_checked", "blob_identity"] {
        let mut fixture = Fixture::new();
        fixture.input["cache"] = if mode == "not_checked" {
            json!({"mode":mode})
        } else {
            json!({"mode":mode,"root":fixture.directory.path().join("unused")})
        };
        let result = fixture.run("preflight");
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("requires gguf_metadata"));
        assert!(!fixture.output("plan.json").exists());
        assert!(!fixture.output("summary.json").exists());
    }
}

#[test]
fn production_ready_receipt_binds_admitted_plan_and_scheduling_without_certification_claim() {
    let fixture = super::metadata::fixture(
        &super::metadata::valid_target(),
        &super::metadata::valid_draft(),
    );
    let result = fixture.run("preflight");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary: Value =
        serde_json::from_slice(&fs::read(fixture.output("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["status"], "ready");
    assert_eq!(summary["certification"], "pending");
    assert_eq!(summary["selected_execution_preflight"], "pending");
    assert_eq!(
        summary["controller_revision"],
        fixture.input["controller_revision"]
    );
    assert_eq!(
        summary["selected_revision"],
        fixture.input["selected_revision"]
    );
    for (field, name) in [
        ("plan_sha256", "plan.json"),
        ("source_plan_sha256", "source-plan.json"),
        ("scheduling_matrix_sha256", "scheduling-matrix.json"),
    ] {
        assert_eq!(
            summary[field],
            hex::encode(Sha256::digest(fs::read(fixture.output(name)).unwrap()))
        );
    }
    assert_eq!(summary["families"], 1);
    assert!(fixture.output("battery-ran").exists());
}

#[test]
fn selected_contract_failure_retains_plan_but_never_ready_receipt() {
    let fixture = super::metadata::fixture(
        &super::metadata::valid_target(),
        &super::metadata::valid_draft(),
    );
    fs::write(
        fixture
            .directory
            .path()
            .join("selected/scripts/skippy-family-battery.sh"),
        "exit 37\n",
    )
    .unwrap();
    let result = fixture.run("preflight");
    assert!(!result.status.success());
    assert!(fixture.output("plan.json").exists());
    assert!(!fixture.output("source-plan.json").exists());
    assert!(!fixture.output("summary.json").exists());
}

#[test]
fn metadata_failure_never_runs_selected_contract_or_publishes_ready_receipt() {
    let fixture = super::metadata::fixture(b"invalid GGUF", &super::metadata::valid_draft());
    let result = fixture.run("preflight");
    assert!(!result.status.success());
    assert!(!fixture.output("battery-ran").exists());
    assert!(!fixture.output("plan.json").exists());
    assert!(!fixture.output("summary.json").exists());
}

#[test]
fn oversized_family_rejects_before_selected_contract_or_ready_receipt() {
    let fixture = super::metadata::fixture(
        &super::metadata::valid_target(),
        &super::metadata::valid_draft(),
    );
    let path = fixture
        .directory
        .path()
        .join("selected/ci/llama-canary/family-certified.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["models"][0]["resources"]["estimated_model_bytes"] =
        json!(180_u64 * 1024 * 1024 * 1024);
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let result = fixture.run("preflight");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("largest runner budget"));
    assert!(!fixture.output("battery-ran").exists());
    assert!(!fixture.output("plan.json").exists());
    assert!(!fixture.output("summary.json").exists());
}
