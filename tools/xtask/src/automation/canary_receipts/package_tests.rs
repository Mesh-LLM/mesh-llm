use super::*;
use serde_json::json;
use std::fs;

const IDENTITY: &[u8] =
    include_bytes!("../../../tests/migration_canary_receipts/fixtures/identity.json");
const PLAN: &[u8] = include_bytes!("../../../tests/migration_canary_receipts/fixtures/plan.json");

#[test]
fn captured_identity_mismatch_refuses_even_when_current_file_matches_expected() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("identity.json");
    fs::write(&path, IDENTITY).unwrap();
    let expected = Digest::of_bytes(IDENTITY);
    let mut captured: Value = serde_json::from_slice(IDENTITY).unwrap();
    captured["candidate"] = json!("c".repeat(40));
    let captured = serde_json::to_vec(&captured).unwrap();
    // The old pre-read file check passes; the bytes actually consumed are foreign.
    assert_eq!(Digest::of_file(&path).unwrap(), expected);
    let error = identity_from_captured(&captured, &expected).unwrap_err();
    assert_eq!(error.kind, ErrorKind::PackageIdentity);
    assert!(error.message.contains("build identity digest mismatch"));
    assert_eq!(fs::read(path).unwrap(), IDENTITY);
}

#[test]
fn captured_plan_mismatch_refuses_even_when_current_artifact_hash_is_correct() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("plan.json"), PLAN).unwrap();
    let identity = json!({"plan_sha256":Digest::of_bytes(PLAN)});
    let mut captured: Value = serde_json::from_slice(PLAN).unwrap();
    captured["historical_source_field"] = json!("different captured source plan");
    let captured = serde_json::to_vec(&captured).unwrap();
    // This remains a valid source plan, and the later pathname verification passes.
    assert!(SourceFamilyPlan::parse(&captured).is_ok());
    verify_artifact(root.path(), &identity, "plan.json", "plan_sha256").unwrap();
    let error = match plan_from_captured(&captured, &identity) {
        Err(error) => error,
        Ok(_) => panic!("captured plan bytes escaped producer digest binding"),
    };
    assert_eq!(error.kind, ErrorKind::PackageArtifact);
    assert!(error.message.contains("plan.json"));
    assert_eq!(fs::read(root.path().join("plan.json")).unwrap(), PLAN);
}

#[test]
fn captured_custody_is_checked_before_json_parse_or_plan_use() {
    let malformed = b"not JSON";
    let error = identity_from_captured(malformed, &Digest::of_bytes(IDENTITY)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::PackageIdentity);
    let identity = json!({"plan_sha256":Digest::of_bytes(PLAN)});
    let error = match plan_from_captured(malformed, &identity) {
        Err(error) => error,
        Ok(_) => panic!("unbound captured plan was accepted"),
    };
    assert_eq!(error.kind, ErrorKind::PackageArtifact);
    // Digest admission alone does not grant malformed JSON a valid identity.
    assert_eq!(
        identity_from_captured(malformed, &Digest::of_bytes(malformed))
            .unwrap_err()
            .kind,
        ErrorKind::Json
    );
}

#[test]
fn valid_captured_identity_and_plan_remain_admitted_after_source_path_changes() {
    let root = tempfile::tempdir().unwrap();
    let identity = json!({"plan_sha256":Digest::of_bytes(PLAN)});
    let identity_bytes = serde_json::to_vec(&identity).unwrap();
    let expected = Digest::of_bytes(&identity_bytes);
    fs::write(root.path().join("identity.json"), b"later unrelated bytes").unwrap();
    fs::write(root.path().join("plan.json"), b"later unrelated plan").unwrap();
    let admitted = identity_from_captured(&identity_bytes, &expected).unwrap();
    assert_eq!(admitted, identity);
    let plan = plan_from_captured(PLAN, &admitted).unwrap();
    assert_eq!(plan.models.len(), 2);
}

#[test]
fn captured_plan_requires_a_valid_declared_digest() {
    for identity in [
        json!({}),
        json!({"plan_sha256":null}),
        json!({"plan_sha256":"not-a-digest"}),
        json!({"plan_sha256":"A".repeat(64)}),
    ] {
        assert!(plan_from_captured(PLAN, &identity).is_err(), "{identity}");
    }
}
