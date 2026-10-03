use super::{
    contract::{Input, Mode, Previous},
    package,
};
use crate::automation::canary_receipts::Digest;
use serde_json::json;
use std::fs;

fn input(directory: &std::path::Path) -> Input {
    let controller = directory.join("controller");
    let selected = directory.join("selected");
    fs::create_dir_all(&controller).unwrap();
    fs::create_dir_all(&selected).unwrap();
    Input {
        controller_root: controller,
        source_root: selected,
        controller_revision: "a".repeat(40),
        selected_revision: "a".repeat(40),
        mesh_source: String::new(),
        upstream_revision: "b".repeat(40),
        mode: Mode::Repair,
        pass_id: "repair-1".into(),
        run_id: "100".into(),
        run_attempt: "2".into(),
        previous: None,
        evidence: directory.join("evidence"),
        export: directory.join("export"),
        agent_timeout_seconds: 41400,
        verification_timeout_seconds: 43200,
    }
}

#[test]
fn selected_source_only_admits_unchanged_pinned_build() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = input(directory.path());
    request.mesh_source = "c".repeat(40);
    request.selected_revision = request.mesh_source.clone();
    assert!(request.validate().is_err());
    request.mode = Mode::Pinned;
    assert!(request.validate().is_ok());
    request.selected_revision = "d".repeat(40);
    assert!(request.validate().is_err());
}

#[test]
fn independent_verification_and_budget_admission_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = input(directory.path());
    request.mode = Mode::Verify;
    assert!(request.validate().is_err());
    request.mode = Mode::Repair;
    for (coding, verification) in [(0, 43200), (41401, 43200), (41400, 43201), (41400, 0)] {
        request.agent_timeout_seconds = coding;
        request.verification_timeout_seconds = verification;
        assert!(request.validate().is_err());
    }
}

#[test]
fn overlapping_existing_and_checkout_outputs_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = input(directory.path());
    request.evidence = request.export.clone();
    assert!(request.validate().is_err());
    request.evidence = request.controller_root.join("evidence");
    assert!(request.validate().is_err());
    request.evidence = directory.path().join("evidence");
    fs::create_dir(&request.evidence).unwrap();
    assert!(request.validate().is_err());
}

fn previous(request: &mut Input, directory: &std::path::Path) {
    fs::create_dir(directory).unwrap();
    let plan = json!({"selected_models":[{"family":"fixture","class":"causal_generation", "certification_lanes":[]}],
        "github_matrix":{"include":[{"shard_index":0,"families":"fixture"}]},
        "shards":[{"shard_index":0,"families":["fixture"]}],
        "required_certification_lanes":["single-step","chain","state-handoff"]});
    fs::write(
        directory.join("plan.json"),
        serde_json::to_vec(&plan).unwrap(),
    )
    .unwrap();
    let mut identity = json!({"schema":3,"platform":"macos-arm64-metal",
        "candidate":"c".repeat(40),"base":request.selected_revision,
        "controller":request.controller_revision,"mesh_source":"",
        "branch":"llama-canary/repair-fixture","pass_id":"repair-1",
        "run_id":request.run_id,"run_attempt":"1"});
    identity["plan_sha256"] = json!(Digest::of_file(&directory.join("plan.json")).unwrap());
    for (name, field) in [
        ("binaries.tar", "binaries_sha256"),
        ("workload-oracles.tar", "workload_oracles_sha256"),
        ("llama-source.bundle", "llama_bundle_sha256"),
        ("llama-source.json", "llama_provenance_sha256"),
        ("candidate.bundle", "bundle_sha256"),
    ] {
        fs::write(directory.join(name), b"bound fixture bytes").unwrap();
        identity[field] = json!(Digest::of_file(&directory.join(name)).unwrap());
    }
    fs::write(
        directory.join("identity.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();
    request.previous = Some(Previous {
        package: directory.into(),
        identity: Digest::of_file(&directory.join("identity.json"))
            .unwrap()
            .as_str()
            .into(),
        candidate: "c".repeat(40),
    });
    request.mode = Mode::Verify;
    request.pass_id = "verify-1".into();
}

#[test]
fn previous_package_requires_exact_dependency_head_base_controller_and_run_identity() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = input(directory.path());
    previous(&mut request, &directory.path().join("package"));
    assert!(request.validate().is_ok());
    assert!(package::previous(&request).unwrap().is_some());
    request.previous.as_mut().unwrap().candidate = "d".repeat(40);
    assert!(package::previous(&request).is_err());
    request.previous.as_mut().unwrap().candidate = "c".repeat(40);
    request.controller_revision = "e".repeat(40);
    assert!(package::previous(&request).is_err());
    request.controller_revision = "a".repeat(40);
    request.selected_revision = "f".repeat(40);
    assert!(package::previous(&request).is_err());
    request.selected_revision = "a".repeat(40);
    request.run_id = "200".into();
    assert!(package::previous(&request).is_err());
}

#[test]
fn previous_artifact_corruption_and_newer_producer_attempt_cannot_start_verifier() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = input(directory.path());
    let package_path = directory.path().join("package");
    previous(&mut request, &package_path);
    request.run_attempt = "1".into();
    assert!(package::previous(&request).is_ok());
    fs::write(package_path.join("binaries.tar"), b"corrupt").unwrap();
    assert!(package::previous(&request).is_err());
    fs::write(package_path.join("binaries.tar"), b"bound fixture bytes").unwrap();
    let mut identity: serde_json::Value =
        serde_json::from_slice(&fs::read(package_path.join("identity.json")).unwrap()).unwrap();
    identity["run_attempt"] = "2".into();
    fs::write(
        package_path.join("identity.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();
    request.previous.as_mut().unwrap().identity =
        Digest::of_file(&package_path.join("identity.json"))
            .unwrap()
            .as_str()
            .into();
    assert!(package::previous(&request).is_err());
}

#[test]
fn exported_package_cannot_change_the_verifier_candidate_or_pinned_source() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = input(directory.path());
    let package_path = directory.path().join("package");
    previous(&mut request, &package_path);
    let admitted = package::previous(&request).unwrap().unwrap().1;
    fs::rename(&package_path, &request.export).unwrap();
    let identity_path = request.export.join("identity.json");
    let mut identity: serde_json::Value =
        serde_json::from_slice(&fs::read(&identity_path).unwrap()).unwrap();
    identity["pass_id"] = "verify-1".into();
    fs::write(&identity_path, serde_json::to_vec(&identity).unwrap()).unwrap();
    assert!(package::exported(&request, Some(&admitted)).is_ok());
    identity["candidate"] = json!("d".repeat(40));
    fs::write(&identity_path, serde_json::to_vec(&identity).unwrap()).unwrap();
    assert!(package::exported(&request, Some(&admitted)).is_err());
    request.mode = Mode::Pinned;
    assert!(package::exported(&request, None).is_err());
    identity["candidate"] = json!(request.selected_revision);
    fs::write(&identity_path, serde_json::to_vec(&identity).unwrap()).unwrap();
    assert!(package::exported(&request, None).is_ok());
    identity["base"] = json!("e".repeat(40));
    fs::write(&identity_path, serde_json::to_vec(&identity).unwrap()).unwrap();
    assert!(package::exported(&request, None).is_err());
}

#[test]
fn admitted_producer_branch_is_preserved_when_current_verifier_attempt_advances() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = input(directory.path());
    let package_path = directory.path().join("package");
    previous(&mut request, &package_path);
    request.run_attempt = "3".into();
    let (_, identity) = package::previous(&request).unwrap().unwrap();
    let environment = super::environment::wrapper(&request, Some((&package_path, &identity)));
    for (key, expected) in [
        ("CANARY_CANDIDATE_BRANCH", "llama-canary/repair-fixture"),
        ("CANARY_CANDIDATE_SHA", identity.candidate.as_str()),
        ("GITHUB_RUN_ATTEMPT", "3"),
        ("CANARY_HARNESS_MODE", "verify-build"),
    ] {
        let value = environment.get(std::ffi::OsStr::new(key)).unwrap();
        assert!(
            matches!(value, crate::process::Value::Public(actual) if actual == std::ffi::OsStr::new(expected)),
            "{key}"
        );
    }
    let bundle = package_path.join("candidate.bundle");
    assert!(
        matches!(environment.get(std::ffi::OsStr::new("CANARY_INPUT_BUNDLE")), Some(crate::process::Value::Public(actual)) if actual == bundle.as_os_str())
    );
    let producer: serde_json::Value =
        serde_json::from_slice(&fs::read(package_path.join("identity.json")).unwrap()).unwrap();
    assert_eq!(producer["run_attempt"], "1");
    assert_eq!(producer["branch"], "llama-canary/repair-fixture");
}
