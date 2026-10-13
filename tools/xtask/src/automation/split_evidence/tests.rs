use super::{
    args::{Mode, NAMES, Request},
    command::execute,
};
use std::{fs, path::PathBuf};

mod adversarial;
mod domains;
mod equality;
mod executable_parity;
mod mismatches;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/migration/split_evidence")
}

fn request(root: &std::path::Path) -> Request {
    for name in NAMES {
        let filename = format!("{}.json", name.replace('_', "-"));
        fs::copy(fixture().join(&filename), root.join(filename)).unwrap();
    }
    Request {
        paths: NAMES.map(|name| root.join(format!("{}.json", name.replace('_', "-")))),
        model_label: " dense \u{a0}".into(),
        mode: Mode::Output(root.join("split-evidence.json")),
    }
}

fn mutate(request: &Request, index: usize, action: impl FnOnce(&mut serde_json::Value)) {
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&request.paths[index]).unwrap()).unwrap();
    action(&mut value);
    fs::write(&request.paths[index], serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn full_golden_when_valid_snapshots_include_uppercase_digests() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    let result = execute(&request).unwrap();
    assert_eq!(
        result,
        "ready=true topology=topology-a run=run-a model=model-a stages=2 observers=2\n"
    );
    assert_eq!(
        fs::read(root.path().join("split-evidence.json")).unwrap(),
        fs::read(fixture().join("expected-ready.json")).unwrap()
    );
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 7);
}

#[test]
fn verifies_without_mutation_when_evidence_matches() {
    let root = tempfile::tempdir().unwrap();
    let mut request = request(root.path());
    let path = root.path().join("split-evidence.json");
    fs::copy(fixture().join("expected-ready.json"), &path).unwrap();
    request.mode = Mode::Verify(path.clone());
    let result = execute(&request).unwrap();
    assert_eq!(
        result,
        format!("Verified two-node split evidence: {}\n", path.display())
    );
    assert_eq!(
        fs::read(path).unwrap(),
        fs::read(fixture().join("expected-ready.json")).unwrap()
    );
}

#[test]
fn verification_preserves_original_when_snapshot_tampered_or_missing() {
    for missing in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut request = request(root.path());
        let path = root.path().join("split-evidence.json");
        let original = fs::read(fixture().join("expected-ready.json")).unwrap();
        fs::write(&path, &original).unwrap();
        if missing {
            fs::remove_file(&request.paths[5]).unwrap();
        } else {
            fs::write(&request.paths[5], b"{\"data\":[{\"id\":\"model-b\"}]}").unwrap();
        }
        request.mode = Mode::Verify(path.clone());
        assert!(execute(&request).is_err());
        assert_eq!(fs::read(path).unwrap(), original);
    }
}

#[test]
fn failed_schema_when_snapshot_missing() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    fs::remove_file(&request.paths[0]).unwrap();
    assert!(execute(&request).is_err());
    let failure: serde_json::Value =
        serde_json::from_slice(&fs::read(root.path().join("split-evidence.json")).unwrap())
            .unwrap();
    assert_eq!(failure.as_object().unwrap().len(), 5);
    assert_eq!(failure["status"], "failed");
    assert_eq!(failure["errors"].as_array().unwrap().len(), 1);
}

#[test]
fn own_temporary_removed_when_replace_fails() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    fs::create_dir(root.path().join("split-evidence.json")).unwrap();
    assert!(execute(&request).is_err());
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 7);
}
