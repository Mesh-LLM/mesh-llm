use super::schema::*;
use super::{Error, load, validation};

struct Fixture {
    directory: tempfile::TempDir,
    receipt: Receipt,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let artifact = write(
            &directory,
            "input",
            b"fake immutable product/model/evidence bytes",
        );
        let contracts = write(&directory, "contracts.json", br#"{"qualification":{"schema_version":1,"required_roots":["build","test-all"],"required_backends":{"macos":["metal"]}}}"#);
        let execution = Execution {
            argv: vec!["just".into(), "build".into()],
            exit_code: 0,
            case_count: 1,
            cleanup_complete: true,
            evidence: artifact.clone(),
        };
        let product = Product {
            backend: Backend::Metal,
            hardware_evidence: artifact.clone(),
            host_manifest: artifact.clone(),
            runtime_manifest: artifact.clone(),
            product_manifest: artifact.clone(),
            files: vec![artifact.clone()],
        };
        let model_ids = ["smollm2-q8-inference", "family-granite-hybrid"];
        let models = model_ids
            .iter()
            .map(|id| Model {
                artifact_id: (*id).into(),
                revision: "a".repeat(40),
                files: vec![artifact.clone()],
            })
            .collect();
        let protocol_cases = model_ids
            .iter()
            .map(|id| ProtocolCase {
                backend: Backend::Metal,
                model_id: (*id).into(),
                execution: execution.clone(),
            })
            .collect();
        let scenarios = [
            Scenario::ProductReadiness,
            Scenario::ProtocolPair,
            Scenario::CorruptRuntime,
            Scenario::ReadinessTimeout,
        ]
        .into_iter()
        .map(|scenario| {
            let failure_kind = match scenario {
                Scenario::ProductReadiness | Scenario::ProtocolPair => None,
                Scenario::CorruptRuntime => Some(FailureKind::DigestMismatch),
                Scenario::ReadinessTimeout => Some(FailureKind::ReadinessTimeout),
            };
            let mut execution = execution.clone();
            if scenario == Scenario::ProtocolPair {
                execution.case_count = 2;
            }
            if failure_kind.is_some() {
                execution.exit_code = 1;
            }
            ScenarioReceipt {
                scenario,
                execution,
                expected_failure_observed: failure_kind.is_some(),
                failure_kind,
            }
        })
        .collect();
        let receipt = Receipt {
            schema_version: 1,
            platform: Platform::Macos,
            source_sha: "b".repeat(40),
            source_snapshot: artifact.clone(),
            contracts,
            interpreters: InterpreterProof {
                path: "/controlled/tools".into(),
                attempts: 0,
                probes: [
                    ProbeKind::Path,
                    ProbeKind::Absolute,
                    ProbeKind::Shebang,
                    ProbeKind::Versioned,
                ]
                .into_iter()
                .map(|kind| Probe {
                    kind,
                    candidates: vec!["python3".into(), "pip".into(), "uv".into()],
                    found: vec![],
                    evidence: artifact.clone(),
                })
                .collect(),
            },
            roots: [
                ("build".into(), execution.clone()),
                ("test-all".into(), execution),
            ]
            .into_iter()
            .collect(),
            products: vec![product],
            models,
            protocol_cases,
            scenarios,
        };
        Self { directory, receipt }
    }

    fn validate(&self) -> Result<(), Error> {
        validation::validate(&self.receipt, &"b".repeat(40))
    }
}

fn write(directory: &tempfile::TempDir, name: &str, bytes: &[u8]) -> Artifact {
    let path = directory.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    let sha256 = crate::product::digest::file_sha256(&path)
        .map_err(|failure| failure.error)
        .unwrap();
    Artifact { path, sha256 }
}

#[test]
fn migration_python_free_accepts_complete_hash_bound_fake_receipt() {
    let fixture = Fixture::new();
    let result = fixture.validate();
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn migration_python_free_rejects_missing_required_field() {
    let fixture = Fixture::new();
    let mut document = serde_json::to_value(&fixture.receipt).unwrap();
    document.as_object_mut().unwrap().remove("interpreters");
    let path = fixture.directory.path().join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
    let result = load(&path);
    assert!(matches!(result, Err(Error::Json(_))));
}

#[test]
fn migration_python_free_rejects_changed_artifact_bytes() {
    let fixture = Fixture::new();
    std::fs::write(&fixture.receipt.products[0].files[0].path, b"corrupt").unwrap();
    let result = fixture.validate();
    assert!(matches!(result, Err(Error::DigestMismatch(_))));
}

#[test]
fn migration_python_free_rejects_interpreter_found_in_each_probe() {
    for index in 0..4 {
        let mut fixture = Fixture::new();
        fixture.receipt.interpreters.probes[index]
            .found
            .push("/usr/bin/python3".into());
        let result = fixture.validate();
        assert!(matches!(
            result,
            Err(Error::Invalid(
                "interpreter found or probe has no candidates"
            ))
        ));
    }
}

#[test]
fn migration_python_free_rejects_skipped_scenario() {
    let mut fixture = Fixture::new();
    fixture.receipt.scenarios[0].execution.case_count = 0;
    let result = fixture.validate();
    assert!(matches!(result, Err(Error::Invalid(_))));
}

#[test]
fn migration_python_free_rejects_missing_model_without_skip() {
    let mut fixture = Fixture::new();
    fixture.receipt.models.pop();
    let result = fixture.validate();
    assert!(matches!(
        result,
        Err(Error::Invalid(
            "required pinned dense/recurrent models missing"
        ))
    ));
}

#[test]
fn migration_python_free_rejects_missing_hardware_without_skip() {
    let mut fixture = Fixture::new();
    fixture.receipt.products.clear();
    let result = fixture.validate();
    assert!(matches!(result, Err(Error::Invalid(_))));
}

#[test]
fn migration_python_free_rejects_wrong_negative_failure() {
    let mut fixture = Fixture::new();
    fixture.receipt.scenarios[2].failure_kind = Some(FailureKind::ReadinessTimeout);
    let result = fixture.validate();
    assert!(matches!(result, Err(Error::Invalid(_))));
}

#[test]
fn migration_python_free_rejects_source_mismatch() {
    let fixture = Fixture::new();
    let result = validation::validate(&fixture.receipt, &"c".repeat(40));
    assert!(matches!(result, Err(Error::Invalid(_))));
}

#[test]
fn migration_python_free_rejects_unexecuted_root() {
    let mut fixture = Fixture::new();
    fixture.receipt.roots.remove("test-all");
    let result = fixture.validate();
    assert!(matches!(result, Err(Error::Invalid(_))));
}

#[test]
fn migration_python_free_rejects_incomplete_protocol_rows() {
    let mut fixture = Fixture::new();
    fixture.receipt.protocol_cases.pop();
    let result = fixture.validate();
    assert!(matches!(result, Err(Error::Invalid(_))));
}

#[test]
fn migration_python_free_rejects_leaked_children() {
    let mut fixture = Fixture::new();
    fixture.receipt.scenarios[3].execution.cleanup_complete = false;
    let result = fixture.validate();
    assert!(matches!(result, Err(Error::Invalid(_))));
}
