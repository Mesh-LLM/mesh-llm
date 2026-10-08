//! Actual smoke consumer body, real extraction and product verification owners.
use super::{Fixture, Value, invoke_arguments, snapshot, tar_member};
use crate::workflow_yaml::{self, Node};
use flate2::{Compression, write::GzEncoder};
use std::{fs, io::Write, path::Path};

fn repository() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn document() -> Node {
    workflow_yaml::parse(
        &fs::read_to_string(repository().join(".github/actions/restore-smoke-inputs/action.yml"))
            .unwrap(),
    )
    .unwrap()
}

fn steps(document: &Node) -> &[Node] {
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("composite steps required");
    };
    steps
}

fn extract_body(backend: &str) -> String {
    let document = document();
    let selected: Vec<_> = steps(&document)
        .iter()
        .filter(|step| {
            step.get("name").and_then(Node::text) == Some("Extract and verify composed product")
        })
        .collect();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].get("shell").and_then(Node::text), Some("bash"));
    let body = selected[0]
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .replace("${{ inputs.artifact_path }}", "restore-input")
        .replace("${{ inputs.binary_name }}", "mesh-llm")
        .replace("${{ inputs.expected_backend }}", backend);
    assert!(!body.contains("${{"));
    body
}

fn prepared() -> Fixture {
    let fixture = Fixture::new("1.2.3");
    let before = snapshot(&fixture.root.join("inputs"));
    fixture.accepted(&fixture.run_version("product", "1.2.3"), &before);
    fs::copy(
        repository().join("scripts/verify-native-runtime-package.sh"),
        fixture
            .root
            .join("scripts/verify-native-runtime-package.sh"),
    )
    .unwrap();
    fs::create_dir_all(fixture.root.join("skippy/scripts")).unwrap();
    fs::copy(
        repository().join("skippy/scripts/verify-native-runtime-package.sh"),
        fixture
            .root
            .join("skippy/scripts/verify-native-runtime-package.sh"),
    )
    .unwrap();
    fs::create_dir(fixture.root.join("restore-input")).unwrap();
    fs::create_dir(fixture.root.join("restore-tmp")).unwrap();
    fs::write(
        fixture.root.join("restore-input/sentinel"),
        b"download keep",
    )
    .unwrap();
    fs::copy(
        fixture.root.join("product.tar.gz"),
        fixture.root.join("restore-input/product.tar.gz"),
    )
    .unwrap();
    fs::write(fixture.root.join("events"), []).unwrap();
    fixture
}

fn restore(fixture: &Fixture, backend: &str) -> crate::process::RawProcessReport {
    let environment = fixture.environment(
        "product",
        "1.2.3",
        &[(
            "TMPDIR",
            fixture.root.join("restore-tmp").to_str().unwrap().into(),
        )],
    );
    invoke_arguments(
        &fixture.root,
        vec![
            Value::Public("-c".into()),
            Value::Public(extract_body(backend).into()),
        ],
        environment,
    )
}

fn replace_archive(fixture: &Fixture) {
    let mut tar = Vec::new();
    for (path, (bytes, mode)) in snapshot(&fixture.root.join("product")) {
        tar_member(&mut tar, path.to_str().unwrap(), &bytes, mode);
    }
    tar.extend_from_slice(&[0; 1024]);
    let mut gzip = GzEncoder::new(Vec::new(), Compression::default());
    gzip.write_all(&tar).unwrap();
    fs::write(
        fixture.root.join("restore-input/product.tar.gz"),
        gzip.finish().unwrap(),
    )
    .unwrap();
}

fn stage(fixture: &Fixture, name: &str) -> crate::process::RawProcessReport {
    let document = document();
    let selected: Vec<_> = steps(&document)
        .iter()
        .filter(|step| step.get("name").and_then(Node::text) == Some(name))
        .collect();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].get("shell").and_then(Node::text), Some("bash"));
    let body = selected[0]
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .replace("${{ inputs.artifact_path }}", "restore-input")
        .replace("${{ inputs.binary_name }}", "mesh-llm")
        .replace("${{ inputs.staged_binary_path }}", "staged/bin/mesh-llm");
    assert!(!body.contains("${{"));
    invoke_arguments(
        &fixture.root,
        vec![Value::Public("-c".into()), Value::Public(body.into())],
        fixture.environment("product", "1.2.3", &[]),
    )
}

#[test]
fn actual_smoke_consumer_reverifies_archive_before_publishing_exact_product() {
    let fixture = prepared();
    let before = snapshot(&fixture.root.join("inputs"));
    let expected = snapshot(&fixture.root.join("product"));
    let report = restore(&fixture, "cpu");
    assert!(report.process.success(), "{:?}", report);
    assert_eq!(snapshot(&fixture.root.join("inputs")), before);
    let mut restored = snapshot(&fixture.root.join("restore-input"));
    assert_eq!(
        restored.remove(Path::new("sentinel")).unwrap().0,
        b"download keep"
    );
    assert_eq!(restored, expected);
    assert!(!fixture.root.join("restore-input/product.tar.gz").exists());
    for name in ["Stage mesh-llm binary", "Stage packaged native runtime"] {
        let report = stage(&fixture, name);
        assert!(report.process.success(), "{name}: {report:?}");
    }
    let mut expected_stage = std::collections::BTreeMap::new();
    for (path, value) in &expected {
        if path == Path::new("mesh-llm") || path.starts_with("native-runtimes") {
            expected_stage.insert(path.clone(), value.clone());
        }
    }
    assert_eq!(snapshot(&fixture.root.join("staged/bin")), expected_stage);
    assert_eq!(
        fixture.events(),
        [
            "artifact extract-tar",
            "automation smoke-inputs",
            "native verify-runtime-package",
            "product compose"
        ]
    );
    assert_eq!(
        fs::read_dir(fixture.root.join("restore-tmp"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn actual_smoke_consumer_refuses_ambiguous_corrupt_or_wrong_backend_before_publication() {
    for case in ["missing", "duplicate", "host", "runtime", "backend"] {
        let fixture = prepared();
        match case {
            "missing" => {
                fs::remove_file(fixture.root.join("restore-input/product.tar.gz")).unwrap()
            }
            "duplicate" => {
                fs::copy(
                    fixture.root.join("restore-input/product.tar.gz"),
                    fixture.root.join("restore-input/duplicate.tar.gz"),
                )
                .unwrap();
            }
            "host" => {
                fs::write(fixture.root.join("product/mesh-llm"), b"host drift").unwrap();
                replace_archive(&fixture);
            }
            "runtime" => {
                fs::write(
                    fixture
                        .root
                        .join("product/native-runtimes/runtime/lib/runtime.bin"),
                    b"runtime drift",
                )
                .unwrap();
                replace_archive(&fixture);
            }
            "backend" => {}
            _ => unreachable!(),
        }
        let before = snapshot(&fixture.root.join("restore-input"));
        let inputs = snapshot(&fixture.root.join("inputs"));
        let report = restore(&fixture, if case == "backend" { "cuda" } else { "cpu" });
        assert!(!report.process.success(), "{case}: {:?}", report);
        assert_eq!(
            snapshot(&fixture.root.join("restore-input")),
            before,
            "{case}"
        );
        assert_eq!(snapshot(&fixture.root.join("inputs")), inputs, "{case}");
        assert!(
            !fixture
                .events()
                .iter()
                .any(|event| event.starts_with("forbidden"))
        );
        assert_eq!(
            fs::read_dir(fixture.root.join("restore-tmp"))
                .unwrap()
                .count(),
            0
        );
    }
}

#[test]
fn smoke_model_handoff_forwards_every_shared_input_and_consumed_output() {
    let document = document();
    let selected: Vec<_> = steps(&document)
        .iter()
        .filter(|step| step.get("id").and_then(Node::text) == Some("resolve-model"))
        .collect();
    assert_eq!(selected.len(), 1);
    let step = selected[0];
    assert_eq!(
        step.get("uses").and_then(Node::text),
        Some("./.github/actions/restore-test-model")
    );
    assert!(step.get("run").is_none());
    for key in [
        "model_url",
        "model_file",
        "model_manifest",
        "model_artifact_id",
        "model_cadence",
        "model_cache_scope",
        "cache_key_prefix",
        "save_model_cache",
    ] {
        assert_eq!(
            step.get("with").unwrap().get(key).and_then(Node::text),
            Some(format!("${{{{ inputs.{key} }}}}").as_str()),
            "{key}"
        );
    }
    for key in [
        "model_url",
        "model_file",
        "model_sha256",
        "model_size_bytes",
        "model_path",
    ] {
        assert_eq!(
            document
                .get("outputs")
                .unwrap()
                .get(key)
                .unwrap()
                .get("value")
                .and_then(Node::text),
            Some(format!("${{{{ steps.resolve-model.outputs.{key} }}}}").as_str()),
            "{key}"
        );
    }
}

#[test]
fn product_action_binds_verified_inputs_and_publishes_the_composer_receipt() {
    let document = workflow_yaml::parse(
        &fs::read_to_string(repository().join(".github/actions/compose-product-input/action.yml"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        document
            .get("runs")
            .unwrap()
            .get("using")
            .and_then(Node::text),
        Some("composite")
    );
    assert_eq!(steps(&document).len(), 1);
    let step = &steps(&document)[0];
    assert_eq!(step.get("id").and_then(Node::text), Some("compose"));
    assert_eq!(step.get("shell").and_then(Node::text), Some("bash"));
    let bindings = step.get("env").unwrap();
    let keys = [
        "host_input_dir",
        "runtime_input_dir",
        "output_dir",
        "backend",
        "version",
        "binary_name",
        "readiness_smoke",
        "attestation_public_key_file",
        "attestation_verifier",
    ];
    assert_eq!(bindings.entries().len(), keys.len());
    for key in keys {
        assert_eq!(
            bindings
                .get(&format!("INPUT_{}", key.to_uppercase()))
                .and_then(Node::text),
            Some(format!("${{{{ inputs.{key} }}}}").as_str()),
            "{key}"
        );
    }
    let fixture = Fixture::new("1.2.3");
    let before = snapshot(&fixture.root.join("inputs"));
    let body = step.get("run").and_then(Node::text).unwrap();
    let report = invoke_arguments(
        &fixture.root,
        vec![Value::Public("-c".into()), Value::Public(body.into())],
        fixture.environment("product", "1.2.3", &[]),
    );
    fixture.accepted(&report, &before);
    for key in [
        "product_dir",
        "binary_path",
        "runtime_root",
        "runtime_dir",
        "archive_path",
    ] {
        assert_eq!(
            document
                .get("outputs")
                .unwrap()
                .get(key)
                .unwrap()
                .get("value")
                .and_then(Node::text),
            Some(format!("${{{{ steps.compose.outputs.{key} }}}}").as_str()),
            "{key}"
        );
        assert!(
            fs::read_to_string(fixture.root.join("github-output"))
                .unwrap()
                .lines()
                .any(|line| line.starts_with(&format!("{key}=")))
        );
    }
}
