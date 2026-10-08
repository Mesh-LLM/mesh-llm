//! Current model resolver action outputs consumed by the runtime-event lane.
use super::fixture::{invoke, repository};
use crate::{process::Value, workflow_yaml};
use std::{collections::BTreeMap, fs, path::Path};

fn action_body() -> String {
    let source =
        fs::read_to_string(repository().join(".github/actions/restore-test-model/action.yml"))
            .unwrap();
    let document = workflow_yaml::parse(&source).unwrap();
    let workflow_yaml::Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap()
    else {
        panic!("action steps required");
    };
    steps
        .iter()
        .find(|step| step.get("id").and_then(workflow_yaml::Node::text) == Some("resolve-model"))
        .unwrap()
        .get("run")
        .and_then(workflow_yaml::Node::text)
        .unwrap()
        .to_owned()
}

fn resolve_action(
    manifest: &Path,
    cadence: &str,
    output: &Path,
) -> crate::process::RawProcessReport {
    let environment = [
        ("PATH", std::env::var("PATH").unwrap()),
        (
            "MESH_LLM_AUTOMATION_BIN",
            env!("CARGO_BIN_EXE_xtask").into(),
        ),
        ("MODEL_MANIFEST", manifest.to_str().unwrap().into()),
        ("MODEL_ARTIFACT_ID", "family-qwen3-dense".into()),
        ("MODEL_CADENCE", cadence.into()),
        ("INPUT_MODEL_URL", String::new()),
        ("INPUT_MODEL_FILE", String::new()),
        ("GITHUB_OUTPUT", output.to_str().unwrap().into()),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), Value::Public(value.into())))
    .collect();
    invoke(
        "/bin/bash".into(),
        repository(),
        vec!["-c".into(), action_body()],
        environment,
    )
}

fn current_manifest() -> (std::path::PathBuf, serde_json::Value) {
    let path = repository().join("ci/model-artifacts/manifests/skippy-ci-smoke.json");
    let document = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    (path, document)
}

#[test]
fn actual_restore_action_emits_current_model_integrity_at_every_workflow_cadence() {
    let (manifest, document) = current_manifest();
    let artifact = document["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == "family-qwen3-dense")
        .unwrap();
    for cadence in ["pull-request", "main", "manual"] {
        let scratch = tempfile::tempdir().unwrap();
        let output = scratch.path().join("action outputs.txt");
        let report = resolve_action(&manifest, cadence, &output);
        assert!(report.process.success(), "{:?}", report.process);
        let text = fs::read_to_string(output).unwrap();
        let mut fields = BTreeMap::new();
        for line in text.lines() {
            let (key, value) = line.split_once('=').unwrap();
            assert!(
                fields.insert(key, value).is_none(),
                "duplicate action output"
            );
        }
        for key in ["file", "url", "sha256"] {
            let expected = artifact[key].as_str().unwrap();
            assert_eq!(fields.get(key).copied(), Some(expected), "{cadence}: {key}");
        }
        assert!(fields["file"].ends_with(".gguf"));
        assert_eq!(fields["sha256"].len(), 64);
        let size: u64 = fields["size_bytes"].parse().unwrap();
        assert!(size > 0);
        assert_eq!(Some(size), artifact["size_bytes"].as_u64());
    }
}

#[test]
fn actual_restore_action_denies_missing_cadence_without_writing_consumed_outputs() {
    let (_, original) = current_manifest();
    for cadence in ["pull-request", "main", "manual"] {
        let mut denied = original.clone();
        let artifact = denied["artifacts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|row| row["id"] == "family-qwen3-dense")
            .unwrap();
        artifact["cadences"]
            .as_array_mut()
            .unwrap()
            .retain(|value| value != cadence);
        let scratch = tempfile::tempdir().unwrap();
        let manifest = scratch.path().join("denied manifest.json");
        let output = scratch.path().join("action outputs.txt");
        fs::write(&manifest, serde_json::to_vec(&denied).unwrap()).unwrap();
        fs::write(&output, b"prior=preserved\n").unwrap();
        let report = resolve_action(&manifest, cadence, &output);
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains("is not allowed at cadence")
        );
        assert_eq!(fs::read(output).unwrap(), b"prior=preserved\n");
    }
}
