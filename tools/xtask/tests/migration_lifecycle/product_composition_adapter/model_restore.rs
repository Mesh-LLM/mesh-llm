//! Actual model action bodies with private inert payloads and no network.
use super::{Fixture, Value, digest, executable, invoke_arguments, snapshot};
use crate::workflow_yaml::{self, Node};
use std::{collections::BTreeMap, fs, path::Path};

fn action() -> Node {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    workflow_yaml::parse(
        &fs::read_to_string(repository.join(".github/actions/restore-test-model/action.yml"))
            .unwrap(),
    )
    .unwrap()
}

fn step(name: &str) -> Node {
    let document = action();
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("composite steps required");
    };
    let selected: Vec<_> = steps
        .iter()
        .filter(|step| step.get("name").and_then(Node::text) == Some(name))
        .collect();
    assert_eq!(selected.len(), 1, "{name}");
    selected[0].clone()
}

fn run(
    fixture: &Fixture,
    name: &str,
    overrides: &[(&str, String)],
) -> crate::process::RawProcessReport {
    let body = step(name)
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .to_owned();
    assert!(!body.contains("${{"));
    let mut environment = fixture.environment("product", "1.2.3", overrides);
    environment.insert(
        "HOME".into(),
        Value::Public(fixture.root.join("model-home").into()),
    );
    environment.insert(
        "GITHUB_OUTPUT".into(),
        Value::Public(fixture.root.join("model-output").into()),
    );
    invoke_arguments(
        &fixture.root,
        vec![Value::Public("-c".into()), Value::Public(body.into())],
        environment,
    )
}

fn fields(fixture: &Fixture) -> BTreeMap<String, String> {
    fs::read_to_string(fixture.root.join("model-output"))
        .unwrap()
        .lines()
        .map(|line| {
            let (key, value) = line.split_once('=').unwrap();
            (key.into(), value.into())
        })
        .collect()
}

fn fixture(id: &str, cadence: &str) -> (Fixture, Vec<(&'static str, String)>, Vec<u8>) {
    let fixture = Fixture::new("1.2.3");
    fs::create_dir_all(fixture.root.join("model-home/.models")).unwrap();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(repository.join("ci/model-artifacts/manifests/skippy-ci-smoke.json")).unwrap(),
    )
    .unwrap();
    let payload = format!("GGUF inert integrity fixture {id}\n").into_bytes();
    let row = manifest["artifacts"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["id"] == id)
        .unwrap();
    let file = row["file"].as_str().unwrap().to_owned();
    row["size_bytes"] = payload.len().into();
    row["sha256"] = digest(&payload).into();
    row["file_integrity"][&file]["size_bytes"] = payload.len().into();
    row["file_integrity"][&file]["blob_id"] = digest(&payload).into();
    fs::write(
        fixture.root.join("model-manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    fs::write(fixture.root.join("payload"), &payload).unwrap();
    fs::write(fixture.root.join("model-output"), []).unwrap();
    let overrides = vec![
        (
            "MODEL_MANIFEST",
            fixture
                .root
                .join("model-manifest.json")
                .display()
                .to_string(),
        ),
        ("MODEL_ARTIFACT_ID", id.into()),
        ("MODEL_CADENCE", cadence.into()),
        ("INPUT_MODEL_URL", String::new()),
        ("INPUT_MODEL_FILE", String::new()),
    ];
    (fixture, overrides, payload)
}

fn curl_observer(fixture: &Fixture, fail: bool) {
    executable(
        &fixture.root.join("bin/curl"),
        &format!(
            r#"#!/bin/sh
[ "$#" -eq 13 ] || exit 95
[ "$1" = --fail ] && [ "$2" = --location ] && [ "$3" = --show-error ] || exit 95
[ "$4" = --retry ] && [ "$5" = 12 ] || exit 95
[ "$6" = --retry-delay ] && [ "$7" = 20 ] || exit 95
[ "$8" = --retry-max-time ] && [ "$9" = 600 ] || exit 95
shift 9
[ "$1" = --retry-all-errors ] && [ "$2" = "$MODEL_URL" ] || exit 95
[ "$3" = -o ] && [ "$4" = "$HOME/.models/$MODEL_FILE.download" ] || exit 95
printf 'curl inert-payload\n' >> "$PRODUCT_EVENTS"
cp payload "$4" || exit 95
exit {}
"#,
            if fail { 73 } else { 0 }
        ),
    );
}

#[test]
fn model_action_selects_both_manifest_artifacts_downloads_and_verifies_same_bytes() {
    for (id, cadence) in [
        ("family-qwen3-dense", "pull-request"),
        ("family-falcon-h1", "manual"),
    ] {
        let (fixture, mut overrides, payload) = fixture(id, cadence);
        assert!(
            run(&fixture, "Resolve immutable test model", &overrides)
                .process
                .success()
        );
        let resolved = fields(&fixture);
        assert_eq!(resolved["sha256"], digest(&payload));
        assert_eq!(resolved["size_bytes"], payload.len().to_string());
        overrides.extend([
            ("MODEL_URL", resolved["url"].clone()),
            ("MODEL_FILE", resolved["file"].clone()),
        ]);
        curl_observer(&fixture, false);
        assert!(
            run(&fixture, "Check restored integration model", &overrides)
                .process
                .success()
        );
        assert_eq!(fields(&fixture)["present"], "false");
        assert!(
            run(&fixture, "Download integration model", &overrides)
                .process
                .success()
        );
        assert!(
            run(&fixture, "Verify integration model integrity", &overrides)
                .process
                .success()
        );
        assert!(
            run(&fixture, "Publish resolved model path", &overrides)
                .process
                .success()
        );
        assert_eq!(fs::read(&fields(&fixture)["path"]).unwrap(), payload);
        assert!(
            run(&fixture, "Check restored integration model", &overrides)
                .process
                .success()
        );
        assert_eq!(fields(&fixture)["present"], "true");
        assert_eq!(
            fixture
                .events()
                .iter()
                .filter(|e| e.as_str() == "curl inert-payload")
                .count(),
            1
        );
        assert!(!fixture.events().iter().any(|e| e.starts_with("forbidden")));
    }
}

#[test]
fn model_action_rejects_corrupt_cache_and_failed_download_without_publishing_path() {
    for case in ["checksum", "size", "download"] {
        let (fixture, mut overrides, payload) = fixture("family-qwen3-dense", "main");
        assert!(
            run(&fixture, "Resolve immutable test model", &overrides)
                .process
                .success()
        );
        let resolved = fields(&fixture);
        overrides.extend([
            ("MODEL_URL", resolved["url"].clone()),
            ("MODEL_FILE", resolved["file"].clone()),
        ]);
        let path = fixture
            .root
            .join("model-home/.models")
            .join(&resolved["file"]);
        let corrupt = if case == "size" {
            b"short".to_vec()
        } else {
            vec![b'x'; payload.len()]
        };
        fs::write(&path, &corrupt).unwrap();
        let before = fs::read(fixture.root.join("model-output")).unwrap();
        if case == "download" {
            curl_observer(&fixture, true);
            assert!(
                !run(&fixture, "Download integration model", &overrides)
                    .process
                    .success()
            );
        } else {
            assert!(
                !run(&fixture, "Verify integration model integrity", &overrides)
                    .process
                    .success()
            );
        }
        assert_eq!(fs::read(path).unwrap(), corrupt);
        assert_eq!(fs::read(fixture.root.join("model-output")).unwrap(), before);
        assert!(!fields(&fixture).contains_key("path"));
        assert!(!fixture.events().iter().any(|e| e.starts_with("forbidden")));
    }
}

#[test]
fn model_action_empty_request_keeps_cache_untouched_and_publishes_empty_path() {
    let (fixture, _, _) = fixture("family-qwen3-dense", "main");
    let before = snapshot(&fixture.root.join("model-home"));
    let overrides = [
        ("MODEL_MANIFEST", String::new()),
        ("MODEL_ARTIFACT_ID", String::new()),
        ("MODEL_CADENCE", String::new()),
        ("INPUT_MODEL_URL", String::new()),
        ("INPUT_MODEL_FILE", String::new()),
    ];
    assert!(
        run(&fixture, "Resolve immutable test model", &overrides)
            .process
            .success()
    );
    for key in ["url", "file", "sha256", "size_bytes"] {
        assert_eq!(fields(&fixture)[key], "");
    }
    assert!(
        run(
            &fixture,
            "Publish resolved model path",
            &[("MODEL_FILE", String::new())]
        )
        .process
        .success()
    );
    assert_eq!(fields(&fixture)["path"], "");
    assert_eq!(snapshot(&fixture.root.join("model-home")), before);
    assert!(
        !fixture
            .events()
            .iter()
            .any(|e| e.starts_with("forbidden") || e.starts_with("curl"))
    );
}

#[test]
fn model_action_cache_authority_and_failure_order_follow_resolved_integrity() {
    let document = action();
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("composite steps required");
    };
    let names: Vec<_> = steps
        .iter()
        .map(|step| step.get("name").and_then(Node::text).unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "Resolve immutable test model",
            "Restore integration model cache",
            "Check restored integration model",
            "Download integration model",
            "Verify integration model integrity",
            "Save integration model cache",
            "Publish resolved model path"
        ]
    );
    let present = "steps.resolve-model.outputs.url != '' && steps.resolve-model.outputs.file != ''";
    for name in [
        "Restore integration model cache",
        "Check restored integration model",
    ] {
        assert_eq!(
            step(name).get("if").and_then(Node::text),
            Some(format!("${{{{ {present} }}}}").as_str())
        );
    }
    assert_eq!(
        step("Download integration model")
            .get("if")
            .and_then(Node::text),
        Some(
            format!("${{{{ {present} && steps.model-file.outputs.present != 'true' }}}}").as_str()
        )
    );
    assert_eq!(
        step("Verify integration model integrity")
            .get("if")
            .and_then(Node::text),
        Some(
            "${{ steps.resolve-model.outputs.sha256 != '' && steps.resolve-model.outputs.file != '' }}"
        )
    );
    assert_eq!(step("Save integration model cache").get("if").and_then(Node::text),
        Some(format!("${{{{ {present} && inputs.save_model_cache == 'true' && steps.cache-model.outputs.cache-hit != 'true' }}}}").as_str()));
    let cache = step("Restore integration model cache");
    assert_eq!(cache.get("id").and_then(Node::text), Some("cache-model"));
    assert_eq!(
        cache.get("uses").and_then(Node::text),
        Some("actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25")
    );
    assert_eq!(
        cache.get("with").unwrap().get("key").and_then(Node::text),
        Some(
            "${{ inputs.cache_key_prefix }}mesh-llm-${{ runner.os }}-${{ inputs.model_cache_scope }}-${{ steps.resolve-model.outputs.file }}-${{ steps.resolve-model.outputs.sha256 || 'unverified' }}-${{ hashFiles('.github/cache-version.txt') }}"
        )
    );
    let save = step("Save integration model cache");
    assert_eq!(
        save.get("uses").and_then(Node::text),
        Some("actions/cache/save@caa296126883cff596d87d8935842f9db880ef25")
    );
    for node in [&cache, &save] {
        assert_eq!(
            node.get("with").unwrap().get("path").and_then(Node::text),
            Some("~/.models/${{ steps.resolve-model.outputs.file }}")
        );
    }
    assert_eq!(
        save.get("with").unwrap().get("key").and_then(Node::text),
        Some("${{ steps.cache-model.outputs.cache-primary-key }}")
    );
    let verification = step("Verify integration model integrity");
    for (key, expression) in [
        ("MODEL_ARTIFACT_ID", "${{ inputs.model_artifact_id }}"),
        ("MODEL_CADENCE", "${{ inputs.model_cadence }}"),
        ("MODEL_MANIFEST", "${{ inputs.model_manifest }}"),
        ("MODEL_FILE", "${{ steps.resolve-model.outputs.file }}"),
        ("MODEL_URL", "${{ steps.resolve-model.outputs.url }}"),
    ] {
        assert_eq!(
            verification
                .get("env")
                .unwrap()
                .get(key)
                .and_then(Node::text),
            Some(expression)
        );
    }
    for node in steps {
        assert!(node.get("continue-on-error").is_none() && node.get("permissions").is_none());
    }
}
