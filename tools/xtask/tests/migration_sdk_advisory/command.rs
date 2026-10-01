use super::*;
use crate::automation::sdk_advisory::command::report;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn args(directory: &Path) -> Vec<String> {
    let event = directory.join("main-linux-event.json");
    let run = directory.join("main-linux-run.json");
    let artifacts = directory.join("main-linux-artifacts.json");
    vec![
        "--event-name".into(),
        "workflow_run".into(),
        "--controller-repository".into(),
        "Mesh-LLM/mesh-llm".into(),
        "--controller-ref".into(),
        "refs/heads/main".into(),
        "--event".into(),
        event.to_string_lossy().into_owned(),
        "--producer-run".into(),
        run.to_string_lossy().into_owned(),
        "--artifacts".into(),
        artifacts.to_string_lossy().into_owned(),
    ]
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/migration_sdk_advisory/fixtures")
}

#[test]
fn saved_evidence_command_returns_only_metadata_selection() {
    let arguments = args(&fixtures());

    let output = report(&root(), &arguments);

    assert_eq!(output.code, 0, "{}", output.stderr);
    let selected = value(output.stdout.as_bytes());
    assert_eq!(selected["validation"], "offline_metadata_selection");
    assert_eq!(selected["producer_run_id"], 101);
    assert_eq!(selected["products"].as_array().expect("products").len(), 2);
    assert!(output.stderr.is_empty());
}

#[test]
fn unknown_event_reports_failure_without_a_selection() {
    let mut arguments = args(&fixtures());
    arguments[1] = "pull_request".into();

    let output = report(&root(), &arguments);

    assert_eq!(output.code, 1);
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}

#[test]
fn missing_artifact_evidence_cannot_trigger_a_rebuild() {
    let directory = tempfile::tempdir().expect("isolated files");
    std::fs::write(directory.path().join("main-linux-event.json"), LINUX_EVENT).expect("event");
    std::fs::write(directory.path().join("main-linux-run.json"), LINUX_RUN).expect("run");
    let arguments = args(directory.path());

    let output = report(&root(), &arguments);

    assert_eq!(output.code, 1);
    assert!(output.stdout.is_empty());
    assert_eq!(
        std::fs::read_dir(directory.path())
            .expect("evidence directory")
            .count(),
        2
    );
}

#[test]
fn expired_artifact_failure_emits_no_partial_result() {
    let directory = tempfile::tempdir().expect("isolated files");
    std::fs::write(directory.path().join("main-linux-event.json"), LINUX_EVENT).expect("event");
    std::fs::write(directory.path().join("main-linux-run.json"), LINUX_RUN).expect("run");
    let mut artifacts = value(LINUX_ARTIFACTS);
    artifacts["artifacts"][2]["expired"] = Value::from(true);
    std::fs::write(
        directory.path().join("main-linux-artifacts.json"),
        bytes(&artifacts),
    )
    .expect("artifacts");

    let output = report(&root(), &args(directory.path()));

    assert_eq!(output.code, 1);
    assert!(output.stdout.is_empty());
}

#[test]
fn duplicate_options_unknown_options_and_missing_values_are_usage_errors() {
    for suffix in [
        vec!["--event-name", "workflow_dispatch"],
        vec!["--trusted"],
        vec!["--artifact-url", "https://example.test/archive"],
        vec!["--event"],
    ] {
        let mut arguments = args(&fixtures());
        arguments.extend(suffix.into_iter().map(str::to_owned));

        let output = report(&root(), &arguments);

        assert_eq!(output.code, 2);
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn help_needs_no_repository_or_evidence_files() {
    let missing = Path::new("/nonexistent-sdk-advisory-root");

    let output = report(missing, &["--help".into()]);

    assert_eq!(output.code, 0);
    assert!(output.stdout.contains("--producer-run"));
}
