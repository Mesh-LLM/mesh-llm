use serde_json::Value;
use std::{
    path::Path,
    process::{Command, Output},
};
fn prepare(directory: &Path, template: Option<&Path>, history: &str, rerun: &str) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .args([
            "automation",
            "replay-matrix",
            "publication-prepare",
            "--publication-dir",
        ])
        .arg(directory)
        .args([
            "--run-id",
            "123",
            "--run-attempt",
            "2",
            "--base-sha",
            &"a".repeat(40),
            "--run-date",
            "2026-10-02",
            "--server-url",
            "https://github.com",
            "--dataset-repo",
            "meshllm/agentic-replay-nightly",
            "--fix-summary",
            "fix & | \\ {{SOURCE_SHA}}",
            "--files-changed",
            "crates/example.rs",
            "--history-outcome",
            history,
            "--rerun-outcome",
            rerun,
        ]);
    if let Some(template) = template {
        command.arg("--template").arg(template);
    }
    command
        .env("HF_TOKEN", "private-publication-token")
        .env("CANARY_REPAIR_TOKEN", "private-github-token")
        .output()
        .unwrap()
}
fn patch(directory: &Path) {
    std::fs::write(
        directory.join("repair.patch"),
        b"From abc\nSubject: fixture repair\n\ntext\n",
    )
    .unwrap();
}
fn verify(directory: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "publication-verify",
            "--publication-dir",
        ])
        .arg(directory)
        .args([
            "--base-sha",
            &"a".repeat(40),
            "--checkout-sha",
            &"a".repeat(40),
            "--run-id",
            "123",
            "--run-attempt",
            "2",
        ])
        .output()
        .unwrap()
}

#[test]
fn producer_roundtrips_through_trusted_verifier_without_capture_or_ambient_secrets() {
    let temporary = tempfile::tempdir().unwrap();
    patch(temporary.path());
    let template = temporary.path().join("template-input.md");
    // Template is an input outside the strict three-file publication directory.
    let publication = temporary.path().join("publication");
    std::fs::create_dir(&publication).unwrap();
    patch(&publication);
    std::fs::write(
        &template,
        "{{FIX_SUMMARY}}\n{{SOURCE_SHA}}\n{{RERUN_GATE_RESULT}}\n{{RESULT_ROWS}}\n",
    )
    .unwrap();
    let result = prepare(&publication, Some(&template), "success", "success");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let body = std::fs::read_to_string(publication.join("pr-body.md")).unwrap();
    assert!(body.starts_with("fix & | \\ {{SOURCE_SHA}}\n"));
    assert!(body.contains("pass\nsee history-repair.jsonl artifact"));
    let status: Value =
        serde_json::from_slice(&std::fs::read(publication.join("status.json")).unwrap()).unwrap();
    assert_eq!(status["resolution"], "fix-verified");
    assert!(verify(&publication).status.success());
    for name in ["pr-body.md", "status.json"] {
        let bytes = std::fs::read_to_string(publication.join(name)).unwrap();
        assert!(!bytes.contains("private-publication-token"));
        assert!(!bytes.contains("private-github-token"));
    }
    std::fs::write(publication.join("repair.patch"), b"tampered").unwrap();
    assert!(!verify(&publication).status.success());
}

#[test]
fn failed_cancelled_skipped_or_unknown_outcomes_never_emit_publication_metadata() {
    for (history, rerun) in [
        ("failure", "success"),
        ("success", "failure"),
        ("cancelled", "success"),
        ("success", "skipped"),
        ("true", "success"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        patch(temporary.path());
        let result = prepare(temporary.path(), None, history, rerun);
        assert!(!result.status.success());
        assert!(!temporary.path().join("status.json").exists());
        assert!(!temporary.path().join("pr-body.md").exists());
    }
}

#[test]
fn missing_template_fallback_is_supported_and_malformed_patch_has_no_status_receipt() {
    let temporary = tempfile::tempdir().unwrap();
    patch(temporary.path());
    assert!(
        prepare(temporary.path(), None, "success", "success")
            .status
            .success()
    );
    assert!(
        std::fs::read_to_string(temporary.path().join("pr-body.md"))
            .unwrap()
            .contains("passes the re-run benchmark")
    );
    let invalid = tempfile::tempdir().unwrap();
    std::fs::write(invalid.path().join("repair.patch"), b"not a format patch").unwrap();
    assert!(
        !prepare(invalid.path(), None, "success", "success")
            .status
            .success()
    );
    assert!(!invalid.path().join("status.json").exists());
}

#[cfg(unix)]
#[test]
fn symlink_payload_or_extra_private_capture_is_not_a_publishable_source() {
    let temporary = tempfile::tempdir().unwrap();
    let outside = temporary.path().join("outside.patch");
    std::fs::write(&outside, b"From abc\nSubject: outside\n").unwrap();
    let publication = temporary.path().join("publication");
    std::fs::create_dir(&publication).unwrap();
    std::os::unix::fs::symlink(&outside, publication.join("repair.patch")).unwrap();
    assert!(
        !prepare(&publication, None, "success", "success")
            .status
            .success()
    );
    assert!(!publication.join("status.json").exists());
    let capture = tempfile::tempdir().unwrap();
    patch(capture.path());
    std::fs::write(
        capture.path().join("private-capture.jsonl"),
        b"private prompt",
    )
    .unwrap();
    assert!(
        !prepare(capture.path(), None, "success", "success")
            .status
            .success()
    );
    assert!(!capture.path().join("status.json").exists());
}

#[test]
fn actual_repair_template_retains_run_provenance_and_artifact_only_result_context() {
    let temporary = tempfile::tempdir().unwrap();
    patch(temporary.path());
    let template = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.github/AGENTIC_REPLAY_REPAIR_PR_TEMPLATE.md");
    let result = prepare(temporary.path(), Some(&template), "success", "success");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let body = std::fs::read_to_string(temporary.path().join("pr-body.md")).unwrap();
    assert!(body.contains("https://github.com/Mesh-LLM/mesh-llm/actions/runs/123/attempts/2"));
    assert!(body.contains("data/runs/2026-10-02/123.jsonl"));
    assert!(body.contains("see history-repair.jsonl artifact"));
    assert!(body.contains("**Gate:** pass (bootstrap state: unavailable)"));
    assert!(verify(temporary.path()).status.success());
}
