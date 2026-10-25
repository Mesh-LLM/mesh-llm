use super::super::write_evidence;
use super::fixture::{Fixture, METRICS};
use std::fs;

#[test]
fn writer_binds_known_bytes_when_comparator_pass_is_last() {
    let fixture = Fixture::new();
    let expected = fixture.body();

    write_evidence(&fixture.write).unwrap();

    let raw = fs::read(&fixture.write.output).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&raw).unwrap(),
        expected
    );
    assert_eq!(raw.last(), Some(&b'\n'));
}

#[test]
fn writer_does_not_overwrite_when_log_or_lane_fails() {
    for (lane, log) in [
        (
            "embedding",
            "embedding local-monolithic oracle passed: fixture",
        ),
        (
            "embedding-smoke-extra",
            "embedding local-monolithic oracle passed: fixture",
        ),
        ("embedding-smoke", "embedding OpenAI HTTP smoke passed"),
        (
            "embedding-smoke",
            "embedding local-monolithic oracle passed: fixture\nfailed later",
        ),
        (
            "embedding-smoke",
            "rerank local-monolithic oracle passed: fixture",
        ),
        (
            "embedding-smoke",
            "embedding local-monolithic oracle passed:",
        ),
        ("embedding-smoke", " \r\n\t"),
    ] {
        let mut fixture = Fixture::new();
        fixture.write.smoke_lane = lane.into();
        fs::write(&fixture.write.comparison_log, log).unwrap();
        fs::write(&fixture.write.output, b"retained").unwrap();

        let result = write_evidence(&fixture.write);

        assert!(result.is_err(), "{lane}: {log}");
        assert_eq!(fs::read(&fixture.write.output).unwrap(), b"retained");
    }
}

#[test]
fn writer_strips_python_whitespace_and_only_replaces_terminal_suffix() {
    let mut fixture = Fixture::new();
    fixture.write.smoke_lane = "fixture-smoke-embedding-smoke".into();
    fs::write(
        &fixture.write.comparison_log,
        "ignored\r\n\t embedding local-monolithic oracle passed: fixture\u{a0}\t",
    )
    .unwrap();

    write_evidence(&fixture.write).unwrap();

    let body: serde_json::Value =
        serde_json::from_slice(&fs::read(&fixture.write.output).unwrap()).unwrap();
    assert_eq!(body["oracle_lane"], "fixture-smoke-embedding-oracle");
    assert_eq!(
        body["comparison"],
        "embedding local-monolithic oracle passed: fixture"
    );
}

#[test]
fn writer_preserves_supplied_opaque_identity_for_an_unknown_class() {
    let mut fixture = Fixture::new();
    fixture.select("future_class", "future-oracle");
    fixture.write.model_sha256 = "opaque-model-hash".into();
    fixture.write.pinned_patch_sha = "opaque-pin".into();
    fixture.write.model_id = "unicode-\u{e9}".into();

    write_evidence(&fixture.write).unwrap();

    let raw = fs::read_to_string(&fixture.write.output).unwrap();
    let body: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(body["class"], "future_class");
    assert_eq!(body["model_sha256"], "opaque-model-hash");
    assert_eq!(body["pinned_patch_sha"], "opaque-pin");
    assert!(raw.contains("unicode-\\u00e9"));
}

#[test]
fn writer_accepts_projector_classes_without_enforcing_verifier_prerequisites() {
    let mut fixture = Fixture::new();
    fixture.select("speech_synthesis", "arbitrary-reference");
    fixture.tts_result(METRICS);

    let result = write_evidence(&fixture.write);

    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn writer_rejects_tts_results_when_status_or_patch_or_metrics_are_absent() {
    for body in [
        "{}".to_owned(),
        "[]".to_owned(),
        format!(
            r#"{{"status":"fail","pinned_patch_sha":"{}","metrics":{METRICS}}}"#,
            "a".repeat(40)
        ),
        format!(
            r#"{{"status":"pass","pinned_patch_sha":"{}","metrics":{METRICS}}}"#,
            "b".repeat(40)
        ),
        format!(
            r#"{{"status":"pass","pinned_patch_sha":"{}"}}"#,
            "a".repeat(40)
        ),
    ] {
        let mut fixture = Fixture::new();
        fixture.select("speech_synthesis", "llama-tts");
        fs::write(fixture.write.work_dir.join("tts-oracle-result.json"), body).unwrap();

        let result = write_evidence(&fixture.write);

        assert!(result.is_err());
        assert!(!fixture.write.output.exists());
    }
}

#[test]
fn writer_rejects_missing_artifacts_without_creating_output() {
    let mut fixture = Fixture::new();
    fixture.write.candidate_executable = fixture.directory.path().join("missing");

    let result = write_evidence(&fixture.write);

    assert!(result.is_err());
    assert!(!fixture.write.output.exists());
}

#[test]
fn writer_does_not_create_missing_output_parent() {
    let mut fixture = Fixture::new();
    fixture.write.output = fixture.directory.path().join("missing/evidence.json");

    let result = write_evidence(&fixture.write);

    assert!(result.is_err());
    assert!(!fixture.directory.path().join("missing").exists());
}
