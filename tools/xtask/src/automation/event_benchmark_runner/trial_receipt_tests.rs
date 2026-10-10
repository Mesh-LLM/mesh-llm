use super::*;
use serde_json::json;

fn logs(stdout: &[u8], stderr: &[u8]) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("server.stdout.log"), stdout).unwrap();
    std::fs::write(directory.path().join("server.stderr.log"), stderr).unwrap();
    directory
}
fn health(count: u64) -> Vec<u8> {
    serde_json::to_vec(&json!({"context":"event_system_health", "message":format!("version=1 dropped_progress={count} ingress_p99_us=4")})).unwrap()
}
#[test]
fn absent_health_remains_unmeasured() {
    let directory = logs(b"unrelated\n", b"");
    assert_eq!(
        final_health(directory.path()).unwrap(),
        health_log::Observation::default()
    );
}
#[test]
fn final_health_includes_post_stop_file_tail_and_skips_malformed_lines() {
    let mut bytes = health(3);
    bytes.extend_from_slice(b"\ninvalid\n");
    bytes.extend_from_slice(&health(5));
    let directory = logs(b"", &bytes);
    let observed = final_health(directory.path()).unwrap();
    assert_eq!(observed.health.unwrap()["dropped_progress"], json!(5));
    assert_eq!(observed.ingress_p99_us, Some(4.0));
}
#[test]
fn separate_streams_cannot_prove_final_temporal_health() {
    let directory = logs(&health(1), &health(2));
    assert!(
        final_health(directory.path())
            .unwrap_err()
            .to_string()
            .contains("ambiguous")
    );
}
#[test]
fn oversized_log_line_is_discarded_with_bounded_retained_memory() {
    let mut bytes = vec![b'a'; MAX_HEALTH_LINE + 1];
    bytes.push(b'\n');
    bytes.extend_from_slice(&health(9));
    let directory = logs(&bytes, b"");
    assert_eq!(
        final_health(directory.path()).unwrap().health.unwrap()["dropped_progress"],
        json!(9)
    );
}
fn receipt() -> serde_json::Value {
    json!({"schema_version":1,"request_sha256":"c".repeat(64),"model":"served-model","prompt_sha256":"a".repeat(64),
        "readiness_ms":4.0,"warmup_ms":5.0,"warmup_error":null,
        "measurement":{"completion_tokens":2,"ttft_ms":1.0,"elapsed_ms":10.0,
            "decode_tok_s":200.0,"decode_only_tok_s":2.0 / 0.009,"malformed":false},"error":null})
}
#[test]
fn worker_receipt_binds_prompt_status_and_timing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("worker.json");
    let mut value = receipt();
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_ok());
    assert!(worker(&path, &"c".repeat(64), &"b".repeat(64), Some(0)).is_err());
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(1)).is_err());
    value["measurement"]["ttft_ms"] = json!(11.0);
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_err());
}
#[test]
fn failed_worker_retains_valid_partial_readiness_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("worker.json");
    let mut value = receipt();
    value["measurement"] = serde_json::Value::Null;
    value["error"] = json!("measurement deadline expired");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let observed = worker(&path, &"c".repeat(64), &"a".repeat(64), Some(1)).unwrap();
    assert_eq!(observed.readiness_ms, Some(4.0));
    assert!(observed.measurement.is_none());
    assert!(observed.error.is_some());
}
#[test]
fn oversized_receipt_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("worker.json");
    std::fs::write(&path, vec![b' '; MAX_RECEIPT as usize + 1]).unwrap();
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_err());
}

#[test]
fn worker_rates_must_match_native_metrics_recomputed_from_raw_observations() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("worker.json");
    for field in ["decode_tok_s", "decode_only_tok_s"] {
        let mut value = receipt();
        value["measurement"][field] = serde_json::json!(999.0);
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_err());
    }
}

#[test]
fn optional_rates_cannot_be_fabricated_when_underlying_intervals_are_missing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("worker.json");
    let mut value = receipt();
    value["measurement"]["ttft_ms"] = serde_json::Value::Null;
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_err());
    value["measurement"]["decode_only_tok_s"] = serde_json::Value::Null;
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_ok());
}

#[test]
fn native_epsilon_is_used_for_sub_microsecond_decode_intervals() {
    let mut value = super::super::stream_metrics::Measurement {
        completion_tokens: Some(2),
        elapsed_ms: 1.0,
        ttft_ms: Some(0.9995),
        decode_tok_s: Some(2000.0),
        decode_only_tok_s: Some(2_000_000.0),
        malformed: false,
    };
    assert!(measurement(&value));
    value.decode_only_tok_s = Some(4_000_000.0);
    assert!(!measurement(&value));
}

#[test]
fn lifecycle_requires_full_request_hash_alongside_status_error_and_prompt() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("worker.json");
    let mut value = receipt();
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(worker(&path, &"d".repeat(64), &"a".repeat(64), Some(0)).is_err());
    value.as_object_mut().unwrap().remove("request_sha256");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_err());
    value = receipt();
    value["error"] = json!("failed");
    value["measurement"] = serde_json::Value::Null;
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(0)).is_err());
    assert!(worker(&path, &"c".repeat(64), &"a".repeat(64), Some(1)).is_ok());
}
