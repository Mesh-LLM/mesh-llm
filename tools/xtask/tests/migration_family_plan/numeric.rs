use super::*;

pub(super) fn replace(raw: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    let start = raw
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("fixture token");
    let mut changed = raw.to_vec();
    changed.splice(start..start + needle.len(), replacement.iter().copied());
    changed
}

pub(super) fn generate(raw: &[u8], families: &str) -> Output {
    let path = temp_path("raw-manifest.json");
    fs::write(&path, raw).expect("manifest");
    let result = run(&[
        "--manifest",
        path.to_str().expect("path"),
        "--families",
        families,
    ]);
    fs::remove_file(path).expect("cleanup");
    result
}

#[test]
fn shard_total_is_exact_when_multiple_u64_weights_exceed_u64_sum() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"estimated_model_bytes\": 9",
        b"\"estimated_model_bytes\": 18446744073709551615",
    );
    let raw = replace(
        &raw,
        b"\"estimated_model_bytes\": 9",
        b"\"estimated_model_bytes\": 1",
    );
    let output = generate(&raw, "zeta,beta");
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .matches("\"estimated_work_bytes\": 18446744073709551616")
            .count(),
        2
    );
}

#[test]
fn out_of_range_model_input_is_rejected_even_when_unselected() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"estimated_model_bytes\": 9",
        b"\"estimated_model_bytes\": 18446744073709551616",
    );
    let output = generate(&raw, "beta");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}

#[test]
fn integer_fields_reject_float_bool_negative_and_object_values() {
    for token in ["0.0", "false", "-1", r#"{"number":0}"#] {
        let raw = replace(
            &fixture("synthetic-manifest", "json"),
            b"\"mtp_layers\": 0",
            format!("\"mtp_layers\": {token}").as_bytes(),
        );
        let output = generate(&raw, "zeta");
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn layer_end_is_exact_when_trunk_and_mtp_sum_exceeds_u64() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"trunk_layers\": 1",
        b"\"trunk_layers\": 18446744073709551615",
    );
    let raw = replace(&raw, b"\"mtp_layers\": 0", b"\"mtp_layers\": 1");
    let output = generate(&raw, "zeta");
    assert_eq!(output.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("\"layer_end\": 18446744073709551616")
    );
}

#[test]
fn adjacent_u64_weights_retain_assignment_order() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"estimated_model_bytes\": 9}",
        b"\"estimated_model_bytes\": 9007199254740993}",
    );
    let raw = replace(
        &raw,
        b"\"estimated_model_bytes\": 9}",
        b"\"estimated_model_bytes\": 9007199254740992}",
    );
    let path = temp_path("adjacent.json");
    fs::write(&path, raw).expect("manifest");
    let output = run(&[
        "--manifest",
        path.to_str().expect("path"),
        "--families",
        "zeta,beta",
        "--shard-count",
        "2",
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value = serde_json::from_slice(&output.stdout).expect("plan");
    assert_eq!(plan["shards"][0]["families"], json!(["zeta"]));
    assert_eq!(plan["github_matrix"]["include"][0]["families"], "beta");
    fs::remove_file(path).expect("cleanup");
}

#[test]
fn bounded_resources_reject_large_integer_input() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"estimated_model_bytes\": 9",
        b"\"estimated_model_bytes\": 9, \"startup_timeout_secs\": 18446744073709551616",
    );
    let output = generate(&raw, "zeta");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}
