use serde_json::{Value, json};
use std::process::Command;

fn inspect(id: &str, path: &str) -> Value {
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "family-model-identity", id, path])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}

#[test]
fn hf_nested_first_shard_preserves_source_identity_and_quant_casing() {
    let pin = "a".repeat(40);
    let result = inspect(
        "explicit id",
        &format!(
            "/cache/models--Org--Repo/snapshots/{pin}/nested/Q4_K_M/Model-ud-q4_k_xl-00001-of-00003.gguf"
        ),
    );
    assert_eq!(
        result,
        json!({
            "model_id": "explicit id", "source_repo": "Org/Repo", "source_revision": pin,
            "source_file": "nested/Q4_K_M/Model-ud-q4_k_xl-00001-of-00003.gguf",
            "canonical_ref": format!("Org/Repo@{pin}/nested/Q4_K_M/Model-ud-q4_k_xl-00001-of-00003.gguf"),
            "distribution_id": "Model-ud-q4_k_xl", "selector": "ud-q4_k_xl"
        })
    );
}

#[test]
fn ordinary_and_incomplete_snapshot_paths_do_not_infer_source_or_pin() {
    for path in [
        "/local/Model-Q4_K_M.gguf",
        "/local/snapshots/pin/model.gguf",
        "/cache/models--Org--Repo/snapshots/pin",
        "models--Org--Repo/other/pin/model.gguf",
    ] {
        assert_eq!(
            inspect("Org/Repo:Q4_K_M", path),
            json!({"model_id": "Org/Repo:Q4_K_M"})
        );
    }
}

#[test]
fn second_shard_and_non_gguf_do_not_claim_first_shard_quant() {
    let second = inspect(
        "id",
        "models--Org--Repo/snapshots/pin/Model-Q8_0-00002-of-00003.gguf",
    );
    assert_eq!(second["distribution_id"], "Model-Q8_0-00002-of-00003");
    assert!(second.get("selector").is_none());
    let other = inspect("id", "models--Org--Repo/snapshots/pin/Model-F16.bin");
    assert_eq!(other["distribution_id"], "Model-F16.bin");
    assert!(other.get("selector").is_none());
}

#[test]
fn malformed_cli_arity_fails_without_identity_output() {
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "family-model-identity", "id"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
}

#[test]
fn quant_inference_does_not_mistake_model_names_for_quant_selectors() {
    for file in ["Quality.gguf", "IQData.gguf", "Qwen3.gguf"] {
        assert!(
            inspect("id", &format!("models--Org--Repo/snapshots/pin/{file}"))
                .get("selector")
                .is_none()
        );
    }
    for (file, selector) in [
        ("Qwen3_Q4_K_M.gguf", "Q4_K_M"),
        ("Model-UD-IQ2_XS.gguf", "UD-IQ2_XS"),
        ("Model-F16.gguf", "F16"),
    ] {
        assert_eq!(
            inspect("id", &format!("models--Org--Repo/snapshots/pin/{file}"))["selector"],
            selector
        );
    }
}
