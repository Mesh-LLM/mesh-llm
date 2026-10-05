use super::*;
use serde_json::{Value, json};

fn input() -> Value {
    json!({"model_id":"fixture", "model_path":std::env::temp_dir().join("fixture.gguf"), "source_model_sha256":"a".repeat(64),
        "layer_end":28, "ctx_size":131072, "lane_count":16, "n_gpu_layers":999,
        "payload":"kv-recurrent", "cache_entries":16})
}

#[test]
fn stage_configuration_preserves_runtime_slice_and_waiting_prefix_record_policy() {
    let input: Input = serde_json::from_value(input()).unwrap();
    let output = serde_json::to_value(config(&input).unwrap()).unwrap();
    assert_eq!(output["model_id"], "fixture");
    assert_eq!(output["source_model_sha256"], "a".repeat(64));
    assert_eq!(output["layer_start"], 0);
    assert_eq!(output["layer_end"], 28);
    assert_eq!(output["lane_count"], 16);
    assert_eq!(output["ctx_size"], 131072);
    assert_eq!(output["load_mode"], "runtime-slice");
    assert_eq!(
        output["kv_cache"],
        json!({"mode":"lookup-record", "payload":"kv-recurrent",
        "max_entries":16,"max_bytes":0,"min_tokens":64,"shared_prefix_stride_tokens":128,
        "shared_prefix_record_limit":1})
    );
    assert_eq!(output["upstream"], Value::Null);
    assert_eq!(output["downstream"], Value::Null);
}

#[test]
fn stage_admission_refuses_invalid_identity_bounds_and_unknown_payloads() {
    for (field, value) in [
        ("model_id", json!(" ")),
        ("model_path", json!("relative.gguf")),
        ("source_model_sha256", json!("x".repeat(64))),
        ("layer_end", json!(0)),
        ("ctx_size", json!(0)),
        ("lane_count", json!(0)),
        ("cache_entries", json!(0)),
        ("n_gpu_layers", json!(-2)),
    ] {
        let mut row = input();
        row[field] = value;
        assert!(
            serde_json::from_value::<Input>(row)
                .unwrap()
                .validate()
                .is_err(),
            "{field}"
        );
    }
    let mut row = input();
    row["payload"] = json!("unrecognized");
    assert!(serde_json::from_value::<Input>(row).is_err());
}
