use super::*;
#[test]
fn paired_parity_rejects_empty_short_invalid_and_changed_structured_continuations() {
    let valid = json!({"passed":true,"results":[{"request_index":0,"valid":true,"completion_tokens":32,"content_sha256":"a".repeat(64)}]});
    assert!(parity_equal(&valid, &valid));
    for changed in [
        json!({"passed":true,"results":[]}),
        json!({"passed":false,"results":valid["results"]}),
        json!({"passed":true,"results":[{"request_index":0,"valid":true,"completion_tokens":31,"content_sha256":"a".repeat(64)}]}),
        json!({"passed":true,"results":[{"request_index":0,"valid":true,"completion_tokens":32,"content_sha256":"b".repeat(64)}]}),
    ] {
        assert!(!parity_equal(&valid, &changed));
    }
    let cell = json!({"platform":"metal","model":"dense","workload":"synthetic","arm":"mesh","concurrency":1,"output_tokens":8});
    let gate = json!({"passed":true,"cell":cell});
    assert!(find_gate(std::slice::from_ref(&gate), &cell));
    let mut changed = cell.clone();
    changed["output_tokens"] = 64.into();
    assert!(!find_gate(&[gate], &changed));
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("worker")).unwrap();
    for mean in [Value::Null, json!(0), json!(-1)] {
        crate::command::write_json_file(
            &root.path().join("worker/result.json"),
            &json!({"benchmarks":[{"tg_throughput":{"mean":mean}}]}),
        )
        .unwrap();
        assert!(throughput(root.path(), &cell, &Value::Null).is_err());
    }
    root.close().unwrap();
}
#[test]
fn promotion_consumer_uses_actual_paired_continuation_gate_before_selecting_gain() {
    let source = json!({"concurrency":[1],"synthetic":{"output_tokens":[8]}});
    let parity = json!({"passed":true,"results":[{"request_index":0,"valid":true,"completion_tokens":32,"content_sha256":"a".repeat(64)}]});
    let mut cells = Vec::new();
    let mut rows = Vec::new();
    for arm in ["llama", "mesh", "mesh-adaptive"] {
        for workload in ["synthetic", "thoughtworks"] {
            let cell = json!({"platform":"metal","model":"dense","arm":arm,"workload":workload,"concurrency":1,"output_tokens":if workload=="synthetic"{8}else{256}});
            cells.push(cell.clone());
            rows.push(json!({"cell":cell,"throughput":if arm=="mesh-adaptive"{120.0}else{100.0},"complete":true,"parity":parity,"capacity_policy":{"mode":"declared-shared-context","comparison_kv_matched":false}}));
        }
    }
    let passing = promotion::evaluate(&source, &cells, &rows, &gates(&rows)).unwrap();
    assert_eq!(
        serde_json::to_value(&passing).unwrap()[0]["winner"],
        "mesh-adaptive"
    );
    let candidate = rows
        .iter_mut()
        .find(|r| r["cell"]["arm"] == "mesh-adaptive" && r["cell"]["workload"] == "synthetic")
        .unwrap();
    candidate["parity"]["results"][0]["content_sha256"] = json!("b".repeat(64));
    let failed = promotion::evaluate(&source, &cells, &rows, &gates(&rows)).unwrap();
    let value = serde_json::to_value(&failed).unwrap();
    assert!(value[0]["winner"].is_null());
    assert_eq!(value[0]["candidates"][0]["c1_parity"], false);
    assert!(promotion::markdown(&failed).contains("hold"));
}
