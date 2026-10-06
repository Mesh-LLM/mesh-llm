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
