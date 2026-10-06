use super::*;
use serde_json::json;
#[test]
fn matrix_uses_complete_ladder_with_all_arm_alternation_and_pinned_exclusion() {
    let root = tempfile::tempdir().unwrap();
    let config: Value = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../evals/skippy-competitive-benchmark.json"
    )))
    .unwrap();
    let artifact = json!({"path":root.path().join("inert"),"sha256":"a".repeat(64)});
    let backend = json!({"executable":artifact,"version_sha256":"b".repeat(64),"cwd":root.path(),"runtime":artifact,"tokenizer":artifact,"hf_config":null,"comparison_model":null,"match_kv_capacity":false});
    let input:Input=serde_json::from_value(json!({"config":root.path().join("config.json"),"config_sha256":"c".repeat(64),"platform":"metal","models":[{"key":"deepseek-v2-moe","model":artifact,"backends":{"llama":backend,"mesh":backend,"mesh-adaptive":backend}}],"workloads":["synthetic","thoughtworks"],"optional_arms":["vllm","sglang"],"required_comparisons":["vllm","sglang"],"adaptive":true,"manifest":null,"benchy":null,"output":root.path().join("output"),"timeout_seconds":120,"cell_timeout_seconds":20,"request_timeout_seconds":2,"resume":false,"force":false})).unwrap();
    let roster = super::super::competitive_roster::select(
        &config,
        &serde_json::to_vec(&config).unwrap(),
        &input,
    )
    .unwrap();
    assert_eq!(roster.cells.len(), 9 * 3 * 3 + 9 * 3);
    let trace: Vec<_> = roster
        .cells
        .iter()
        .filter(|cell| cell["workload"] == "thoughtworks")
        .collect();
    assert_eq!(
        trace
            .iter()
            .take(6)
            .map(|cell| cell["arm"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "llama",
            "mesh",
            "mesh-adaptive",
            "mesh-adaptive",
            "mesh",
            "llama"
        ]
    );
    for arm in ["llama", "mesh", "mesh-adaptive"] {
        for concurrency in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            assert!(
                trace
                    .iter()
                    .any(|cell| cell["arm"] == arm && cell["concurrency"] == concurrency)
            );
        }
    }
    assert!(
        roster.availability["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["selected"] == false
                && row["source_pinned_exclusion"] == true
                && row["reason"]
                    .as_str()
                    .is_some_and(|reason| !reason.is_empty()))
    );
    root.close().unwrap();
}
#[test]
fn resume_refuses_incomplete_or_hash_changed_cells_and_force_quarantines_original_bytes() {
    let root = tempfile::tempdir().unwrap();
    let cell = json!({"platform":"metal","model":"fixture","arm":"mesh","workload":"synthetic","output_tokens":8,"concurrency":1});
    let directory = cell_directory(root.path(), &cell).unwrap();
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("result.json"), b"original partial bytes").unwrap();
    assert!(
        !super::super::competitive_resume::completed(&directory, &cell, "config", &json!({}))
            .unwrap()
    );
    let moved = super::super::competitive_resume::quarantine(root.path(), &directory).unwrap();
    assert!(!directory.exists());
    assert_eq!(
        std::fs::read(moved.join("result.json")).unwrap(),
        b"original partial bytes"
    );
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("complete.json"),serde_json::to_vec(&json!({"schema_version":2,"scope":"competitive_retained_cell","completed":true,"cell":cell,"config_sha256":"wrong"})).unwrap()).unwrap();
    assert!(
        super::super::competitive_resume::completed(&directory, &cell, "config", &json!({}))
            .is_err()
    );
    assert_eq!(
        std::fs::read(moved.join("result.json")).unwrap(),
        b"original partial bytes"
    );
    root.close().unwrap();
}
#[test]
fn optional_matrix_requires_linux_cuda_and_preserves_required_refusal() {
    let root = tempfile::tempdir().unwrap();
    let mut config: Value = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../evals/skippy-competitive-benchmark.json"
    )))
    .unwrap();
    let artifact = json!({"path":root.path().join("inert"),"sha256":"a".repeat(64)});
    let backend = json!({"executable":artifact,"version_sha256":"b".repeat(64),"cwd":root.path(),"runtime":artifact,"tokenizer":artifact,"hf_config":null,"comparison_model":null,"match_kv_capacity":false});
    let mut input:Input=serde_json::from_value(json!({"config":root.path().join("config.json"),"config_sha256":"c".repeat(64),"platform":"cuda","models":[{"key":"llama32-dense","model":artifact,"backends":{"llama":backend,"mesh":backend,"vllm":backend,"sglang":backend}}],"workloads":["synthetic"],"optional_arms":["vllm","sglang"],"required_comparisons":[],"adaptive":false,"manifest":null,"benchy":null,"output":root.path().join("output"),"timeout_seconds":120,"cell_timeout_seconds":20,"request_timeout_seconds":2,"resume":false,"force":false})).unwrap();
    for arm in ["vllm", "sglang"] {
        config["models"][0]["comparison_support"][arm] = json!({"available":true});
    }
    let bytes = serde_json::to_vec(&config).unwrap();
    let off = super::super::competitive_roster::select_on(&config, &bytes, &input, false).unwrap();
    assert!(
        off.cells
            .iter()
            .all(|cell| cell["arm"] == "llama" || cell["arm"] == "mesh")
    );
    assert!(
        off.availability["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["selected"] == false
                && row["reason"] == "optional engine requires Linux CUDA")
    );
    let on = super::super::competitive_roster::select_on(&config, &bytes, &input, true).unwrap();
    for arm in ["vllm", "sglang"] {
        assert!(on.cells.iter().any(|cell| cell["arm"] == arm));
    }
    input.required_comparisons = vec!["vllm".into(), "sglang".into()];
    assert!(super::super::competitive_roster::select_on(&config, &bytes, &input, false).is_err());
    input.platform = "metal".into();
    assert!(super::super::competitive_roster::select_on(&config, &bytes, &input, true).is_err());
    root.close().unwrap();
}
