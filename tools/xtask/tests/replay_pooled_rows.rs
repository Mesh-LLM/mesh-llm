#[test]
fn pooling_weights_tokens_and_suppresses_deltas_for_changed_outputs() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("passes.json");
    let output = state.path().join("rows.json");
    let cell = |tokens: u64, seconds: u64, digest: &str| {
        serde_json::json!({
            "concurrency":1,"trajectories":1,"requests":1,"successful_requests":1,
            "successful_request_ids":["s:0"],"failed_request_ids":[],"content_sha256_by_request":{"s:0":digest},
            "prompt_tokens_min":40,"prompt_tokens_max":40,"completion_tokens":tokens,"prompt_tokens":40,"cached_tokens":30,
            "generation_seconds":seconds,"workload_window_seconds":seconds,"ttft_samples":[1,9],
            "decode_tokens_per_second":tokens/seconds,"agent_steps_per_second":1,"workload_output_tokens_per_second":tokens/seconds,
            "ttft_p50_seconds":5,"ttft_p95_seconds":9,"mean_in_flight":1,"budget_exhausted_requests":0
        })
    };
    let mut passes = serde_json::json!([
        {"label":"base","ref":"base-ref","commit":"base","cells":[cell(10,1,"same")]},
        {"label":"base","ref":"base-ref","commit":"base","cells":[cell(90,9,"same")]},
        {"label":"candidate","ref":"head","commit":"head","cells":[cell(20,1,"same")]},
        {"label":"candidate","ref":"head","commit":"head","cells":[cell(180,9,"same")]}
    ]);
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "pooled-rows", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
    };
    std::fs::write(&input, serde_json::to_vec(&passes).unwrap()).unwrap();
    assert!(run().status.success());
    let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(rows[0]["decode_tokens_per_second"], 10.0);
    assert_eq!(rows[0]["ttft_p50_seconds"], 1.0);
    assert_eq!(rows[1]["decode_tokens_per_second_delta_pct"], 100.0);
    passes[3]["cells"][0]["content_sha256_by_request"]["s:0"] = "different".into();
    std::fs::write(&input, serde_json::to_vec(&passes).unwrap()).unwrap();
    assert!(run().status.success());
    let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(rows[1]["delta_comparable"], false);
    assert!(rows[1]["decode_tokens_per_second_delta_pct"].is_null());
}
