#[test]
fn acceptance_cli_retains_negative_reports_for_unpaired_or_regressed_rows() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("gates.json");
    let mut document = serde_json::json!({
        "rows":[{"label":"candidate","concurrency":4,"failed_requests":0,
            "prompt_tokens_min":18000,"prompt_tokens_max":22000,"cache_pct":75,
            "content_identity_known":true,"delta_comparable":true,"ttft_p50_seconds_delta_pct":4}],
        "prompt_token_range":[18000,22000],"min_cache_pct":70,"require_output_match":true,"max_ttft_regression_pct":5
    });
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "acceptance-gates", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
    };
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    assert!(run().status.success());
    document["rows"][0]["delta_comparable"] = false.into();
    document["rows"][0]["ttft_p50_seconds_delta_pct"] = 6.into();
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    assert!(!run().status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["passed"], false);
    let failed: Vec<_> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|check| check["passed"] == false)
        .map(|check| check["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        failed,
        [
            "deterministic-output:candidate/c4",
            "ttft-regression:candidate/c4"
        ]
    );
}
