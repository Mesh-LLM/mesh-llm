#[test]
fn arm_pass_runs_disjoint_manifest_cohorts_and_retains_complete_result() {
    let state = tempfile::tempdir().unwrap();
    let reservation = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = reservation.local_addr().unwrap().port();
    let trajectory = |session: &str| {
        serde_json::json!({
            "session_id":session,"source_dataset":"fixture","agent_framework":"goose","recorded_model":null,
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]
        })
    };
    let manifest = state.path().join("manifest.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({"cohorts":{
            "warmup":[trajectory("warmup")],"1":[trajectory("first"),trajectory("second")]
        }}))
        .unwrap(),
    )
    .unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("pass.json");
    std::fs::write(&input,serde_json::to_vec(&serde_json::json!({
        "manifest":manifest,"requirements":{"concurrency":[1],"minimum_worker_waves":2,"warmup_turns":1,"required_frameworks":["goose"]},
        "binary":env!("CARGO_BIN_EXE_laya-product-fixture"),"native_runtime_root":state.path(),
        "model":"fixture","label":"candidate","ref":"head","commit":"fixture","pass":1,
        "max_output_tokens":2048,"request_timeout_seconds":2,"startup_timeout_seconds":3,"timeout_seconds":10,
        "port":port,"output":state.path().join("data/pass-1/candidate")
    })).unwrap()).unwrap();
    drop(reservation);
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "arm-pass", "--input"])
        .arg(input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let pass: serde_json::Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(pass["passed"], true);
    assert_eq!(pass["cells"][0]["requests"], 2);
    assert_eq!(pass["cells"][0]["completeness"]["passed"], true);
    assert_eq!(pass["warmup"]["acceptance"]["expected_turns"], 1);
    assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
}
