use std::process::Command;

#[test]
fn manifest_preflight_checks_disjoint_whole_session_capacity_before_output() {
    let state = tempfile::tempdir().unwrap();
    let manifest = state.path().join("manifest.json");
    let requirements = state.path().join("requirements.json");
    let output = state.path().join("preflight.json");
    let trajectory = |session: &str| {
        serde_json::json!({
            "session_id":session,"source_dataset":"fixture","agent_framework":"goose","recorded_model":null,
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]
        })
    };
    let mut document = serde_json::json!({"cohorts":{"warmup":[trajectory("warmup")],"1":[trajectory("first"),trajectory("second")]}});
    std::fs::write(&requirements,br#"{"concurrency":[1],"minimum_worker_waves":2,"warmup_turns":1,"required_frameworks":["goose"]}"#).unwrap();
    std::fs::write(&manifest, serde_json::to_vec(&document).unwrap()).unwrap();
    let run = |output: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args([
                "automation",
                "replay-matrix",
                "manifest-preflight",
                "--manifest",
            ])
            .arg(&manifest)
            .arg("--requirements")
            .arg(&requirements)
            .arg("--output")
            .arg(output)
            .output()
            .unwrap()
    };
    let result = run(&output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(report["cohorts"]["1"]["trajectory_count"], 2);
    assert_eq!(report["cohorts"]["1"]["assistant_turns"], 2);
    document["cohorts"]["1"][1]["session_id"] = "warmup".into();
    std::fs::write(&manifest, serde_json::to_vec(&document).unwrap()).unwrap();
    let denied = state.path().join("denied.json");
    assert!(!run(&denied).status.success());
    assert!(!denied.exists());
    document["cohorts"]["1"] = serde_json::json!([trajectory("first")]);
    std::fs::write(&manifest, serde_json::to_vec(&document).unwrap()).unwrap();
    assert!(!run(&denied).status.success());
    assert!(!denied.exists());
}
