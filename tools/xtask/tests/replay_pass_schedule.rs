use std::process::Command;

#[test]
fn resume_cli_preserves_alternating_order_and_rejects_changed_builds() {
    let state = tempfile::tempdir().unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("schedule.json");
    let snapshot = serde_json::json!({
        "plan_sha256":"plan", "manifest_sha256":"manifest", "completed":[],
        "builds":{
            "baseline":{"engine":"mesh","commit":"base","binary_sha256":"binary-a","runtime_sha256":"runtime-a"},
            "candidate":{"engine":"mesh","commit":"head","binary_sha256":"binary-b","runtime_sha256":"runtime-b"}
        }
    });
    let mut document = serde_json::json!({
        "passes":2,"labels":["baseline","candidate"],"current":snapshot,"previous":snapshot
    });
    document["previous"]["completed"] = serde_json::json!([{"pass":1,"label":"baseline"}]);
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "pass-schedule", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&output).unwrap();
    let schedule: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        schedule,
        serde_json::json!([
            {"pass":1,"label":"candidate"},
            {"pass":2,"label":"candidate"},
            {"pass":2,"label":"baseline"}
        ])
    );
    document["previous"]["builds"]["candidate"]["runtime_sha256"] = "changed".into();
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    assert!(!run().status.success());
    assert_eq!(std::fs::read(output).unwrap(), bytes);
}
