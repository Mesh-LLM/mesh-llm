#[test]
fn disposable_preflight_captures_all_cohort_tokens_without_a_live_model() {
    preflight_case(1, true);
}

#[test]
fn rejected_long_prompt_coverage_retains_every_cohort() {
    preflight_case(32768, false);
}

fn preflight_case(minimum_prompt: u64, passed: bool) {
    use sha2::{Digest, Sha256};
    let state = tempfile::tempdir().unwrap();
    let reservation = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = reservation.local_addr().unwrap().port();
    let model = state.path().join("model.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", Some("fixture")),
        ("fixture.context_length", None),
    ] {
        bytes.extend(u64::try_from(key.len()).unwrap().to_le_bytes());
        bytes.extend(key.as_bytes());
        if let Some(value) = value {
            bytes.extend(8_u32.to_le_bytes());
            bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
            bytes.extend(value.as_bytes());
        } else {
            bytes.extend(4_u32.to_le_bytes());
            bytes.extend(131072_u32.to_le_bytes());
        }
    }
    let digest = hex::encode(Sha256::digest(&bytes));
    std::fs::write(&model, bytes).unwrap();
    let trajectory = |session: &str| {
        serde_json::json!({
            "session_id":session,"source_dataset":"fixture","agent_framework":"goose","recorded_model":null,
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"},
                {"role":"user","content":"next"},{"role":"assistant","content":"final"}]
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
    let output = state.path().join("eligibility.json");
    std::fs::write(&input,serde_json::to_vec(&serde_json::json!({
        "manifest":manifest,"requirements":{"concurrency":[1],"minimum_worker_waves":2,"warmup_turns":1,"required_frameworks":["goose"]},
        "binary":env!("CARGO_BIN_EXE_laya-product-fixture"),"native_runtime_root":state.path(),
        "model":model,"model_sha256":digest,"minimum_context_tokens":131072,"minimum_session_prompt_tokens":minimum_prompt,
        "max_output_tokens":2048,"request_timeout_seconds":2,"startup_timeout_seconds":3,"timeout_seconds":10,
        "port":port,"output":state.path().join("context-preflight")
    })).unwrap()).unwrap();
    drop(reservation);
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "context-preflight",
            "--input",
        ])
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success() == passed,
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["passed"], passed);
    assert_eq!(report["cohorts"]["warmup"]["passed"], true);
    assert_eq!(report["cohorts"]["1"]["passed"], passed);
    assert_eq!(report["cohorts"]["1"]["turns"].as_array().unwrap().len(), 4);
    assert_eq!(report["prompt_tokens_by_cohort"]["1"]["first:0"], 40);
    assert_eq!(report["prompt_tokens_by_cohort"]["warmup"]["warmup:0"], 40);
    assert_eq!(report["model"]["sha256"], digest);
    assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
    if passed {
        let mut arm: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
        arm["label"] = "candidate".into();
        arm["ref"] = "head".into();
        arm["commit"] = "fixture".into();
        arm["pass"] = 1.into();
        arm["output"] = serde_json::to_value(state.path().join("measured-pass")).unwrap();
        let mut tokens = report["prompt_tokens_by_cohort"].clone();
        tokens.as_object_mut().unwrap().remove("warmup");
        arm["qualification"] = serde_json::json!({"model_sha256":digest,"minimum_context_tokens":131072,
            "minimum_session_prompt_tokens":1,"require_recurrent_restores":true,"prompt_tokens_by_cohort":tokens});
        let arm_input = state.path().join("arm.json");
        std::fs::write(&arm_input, serde_json::to_vec(&arm).unwrap()).unwrap();
        let arm_output = state.path().join("pass.json");
        let invoke = |path: &std::path::Path| {
            std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
                .args(["automation", "replay-matrix", "arm-pass", "--input"])
                .arg(&arm_input)
                .arg("--output")
                .arg(path)
                .output()
                .unwrap()
        };
        let result = invoke(&arm_output);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let measured: serde_json::Value =
            serde_json::from_slice(&std::fs::read(arm_output).unwrap()).unwrap();
        assert_eq!(measured["cells"][0]["acceptance"]["passed"], true);
        assert_eq!(measured["cells"][0]["recurrent_state"]["passed"], true);
        assert_eq!(measured["model_id"], "laya-fixture");
        arm["output"] = serde_json::to_value(state.path().join("mismatch-pass")).unwrap();
        arm["qualification"]["prompt_tokens_by_cohort"]["1"]["first:0"] = 41.into();
        std::fs::write(&arm_input, serde_json::to_vec(&arm).unwrap()).unwrap();
        let denied = state.path().join("denied-pass.json");
        assert!(!invoke(&denied).status.success());
        let measured: serde_json::Value =
            serde_json::from_slice(&std::fs::read(denied).unwrap()).unwrap();
        assert_eq!(measured["passed"], false);
        assert_eq!(measured["cells"][0]["acceptance"]["passed"], false);
    }
}
