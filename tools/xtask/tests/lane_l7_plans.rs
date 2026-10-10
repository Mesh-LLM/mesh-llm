use std::path::Path;

#[test]
fn logging_plan_is_side_effect_free_and_preserves_optional_endpoint_contract() {
    let root = tempfile::tempdir().unwrap();
    let evidence = root.path().join("not-created");
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    for supplied in [false, true] {
        let mut command = std::process::Command::new("bash");
        command
            .arg(repository.join("scripts/qa-logging-recovery.sh"))
            .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
            .args(["--current-binary", "/not/an/executable", "--evidence-dir"])
            .arg(&evidence)
            .arg("--print-plan");
        if supplied {
            command.args([
                "--deterministic-openai-endpoint",
                "http://127.0.0.1:9/v1",
                "--deterministic-openai-model",
                "fixture-model",
            ]);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(plan["deterministic_openai_endpoint_supplied"], supplied);
        assert_eq!(
            plan["optional_plugin_behavior"]["without_endpoint"]["logging_fail_open_inference"],
            "PREREQ"
        );
        assert_eq!(
            plan["optional_plugin_behavior"]["with_endpoint"]["logging_fail_open_inference"],
            "execute"
        );
        assert_eq!(plan["checks"].as_array().unwrap().len(), 7);
        assert_eq!(plan["evidence_files"][0], "manifest.json");
        if supplied {
            assert_eq!(plan["deterministic_openai_model"], "fixture-model");
        }
        assert!(!evidence.exists());
    }
}
