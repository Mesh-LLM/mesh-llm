use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    process::{Command, Output},
};

fn execute(model: &Path, output: &Path, layer: &str, ctx: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "openai-smoke-config", "--model-path"])
        .arg(model)
        .arg("--output")
        .arg(output)
        .args([
            "--model-id",
            "model \"雪\"\nidentity",
            "--layer-end",
            layer,
            "--ctx-size",
            ctx,
        ])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn openai_smoke_cli_hashes_real_model_and_preserves_exact_stage_shape() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model with \"雪\".gguf");
    let bytes = vec![0xa5; 1024 * 1024 + 17];
    std::fs::write(&model, &bytes).unwrap();
    let output = dir.path().join("stage.json");
    let result = execute(&model, &output, "32", "256", &[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let bytes_out = std::fs::read(&output).unwrap();
    assert!(bytes_out.ends_with(b"\n"));
    let actual: Value = serde_json::from_slice(&bytes_out).unwrap();
    assert_eq!(
        actual,
        json!({
            "run_id":"openai-smoke", "topology_id":"openai-smoke-single-stage",
            "model_id":"model \"雪\"\nidentity", "model_path":model,
            "source_model_sha256":hex::encode(Sha256::digest(&bytes)), "stage_id":"stage-0",
            "stage_index":0, "layer_start":0, "layer_end":32, "ctx_size":256,
            "n_gpu_layers":0, "load_mode":"runtime-slice", "execution_contract":"",
            "bind_addr":"127.0.0.1:19000", "upstream":null, "downstream":null, "kv_server":null
        })
    );
    // Model identity must follow admitted bytes, not a caller-supplied digest.
    std::fs::write(&model, b"changed model fixture\n").unwrap();
    let result = execute(&model, &output, "48", "8192", &[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let changed: Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(
        changed["source_model_sha256"],
        hex::encode(Sha256::digest(b"changed model fixture\n"))
    );
    assert_ne!(
        changed["source_model_sha256"],
        actual["source_model_sha256"]
    );
    assert_eq!(changed["layer_end"], 48);
    assert_eq!(changed["ctx_size"], 8192);
}

#[test]
fn invalid_openai_smoke_inputs_do_not_truncate_existing_config_or_modify_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.gguf");
    std::fs::write(&model, b"model fixture\n").unwrap();
    let output = dir.path().join("retained.json");
    for (layer, ctx, extra) in [
        ("0", "256", vec![]),
        ("-1", "256", vec![]),
        ("32", "0", vec![]),
        ("32", "true", vec![]),
        ("32", "4294967296", vec![]),
        ("32", "1.5", vec![]),
        ("32", "256", vec!["--load-mode", "capture-only"]),
        ("32", "256", vec!["--model-sha256", "unverified"]),
        ("32", "256", vec!["--model-id", ""]),
    ] {
        std::fs::write(&output, b"retained\n").unwrap();
        let result = execute(&model, &output, layer, ctx, &extra);
        assert!(!result.status.success());
        assert_eq!(std::fs::read(&output).unwrap(), b"retained\n");
        assert_eq!(std::fs::read(&model).unwrap(), b"model fixture\n");
    }
    let result = execute(&dir.path().join("missing.gguf"), &output, "32", "256", &[]);
    assert!(!result.status.success());
    assert_eq!(std::fs::read(output).unwrap(), b"retained\n");
}

#[test]
fn openai_smoke_cli_requires_complete_inputs_and_help_has_no_file_side_effect() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "openai-smoke-config", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("--ctx-size N")
    );
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "openai-smoke-config"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[cfg(unix)]
#[test]
fn smoke_wrapper_selects_absolute_automation_and_rejects_invalid_override_without_fallback() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let owner = dir.path().join("automation fixture");
    let log = dir.path().join("selected-owner.txt");
    let fallback_log = dir.path().join("fallback.txt");
    let cargo_log = dir.path().join("cargo-must-not-run.txt");
    for (path, marker) in [
        (&owner, &log),
        (&dir.path().join("just"), &fallback_log),
        (&dir.path().join("cargo"), &cargo_log),
    ] {
        std::fs::write(
            path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nexit 17\n",
                marker.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/skippy-openai-smoke.sh");
    let nonexecutable = dir.path().join("nonexecutable");
    std::fs::write(&nonexecutable, b"not executable\n").unwrap();
    for configured in [
        std::ffi::OsString::from(""),
        "relative-tool".into(),
        "/nonexistent/automation".into(),
        nonexecutable.into_os_string(),
        dir.path().as_os_str().to_owned(),
    ] {
        let output = Command::new("/bin/bash")
            .arg(&script)
            .env("MESH_LLM_AUTOMATION_BIN", configured)
            .env("PATH", format!("{}:/usr/bin:/bin", dir.path().display()))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("must be an absolute executable"));
        assert!(!log.exists());
        assert!(!fallback_log.exists());
        assert!(!cargo_log.exists());
    }
    let output = Command::new("/bin/bash")
        .arg(&script)
        .env("MESH_LLM_AUTOMATION_BIN", &owner)
        .env("PATH", format!("{}:/usr/bin:/bin", dir.path().display()))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(17));
    assert!(
        std::fs::read_to_string(&log)
            .unwrap()
            .starts_with("models\nresolve\n")
    );
    assert!(!fallback_log.exists());
    let output = Command::new("/bin/bash")
        .arg(&script)
        .env_remove("MESH_LLM_AUTOMATION_BIN")
        .env("PATH", format!("{}:/usr/bin:/bin", dir.path().display()))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(17));
    assert!(
        std::fs::read_to_string(fallback_log)
            .unwrap()
            .contains("automation-run\nmodels\nresolve\n")
    );
    assert!(!cargo_log.exists());
}
