//! Exercise the standalone command's output streams without loading a model.
#[test]
fn example_config_is_one_json_document_on_stdout() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .arg("example-config")
        .output()
        .expect("start standalone command");
    assert!(output.status.success(), "{:?}", output);
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .lines()
            .all(|line| line.starts_with("⚠ ")),
        "{:?}",
        output.stderr
    );
    let config: skippy_protocol::StageConfig =
        serde_json::from_slice(&output.stdout).expect("stdout contains only stage config JSON");
    assert!(!config.stage_id.is_empty());
}

#[cfg(feature = "dynamic-native-runtime")]
#[test]
fn standalone_rejects_missing_runtime_before_reading_stage_config() {
    let temp = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env_remove("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
        .args([
            "--output",
            "human",
            "--runtime-release",
            "999.999.999-test",
            "--runtime-cache",
        ])
        .arg(temp.path())
        .args(["serve", "--config", "/nonexistent/skippy-stage.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("no compatible local Skippy runtime"),
        "{error}"
    );
    assert!(!error.contains("load stage config"), "{error}");
    assert!(std::fs::read_dir(temp.path()).unwrap().next().is_none());
}

#[test]
fn standalone_selection_uses_verified_bundle_and_rejects_abi_mismatch() {
    use skippy_api::native_runtime::{NativeRuntimeOptions, local_native_runtime_plan};
    let temp = tempfile::tempdir().unwrap();
    let bundle = temp.path().join("runtime");
    std::fs::create_dir_all(bundle.join("lib")).unwrap();
    std::fs::write(bundle.join("lib/runtime.bin"), b"fixture, never executed").unwrap();
    let profile = skippy_runtime_install::host_runtime_profile();
    let mut manifest: skippy_native_runtime::NativeRuntimeManifest = serde_json::from_value(
        serde_json::json!({"schema_version": 2, "runtime": {
            "id": "standalone-test-runtime", "release_version": "999.999.999-test",
            "skippy_abi": skippy_runtime_install::current_skippy_abi_version(),
            "platform": {"os": profile.os, "arch": profile.arch, "min_glibc": profile.glibc_version},
            "backend": {"kind": "cpu"}, "libraries": ["lib/runtime.bin"]
        }})
    ).unwrap();
    manifest.write_to_dir(&bundle).unwrap();
    let args = NativeRuntimeOptions {
        bundle_dirs: vec![bundle.clone()],
        cache_dir: Some(temp.path().join("empty-cache")),
        release: Some("999.999.999-test".into()),
        selection: Some("cpu".into()),
    };
    let plan = local_native_runtime_plan(&args).unwrap();
    assert_eq!(
        plan.root.canonicalize().unwrap(),
        bundle.canonicalize().unwrap()
    );
    assert_eq!(plan.native_runtime_id, "standalone-test-runtime");
    assert_eq!(plan.libraries.len(), 1);
    #[cfg(feature = "dynamic-native-runtime")]
    {
        // A digest-valid non-library reaches the loader, then fails before model access.
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
            .arg("--runtime-bundle")
            .arg(&bundle)
            .env_remove("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
            .args([
                "--output",
                "human",
                "--runtime-release",
                "999.999.999-test",
                "--runtime-cache",
            ])
            .arg(temp.path().join("empty-cache"))
            .args(["serve", "--config", "/nonexistent/skippy-stage.json"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("load Skippy runtime standalone-test-runtime"),
            "{error}"
        );
        assert!(!error.contains("load stage config"), "{error}");
    }
    manifest.runtime.skippy_abi = "0.0.0".into();
    manifest.write_to_dir(&bundle).unwrap();
    assert!(
        local_native_runtime_plan(&args)
            .unwrap_err()
            .to_string()
            .contains("no compatible local Skippy runtime")
    );
}

#[test]
fn legacy_import_is_not_a_subcommand() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["runtime", "import-legacy"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"));
}

#[test]
fn old_serve_commands_are_not_subcommands() {
    for command in ["serve-openai", "serve-binary"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
            .arg(command)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"));
    }
}

#[test]
fn jsonl_error_is_terminal_versioned_event() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["--output", "jsonl", "serve"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stderr.is_empty());
    let lines = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert!(!lines.is_empty());
    let event: serde_json::Value = serde_json::from_slice(lines.last().unwrap()).unwrap();
    assert_eq!(event["schema_version"], 1);
    assert_eq!(event["sequence"], lines.len());
    assert_eq!(event["type"], "error");
    assert!(
        event["data"]["message"]
            .as_str()
            .unwrap()
            .contains("provide --model")
    );
}

#[test]
fn jsonl_syntax_error_is_structured_too() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["--output=jsonl", "serve", "--not-a-switch"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stderr.is_empty());
    let event: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(event["type"], "error");
    assert!(
        event["data"]["message"]
            .as_str()
            .unwrap()
            .contains("unexpected argument")
    );
}

#[test]
fn recommended_models_have_human_and_json_presentations() {
    let root = tempfile::tempdir().unwrap();
    let entries = root.path().join("meshllm-catalog/entries/test");
    std::fs::create_dir_all(&entries).unwrap();
    std::fs::write(entries.join("tiny.json"), serde_json::to_vec(&serde_json::json!({
        "schema_version": 1,
        "source_repo": "test/tiny-GGUF",
        "variants": {"tiny-Q4_K_M": {
            "source": {"repo": "test/tiny-GGUF", "revision": "main", "file": "tiny-Q4_K_M.gguf"},
            "curated": {"name": "Tiny Test", "size": "1GB", "description": "Offline fixture"}
        }}
    })).unwrap()).unwrap();
    std::fs::write(
        root.path().join("meshllm-catalog/entries/.last_refresh"),
        b"",
    )
    .unwrap();
    let human = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_HOME", root.path())
        .args(["--output", "human", "models", "recommended"])
        .output()
        .unwrap();
    assert!(human.status.success());
    assert!(String::from_utf8_lossy(&human.stdout).contains("• Tiny Test  1GB"));
    let default = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_HOME", root.path())
        .args(["models", "recommended"])
        .output()
        .unwrap();
    assert!(default.status.success());
    assert_eq!(default.stdout, human.stdout);

    let json = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_HOME", root.path())
        .args(["--output", "json", "models", "recommended"])
        .output()
        .unwrap();
    assert!(json.status.success());
    let models: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(models["source"], "catalog");
    assert_eq!(models["results"][0]["name"], "Tiny Test");
    assert_eq!(models["results"][0]["type"], "gguf");
    assert!(models["results"][0]["capabilities"].is_object());
    assert!(
        models["results"][0]["show"]
            .as_str()
            .unwrap()
            .starts_with("skippy models show ")
    );
    let flag_json = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_HOME", root.path())
        .args(["models", "recommended", "--json"])
        .output()
        .unwrap();
    assert!(flag_json.status.success());
    assert_eq!(
        models,
        serde_json::from_slice::<serde_json::Value>(&flag_json.stdout).unwrap()
    );
}

#[test]
fn runtime_uses_shared_mesh_cache_override_without_loading_native_code() {
    let dir = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env(
            "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
            dir.path().join("empty"),
        )
        .args(["doctor"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report["runtime_cache"],
        dir.path().join("empty").to_string_lossy().as_ref()
    );
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[test]
fn models_list_uses_hugging_face_cache_without_native_runtime_or_network() {
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("models");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env("HF_HUB_CACHE", &cache)
        .args(["models", "installed", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["cache_dir"], cache.to_string_lossy().as_ref());
    assert_eq!(value["results"], serde_json::json!([]));
    assert!(cache.is_dir());
    let jsonl = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env("HF_HUB_CACHE", &cache)
        .args(["--output", "jsonl", "models", "installed"])
        .output()
        .unwrap();
    assert!(jsonl.status.success());
    assert!(jsonl.stderr.is_empty());
    let events = String::from_utf8(jsonl.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    let result = events.last().unwrap();
    assert_eq!(result["schema_version"], 1);
    assert_eq!(result["type"], "result");
    assert_eq!(result["data"], value);
}

#[test]
fn model_download_has_no_skippy_only_pin_flags() {
    let root = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env("HF_HUB_CACHE", root.path())
        .args(["models", "download", "org/repo", "--sha256", "bad"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unexpected argument"));
    assert!(!root.path().join("models--org--repo").exists());
}

#[test]
fn model_delete_uses_mesh_preview_first_policy() {
    let root = tempfile::tempdir().unwrap();
    let repo = root.path().join("models--org--model");
    let snapshot = repo.join("snapshots/abcdef1234567890");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs/main"), b"abcdef1234567890").unwrap();
    let model = snapshot.join("model-Q4_K_M.gguf");
    std::fs::write(&model, b"GGUF fixture, never loaded").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env("HF_HUB_CACHE", root.path())
        .args(["models", "delete", "org/model/model-Q4_K_M.gguf", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let preview: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(preview["dry_run"], true);
    assert_eq!(preview["paths"], serde_json::json!([model]));
    assert!(model.is_file());
}

#[test]
fn runtime_install_accepts_recommended_or_explicit_selection() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["runtime", "install", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("[RUNTIME]"), "{help}");
}
