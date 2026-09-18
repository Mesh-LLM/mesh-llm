//! Exercise the standalone command's output streams without loading a model.
#[test]
fn example_config_is_one_json_document_on_stdout() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy-server"))
        .arg("example-config")
        .output()
        .expect("start standalone command");
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty(), "{:?}", output.stderr);
    let config: skippy_protocol::StageConfig =
        serde_json::from_slice(&output.stdout).expect("stdout contains only stage config JSON");
    assert!(!config.stage_id.is_empty());
}

#[cfg(feature = "dynamic-native-runtime")]
#[test]
fn standalone_rejects_missing_runtime_before_reading_stage_config() {
    let temp = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy-server"))
        .args(["--runtime-release", "999.999.999-test", "--runtime-cache"])
        .arg(temp.path())
        .args(["serve-openai", "--config", "/nonexistent/skippy-stage.json"])
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
    use skippy_server::{cli::NativeRuntimeArgs, native_runtime::local_native_runtime_plan};
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
    let args = NativeRuntimeArgs {
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
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy-server"))
            .arg("--runtime-bundle")
            .arg(&bundle)
            .args(["--runtime-release", "999.999.999-test", "--runtime-cache"])
            .arg(temp.path().join("empty-cache"))
            .args(["serve-openai", "--config", "/nonexistent/skippy-stage.json"])
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
