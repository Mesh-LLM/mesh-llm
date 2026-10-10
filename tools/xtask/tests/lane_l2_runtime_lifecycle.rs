#[cfg(target_os = "linux")]
use serde_json::{Value, json};
use std::{error::Error, process::Command};
#[cfg(target_os = "linux")]
use std::{fs, path::Path};

type TestResult = Result<(), Box<dyn Error>>;

#[cfg(target_os = "linux")]
#[test]
fn preflight_restore_start_finish_retains_verified_negative_measurement() -> TestResult {
    use std::os::unix::fs::PermissionsExt;
    let sandbox = tempfile::tempdir()?;
    let root = sandbox.path();
    let tools = root.join("tools");
    fs::create_dir(&tools)?;
    let key = format!(
        "mesh-llm-sccache-seed-linux-x86_64-img-f499b79b-epoch-f499b79b-v3-{}",
        "a".repeat(64)
    );
    let cache = json!({"id":123,"key":key,"version":"b".repeat(64),"ref":"refs/heads/main","size_in_bytes":1024});
    fs::write(
        root.join("listing.json"),
        serde_json::to_vec(&json!({"total_count":1,"actions_caches":[cache]}))?,
    )?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("root")?;
    let raw = fs::read(repository.join(
        "ci/runtime-seed-evidence/34272984200-1/runtime-seed-evidence-3-warm-1/raw-stats.json",
    ))?;
    fs::write(root.join("stats.json"), raw)?;
    for (name, body) in [
        ("git", "printf '%040d\\n' 0"),
        ("uname", "printf 'Linux fixture x86_64\\n'"),
        ("curl", "cat listing.json"),
        (
            "lscpu",
            r#"printf '%s' '{"lscpu":[{"field":"Architecture:","data":"x86_64"},{"field":"CPU(s):","data":"4"},{"field":"Model name:","data":"fixture"},{"field":"Vendor ID:","data":"fixture"},{"field":"Thread(s) per core:","data":"1"}]}'"#,
        ),
        (
            "sccache",
            "if [ \"$1\" != --zero-stats ]; then cat stats.json; fi",
        ),
    ] {
        let path = tools.join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n"))?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    }
    let evidence = root.join("evidence");
    let invoke = |operation: &str| -> Result<std::process::Output, std::io::Error> {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(root)
            .args(["ci-ops", "runtime-seed", operation])
            .arg(&evidence)
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", tools.display()))
            .env("GITHUB_EVENT_NAME", "workflow_dispatch")
            .env("RUNNER_ENVIRONMENT", "github-hosted")
            .env("RUNNER_ARCH", "X64")
            .env(
                "LLAMA_STAGE_BUILD_DIR",
                ".deps/llama.cpp/build-stage-abi-dynamic-cpu",
            )
            .env("RUSTC_WRAPPER", "sccache")
            .env("MESH_LLM_REQUIRE_SCCACHE", "1")
            .env("CARGO_INCREMENTAL", "0")
            .env("CACHE_NAMESPACE", "mesh-llm")
            .env("SCCACHE_GHA_ENABLED", "false")
            .env("SCCACHE_MULTILEVEL_CHAIN", "disk")
            .env("SCCACHE_CACHE_SIZE", "2G")
            .env("LLAMA_STAGE_BACKEND", "cpu")
            .env("RUNNER_TEMP", root)
            .env("GITHUB_SHA", "0".repeat(40))
            .env("GITHUB_RUN_ID", "123")
            .env("GITHUB_RUN_ATTEMPT", "1")
            .env("CANARY_PAIR", "1")
            .env("CANARY_ARM", "warm")
            .env("CANARY_KEY", &key)
            .env("GITHUB_OUTPUT", root.join("output"))
            .env("GH_TOKEN", "fixture-token")
            .env("CANARY_CACHE_HIT", "true")
            .env("CANARY_BUILD_OUTCOME", "success")
            .output()
    };
    for operation in ["preflight", "restore_start", "restored"] {
        let output = invoke(operation)?;
        assert!(
            output.status.success(),
            "{operation}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let mut raw: Value = serde_json::from_slice(&fs::read(root.join("stats.json"))?)?;
    raw["stats"]["compile_requests"] = 0.into();
    fs::write(root.join("stats.json"), serde_json::to_vec(&raw)?)?;
    assert!(invoke("start")?.status.success());
    let build = root.join(".deps/llama.cpp/build-stage-abi-dynamic-cpu");
    fs::create_dir_all(&build)?;
    fs::write(
        build.join("CMakeCache.txt"),
        "CMAKE_C_COMPILER_LAUNCHER:STRING=/usr/bin/sccache\nCMAKE_CXX_COMPILER_LAUNCHER:STRING=sccache\n",
    )?;
    fs::create_dir(root.join("runtime-input"))?;
    fs::write(root.join("runtime-input/manifest.json"), "{}")?;
    assert_eq!(invoke("finish")?.status.code(), Some(1));
    let result: Value = serde_json::from_slice(&fs::read(evidence.join("result.json"))?)?;
    assert_eq!(result["classification"], "warm-floor-failure");
    assert_eq!(result["verified"], true);
    assert_eq!(result["eligibility_changed"], false);
    assert_eq!(
        fs::read_to_string(root.join("output"))?,
        format!("key={key}\n")
    );
    Ok(())
}

#[test]
fn help_and_invalid_operation_are_visible_without_evidence() -> TestResult {
    let directory = tempfile::tempdir()?;
    let help = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci-ops", "runtime-seed", "--help"])
        .output()?;
    assert!(help.status.success());
    assert!(
        String::from_utf8(help.stdout)?
            .contains("preflight|restore_start|restored|start|finish|summarize")
    );
    let bad = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci-ops", "runtime-seed", "invalid"])
        .arg(directory.path().join("evidence"))
        .output()?;
    assert!(!bad.status.success());
    assert!(!directory.path().join("evidence").exists());
    Ok(())
}
