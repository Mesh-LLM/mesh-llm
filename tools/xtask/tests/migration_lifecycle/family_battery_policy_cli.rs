use serde_json::Value;
use std::{fs, path::PathBuf, process::Command};

#[test]
fn direct_default_one_shard_and_family_filter_survive_admission() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let manifest = root.join("ci/llama-canary/family-certified.json");
    let plan = directory.path().join("direct plan.json");
    for filter in [None, Some("qwen3-dense")] {
        let mut planner = Command::new(env!("CARGO_BIN_EXE_xtask"));
        planner
            .current_dir(&root)
            .args(["ci", "family-plan", "--manifest"])
            .arg(&manifest)
            .arg("--output")
            .arg(&plan);
        if let Some(filter) = filter {
            planner.args(["--families", filter]);
        }
        let output = planner.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let parsed: Value = serde_json::from_slice(&fs::read(&plan).unwrap()).unwrap();
        assert_eq!(parsed["shards"].as_array().unwrap().len(), 1);
        if filter.is_some() {
            assert_eq!(parsed["selected_models"].as_array().unwrap().len(), 1);
            assert_eq!(parsed["selected_models"][0]["family"], "qwen3-dense");
        }
        for (shard, success) in [("", true), ("0", true), ("1", false)] {
            let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
                .current_dir(&root)
                .args(["automation", "family-battery-policy"])
                .arg(&root)
                .arg(&manifest)
                .arg(&plan)
                .arg(shard)
                .output()
                .unwrap();
            assert_eq!(
                output.status.success(),
                success,
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

fn executable(path: &std::path::Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn copied_battery(root: &std::path::Path) -> PathBuf {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scripts = root.join("scripts");
    fs::create_dir_all(&scripts).unwrap();
    let battery = scripts.join("skippy-family-battery.sh");
    fs::copy(
        repository.join("scripts/skippy-family-battery.sh"),
        &battery,
    )
    .unwrap();
    executable(
        &scripts.join("plan-family-battery.py"),
        "#!/bin/sh\necho forbidden-planner >&2\nexit 91\n",
    );
    let manifest = root.join("ci/llama-canary/family-certified.json");
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    fs::write(manifest, "manifest bytes").unwrap();
    fs::write(root.join("supplied plan.json"), "supplied immutable bytes").unwrap();
    let bin = root.join("bin");
    fs::create_dir(&bin).unwrap();
    for tool in ["jq", "python3", "cargo"] {
        executable(
            &bin.join(tool),
            "#!/bin/sh\necho forbidden-generic-fallback >&2\nexit 92\n",
        );
    }
    executable(
        &bin.join("just"),
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\nexit 17\n",
    );
    battery
}

fn invoke_battery(
    root: &std::path::Path,
    configured: Option<&std::ffi::OsStr>,
) -> std::process::Output {
    let mut command = Command::new("/bin/bash");
    command
        .arg(root.join("scripts/skippy-family-battery.sh"))
        .args(["--dry-run", "--skip-build", "--plan"])
        .arg(root.join("supplied plan.json"))
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", root.join("bin").display()),
        )
        .env("FAMILY_BATTERY_ARTIFACT_ROOT", root.join("artifacts"))
        .env("FAMILY_BATTERY_RUN_ID", "fixture")
        .env_remove("HF_CACHE")
        .env_remove("SKIPPY_WORKLOAD_PRODUCER_MANIFEST")
        .env_remove("MESH_LLM_AUTOMATION_BIN");
    if let Some(configured) = configured {
        command.env("MESH_LLM_AUTOMATION_BIN", configured);
    }
    command.output().unwrap()
}

#[test]
fn current_battery_propagates_rust_rejection_without_legacy_fallback() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    copied_battery(&root);
    let owner = root.join("admission owner");
    executable(&owner, "#!/bin/sh\nprintf '%s\\n' \"$@\"\nexit 17\n");
    let output = invoke_battery(&root, Some(owner.as_os_str()));
    assert_eq!(output.status.code(), Some(17));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!(
            "automation\nfamily-battery-policy\n{}\n{}\n{}\n\n",
            root.display(),
            root.join("ci/llama-canary/family-certified.json").display(),
            root.join("artifacts/fixture/policy-plan.json").display()
        )
    );
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(root.join("artifacts/fixture/policy-plan.json")).unwrap(),
        b"supplied immutable bytes"
    );
}

#[test]
fn standalone_current_battery_uses_just_facade_for_typed_admission() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    copied_battery(&root);
    let output = invoke_battery(&root, None);
    assert_eq!(output.status.code(), Some(17));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!(
            "--justfile\n{}\nautomation-run\nautomation\nfamily-battery-policy\n{}\n{}\n{}\n\n",
            root.join("Justfile").display(),
            root.display(),
            root.join("ci/llama-canary/family-certified.json").display(),
            root.join("artifacts/fixture/policy-plan.json").display()
        )
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn battery_invalid_configured_controller_fails_before_artifact_or_planner_work() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    copied_battery(&root);
    let directory_owner = root.join("directory owner");
    fs::create_dir(&directory_owner).unwrap();
    let non_executable = root.join("not executable");
    fs::write(&non_executable, "not executable").unwrap();
    for value in [
        std::ffi::OsString::from(""),
        std::ffi::OsString::from("relative"),
        root.join("missing").into_os_string(),
        directory_owner.into_os_string(),
        non_executable.into_os_string(),
    ] {
        let output = invoke_battery(&root, Some(&value));
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_eq!(
            output.stderr,
            b"MESH_LLM_AUTOMATION_BIN must be an absolute executable\n"
        );
        assert!(!root.join("artifacts").exists());
    }
}
