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

#[test]
fn current_battery_propagates_rust_rejection_without_legacy_fallback() {
    use std::os::unix::fs::PermissionsExt;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = fs::read_to_string(root.join("scripts/skippy-family-battery.sh")).unwrap();
    let function = source
        .split("prepare_policy_plan() {")
        .nth(1)
        .unwrap()
        .split("\nprepare_policy_plan\n")
        .next()
        .unwrap();
    assert!(function.contains("automation family-battery-policy"));
    let directory = tempfile::tempdir().unwrap();
    let owner = directory.path().join("admission owner");
    let planner = directory.path().join("legacy planner");
    let plan = directory.path().join("supplied plan.json");
    fs::write(&plan, "supplied immutable bytes").unwrap();
    fs::write(&owner, "#!/bin/bash\nprintf '%s\\n' \"$@\"\nexit 17\n").unwrap();
    fs::write(
        &planner,
        "#!/bin/bash\necho forbidden-legacy-fallback >&2\nexit 91\n",
    )
    .unwrap();
    for path in [&owner, &planner] {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let shell = format!("prepare_policy_plan() {{{function}\nprepare_policy_plan\n");
    let output = Command::new("/bin/bash")
        .args(["-c", &shell])
        .env("PLANNER", &planner)
        .env("MESH_LLM_AUTOMATION_BIN", &owner)
        .env("POLICY_PLAN", &plan)
        .env("POLICY_PLAN_COPY", &plan)
        .env("ROOT", "root with spaces")
        .env("MANIFEST", "manifest with spaces")
        .env("SHARD_INDEX", "")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(17));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!(
            "automation\nfamily-battery-policy\nroot with spaces\nmanifest with spaces\n{}\n\n",
            plan.display()
        )
    );
    assert!(output.stderr.is_empty());
    assert_eq!(
        fs::read_to_string(plan).unwrap(),
        "supplied immutable bytes"
    );
}
