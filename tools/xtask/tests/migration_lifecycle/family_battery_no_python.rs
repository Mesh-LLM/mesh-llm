//! Complete current battery dry-run with actual typed plan admission and no interpreter on PATH.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    time::Duration,
};

fn native_tool(name: &str) -> PathBuf {
    let configured = (name == "jq")
        .then(|| std::env::var_os("MIGRATION_TEST_JQ"))
        .flatten();
    let path = configured.map(PathBuf::from).unwrap_or_else(|| {
        ["/usr/bin", "/bin", "/opt/homebrew/bin"]
            .iter()
            .map(|directory| Path::new(directory).join(name))
            .find(|path| path.is_file())
            .unwrap_or_else(|| panic!("native fixture tool missing: {name}"))
    });
    assert!(path.is_absolute() && path.is_file());
    path.canonicalize().unwrap()
}
fn finite_path(directory: &Path) {
    fs::create_dir(directory).unwrap();
    for name in [
        "awk", "basename", "cat", "cp", "cut", "date", "dirname", "mkdir", "paste", "sed", "tail",
        "tr", "wc", "jq",
    ] {
        symlink(native_tool(name), directory.join(name)).unwrap();
    }
    assert!(!directory.join("python3").exists());
    assert!(!directory.join("python").exists());
}
fn run(
    root: &Path,
    tools: &Path,
    artifact: &Path,
    id: &str,
    supplied: Option<&Path>,
) -> process::ProcessReport {
    let script = root.join("scripts/skippy-family-battery.sh");
    let mut arguments: Vec<std::ffi::OsString> = vec!["-c".into(), "if command -v python3 >/dev/null || command -v python >/dev/null; then exit 97; fi; exec /bin/bash \"$@\"".into(), "finite-no-python".into(), script.into(), "--skip-build".into(), "--dry-run".into()];
    if let Some(plan) = supplied {
        arguments.extend(["--plan".into(), plan.into()]);
    }
    let environment = BTreeMap::from([
        ("PATH".into(), Value::Public(tools.into())),
        (
            "MESH_LLM_AUTOMATION_BIN".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        ("FAMILY_BATTERY_RUN_ID".into(), Value::Public(id.into())),
        (
            "FAMILY_BATTERY_ARTIFACT_ROOT".into(),
            Value::Public(artifact.into()),
        ),
        (
            "FAMILY_BATTERY_BIN_DIR".into(),
            Value::Public(root.join("absent-native-bin").into()),
        ),
    ]);
    let report = process::supervise(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            arguments: arguments.into_iter().map(Value::Public).collect(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    report
}
#[test]
fn actual_battery_causal_dry_run_needs_no_python_and_rejects_tampered_plan_before_lanes() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    fs::create_dir(root.join("scripts")).unwrap();
    fs::copy(
        super::support::repository().join("scripts/skippy-family-battery.sh"),
        root.join("scripts/skippy-family-battery.sh"),
    )
    .unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("tools/xtask")).unwrap();
    fs::write(
        root.join("tools/xtask/Cargo.toml"),
        "[package]\nname='fixture'\n",
    )
    .unwrap();
    let manifest = root.join("ci/llama-canary/family-certified.json");
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    let mut policy: serde_json::Value = serde_json::from_slice(
        &fs::read(
            super::support::repository()
                .join("tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    policy["models"].as_array_mut().unwrap().truncate(1);
    policy["models"][0]["execution"]["trunk_layers"] = json!(8);
    policy["models"][0]["execution"]["activation_width"] = json!(64);
    let original = serde_json::to_vec(&policy).unwrap();
    fs::write(&manifest, &original).unwrap();
    let tools = root.join("finite-tools");
    finite_path(&tools);
    let artifacts = root.join("artifacts");
    let accepted = run(&root, &tools, &artifacts, "accepted", None);
    assert!(accepted.success(), "{accepted:?}");
    assert!(
        String::from_utf8_lossy(&accepted.stdout.bytes_retained)
            .contains("1 certifications planned; no lanes executed")
    );
    assert!(!root.join("absent-native-bin").exists());
    assert_eq!(fs::read(&manifest).unwrap(), original);
    let admitted = artifacts.join("accepted/policy-plan.json");
    let mut plan: serde_json::Value =
        serde_json::from_slice(&fs::read(&admitted).unwrap()).unwrap();
    assert_eq!(plan["selected_models"][0]["class"], "causal_generation");
    plan["selected_models"][0]["execution"]["activation_width"] = json!(65);
    let forged = root.join("forged-plan.json");
    fs::write(&forged, serde_json::to_vec(&plan).unwrap()).unwrap();
    let rejected = run(&root, &tools, &artifacts, "rejected", Some(&forged));
    assert!(!rejected.success(), "{rejected:?}");
    assert!(
        !String::from_utf8_lossy(&rejected.stdout.bytes_retained).contains("==> family-certify:")
    );
    assert!(
        fs::read_dir(artifacts.join("rejected/certifications"))
            .unwrap()
            .next()
            .is_none()
    );
    assert_eq!(fs::read(&manifest).unwrap(), original);
}
