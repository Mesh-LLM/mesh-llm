//! Actual typed build dispatch through private controller/selected roots, with no native work.
use crate::process;
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

fn invoke(executable: &str, root: &Path, args: &[&str]) -> process::ProcessReport {
    let spec = process::ProcessSpec {
        executable: executable.into(),
        cwd: root.into(),
        arguments: args
            .iter()
            .map(|s| process::Value::Public((*s).into()))
            .collect(),
        environment: BTreeMap::from([(
            "PATH".into(),
            process::Value::Public("/usr/bin:/bin:/opt/homebrew/bin".into()),
        )]),
    };
    let result = process::supervise(
        &spec,
        &process::Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::OutputFiles::default(),
    )
    .unwrap();
    assert!(result.cleanup.complete);
    result
}
fn git(root: &Path, args: &[&str]) -> String {
    let executable = std::env::var("MIGRATION_TEST_GIT")
        .expect("set MIGRATION_TEST_GIT to an absolute Git executable");
    let result = invoke(&executable, root, args);
    assert!(result.success(), "{result:?}");
    String::from_utf8(result.stdout.bytes_retained)
        .unwrap()
        .trim()
        .to_owned()
}
fn checkout(root: &Path) -> String {
    fs::create_dir(root).unwrap();
    git(root, &["init", "--quiet"]);
    git(
        root,
        &[
            "-c",
            "user.name=Finite Dispatch",
            "-c",
            "user.email=finite-dispatch@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            root.file_name().unwrap().to_str().unwrap(),
        ],
    );
    git(root, &["rev-parse", "HEAD"])
}
#[test]
fn actual_build_dispatch_uses_frozen_controller_wrapper_and_preserves_selected_source() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let controller = root.join("controller");
    let selected = root.join("selected");
    let controller_revision = checkout(&controller);
    let selected_revision = checkout(&selected);
    assert_ne!(controller_revision, selected_revision);
    fs::create_dir_all(controller.join("scripts")).unwrap();
    fs::create_dir_all(selected.join("scripts")).unwrap();
    fs::create_dir_all(selected.join("ci/llama-canary")).unwrap();
    fs::copy(
        super::support::repository()
            .join("tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json"),
        selected.join("ci/llama-canary/family-certified.json"),
    )
    .unwrap();
    // The actual typed source-plan owner invokes this finite selected battery.
    // It still must deliver an existing plan and the frozen controller binary.
    fs::write(selected.join("scripts/skippy-family-battery.sh"),"set -eu\ntest \"$1\" = --skip-build\ntest \"$2\" = --dry-run\ntest \"$3\" = --plan\ntest -f \"$4\"\ntest -n \"$MESH_LLM_AUTOMATION_BIN\"\nprintf '%s\\n' admitted > battery-admitted\n").unwrap();
    fs::write(
        selected.join("scripts/llama-canary-agent-repair.sh"),
        "printf untrusted > selected-wrapper-ran\nexit 0\n",
    )
    .unwrap();
    let wrapper = controller.join("scripts/llama-canary-agent-repair.sh");
    fs::write(&wrapper,"set -eu\nprintf '%s\\n' \"$PWD\" \"$CANARY_SOURCE_ROOT\" \"$CANARY_CONTROLLER_SHA\" \"$CANARY_MESH_SOURCE\" \"$CANARY_HARNESS_MODE\" \"$GITHUB_RUN_ATTEMPT\" > wrapper-context\nexit 73\n").unwrap();
    let input = json!({"controller_root":controller,"source_root":selected,"controller_revision":controller_revision,"selected_revision":selected_revision,"mesh_source":selected_revision,"upstream_revision":"a".repeat(40),"mode":"pinned-build","pass_id":"repair-1","run_id":"123","run_attempt":"3","previous":null,"evidence":root.join("evidence"),"export":root.join("export"),"agent_timeout_seconds":1,"verification_timeout_seconds":1});
    let path = root.join("request.json");
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let result = invoke(
        env!("CARGO_BIN_EXE_xtask"),
        &root,
        &[
            "automation",
            "canary-receipts",
            "build",
            "--input",
            path.to_str().unwrap(),
        ],
    );
    assert!(!result.success());
    assert!(
        String::from_utf8_lossy(&result.stderr.bytes_retained).contains("73"),
        "{result:?}"
    );
    assert_eq!(
        fs::read_to_string(controller.join("wrapper-context")).unwrap(),
        format!(
            "{}\n{}\n{}\n{}\npinned-build\n3\n",
            controller.display(),
            selected.display(),
            controller_revision,
            selected_revision
        )
    );
    assert_eq!(
        fs::read(selected.join("battery-admitted")).unwrap(),
        b"admitted\n"
    );
    assert!(!selected.join("selected-wrapper-ran").exists());
    assert!(!controller.join("selected-wrapper-ran").exists());
    assert!(!root.join("export").exists());
    // The same private sources refuse repair before either finite boundary runs.
    fs::remove_file(controller.join("wrapper-context")).unwrap();
    fs::remove_file(selected.join("battery-admitted")).unwrap();
    let mut repair = input;
    repair["mode"] = "repair-build".into();
    repair["evidence"] = json!(root.join("repair-evidence"));
    fs::write(&path, serde_json::to_vec(&repair).unwrap()).unwrap();
    let result = invoke(
        env!("CARGO_BIN_EXE_xtask"),
        &root,
        &[
            "automation",
            "canary-receipts",
            "build",
            "--input",
            path.to_str().unwrap(),
        ],
    );
    assert!(!result.success());
    assert!(!controller.join("wrapper-context").exists());
    assert!(!selected.join("battery-admitted").exists());
    assert!(!root.join("repair-evidence").exists());
}
