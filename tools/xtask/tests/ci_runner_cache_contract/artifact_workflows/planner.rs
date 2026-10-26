//! Execute the real plan action's input and output projection around a finite
//! planner observer. The planner itself has independent native golden tests.
use super::{Node, support};
use serde_json::{Value, json};
use std::{fs, os::unix::fs::symlink, process::Command};

fn fixture() -> support::Fixture {
    let fixture = support::Fixture::new();
    for tool in ["jq", "awk", "mktemp"] {
        let executable = ["/usr/bin", "/bin", "/opt/homebrew/bin", "/usr/local/bin"]
            .into_iter()
            .map(|dir| std::path::Path::new(dir).join(tool))
            .find(|path| path.is_file())
            .unwrap();
        symlink(executable, fixture.path().join("bin").join(tool)).unwrap();
    }
    fixture.executable(
        "sha256sum",
        "[[ $# == 0 ]] || exit 91; exec /usr/bin/shasum -a 256",
    );
    fs::copy(
        support::root().join("tools/xtask/tests/fixtures/ci_plan/expected/main.plan.json"),
        fixture.path().join("golden.json"),
    )
    .unwrap();
    fixture.executable("automation",r#"[[ "$*" == "ci plan --manifest-root $GITHUB_WORKSPACE" ]] || exit 92
/bin/cat > planner-input.json
jq -e '.profile == "main" and .event_name == "push" and .changed_files == ["docs/MESHES.md"]' planner-input.json >/dev/null || exit 93
if [[ "$AFFECTED_CRATES" == '[]' ]]; then
  jq -e 'has("affected_crates") | not' planner-input.json >/dev/null || exit 94
else
  jq -e '.affected_crates == ["mesh-llm"]' planner-input.json >/dev/null || exit 95
fi
/bin/cat golden.json"#);
    fixture
}

fn project(fixture: &support::Fixture, affected: &str) -> Value {
    let action = support::action("plan-ci");
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    let plan = steps
        .iter()
        .find(|step| step.get("id").and_then(Node::text) == Some("plan"))
        .unwrap();
    let mut command = Command::new("/bin/bash");
    command
        .env_clear()
        .current_dir(fixture.path())
        .env("PATH", fixture.path().join("bin"));
    for (key, value) in [
        ("EVENT_NAME", "push"),
        ("SOURCE_SHA", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("BASE_SHA", ""),
        ("DRAFT", "false"),
        ("AFFECTED_CRATES", affected),
        ("UPLOAD_ARTIFACT", "false"),
        ("ARTIFACT_NAME", ""),
    ] {
        command.env(key, value);
    }
    command
        .env("GITHUB_WORKSPACE", support::root())
        .env("RUNNER_TEMP", fixture.path())
        .env("HOME", fixture.path())
        .env("TMPDIR", fixture.path());
    command
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"))
        .env("GITHUB_STEP_SUMMARY", fixture.path().join("summary"));
    command.env(
        "MESH_LLM_AUTOMATION_BIN",
        fixture.path().join("bin/automation"),
    );
    fs::write(fixture.path().join("changed.txt"), "docs/MESHES.md\n").unwrap();
    // Isolate only the action's fixed file transport, preserving its actual body.
    let source = plan.get("run").unwrap().text().unwrap().replace(
        "/tmp/changed_files.txt",
        &format!("\"{}\"", fixture.path().join("changed.txt").display()),
    );
    command.args(["-c", &source]);
    let output = fixture.run(command);
    assert!(output.status.success(), "{output:?}");
    fixture.outputs()
}

#[test]
fn artifact_plan_action_emits_exact_platform_matrices_and_omits_empty_affected_crates() {
    for affected in ["[]", r#"["mesh-llm"]"#] {
        let fixture = fixture();
        let outputs = project(&fixture, affected);
        let plan: Value =
            serde_json::from_slice(&fs::read(fixture.path().join("golden.json")).unwrap()).unwrap();
        for platform in ["linux", "macos", "windows"] {
            for (matrix, output) in [("hosts", "hosts"), ("runtime_products", "runtime_products")] {
                let expected = plan["matrices"][matrix]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|row| row["platform"] == platform)
                    .cloned()
                    .collect::<Vec<_>>();
                let actual: Value = serde_json::from_str(
                    outputs[format!("{platform}_{output}_matrix")]
                        .as_str()
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(actual, json!(expected));
            }
        }
        for field in [
            "linux_max_parallel",
            "macos_max_parallel",
            "windows_max_parallel",
            "total_max_workers",
        ] {
            assert_eq!(
                outputs[field].as_str().unwrap(),
                plan["budgets"][field].to_string()
            );
        }
        for (field, expected) in plan["signals"].as_object().unwrap() {
            assert_eq!(outputs[field].as_str().unwrap(), expected.to_string());
        }
    }
}

#[path = "change_transport.rs"]
mod change_transport;
#[path = "pr_manifest_intent.rs"]
mod pr_manifest_intent;
#[path = "projection_intent.rs"]
mod projection_intent;
