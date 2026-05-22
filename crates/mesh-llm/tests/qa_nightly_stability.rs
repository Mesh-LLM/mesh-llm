use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("mesh-llm crate should live two levels below repo root")
        .to_path_buf()
}

fn harness_path() -> PathBuf {
    repo_root().join("scripts").join("qa-nightly-stability.py")
}

fn workflow_path(name: &str) -> PathBuf {
    repo_root().join(".github").join("workflows").join(name)
}

fn unique_output_dir(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "mesh-llm-{test_name}-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn nightly_stability_harness_exists_and_is_executable() {
    let path = harness_path();
    assert!(
        path.is_file(),
        "nightly stability harness is missing at {}",
        path.display()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = std::fs::metadata(&path)
            .expect("harness metadata should be readable")
            .permissions()
            .mode();
        assert_ne!(mode & 0o111, 0, "harness should be executable");
    }
}

#[test]
fn nightly_stability_wrapper_calls_reusable_workflow() {
    let wrapper = std::fs::read_to_string(workflow_path("nightly-stability.yml"))
        .expect("nightly stability workflow should be readable");
    assert!(
        wrapper.contains("uses: ./.github/workflows/nightly-stability-run.yml"),
        "scheduled/manual wrapper should delegate execution to the reusable workflow"
    );
    for expected in [
        "workflow_dispatch:",
        "base_url:",
        "models:",
        "attempts:",
        "agent_smokes:",
        "skip_streaming:",
        "timeout:",
        "runs_on:",
        "output_dir:",
    ] {
        assert!(
            wrapper.contains(expected),
            "wrapper should expose {expected:?}; workflow was:\n{wrapper}"
        );
    }
}

#[test]
fn nightly_stability_reusable_workflow_owns_execution() {
    let reusable = std::fs::read_to_string(workflow_path("nightly-stability-run.yml"))
        .expect("reusable nightly stability workflow should be readable");
    for expected in [
        "workflow_call:",
        "runs-on: ${{ fromJson(inputs.runs_on) }}",
        "scripts/qa-nightly-stability.py",
        "Publish run summary",
        "$GITHUB_STEP_SUMMARY",
        "actions/upload-artifact@v6",
    ] {
        assert!(
            reusable.contains(expected),
            "reusable workflow should contain {expected:?}; workflow was:\n{reusable}"
        );
    }
}

#[test]
fn nightly_stability_harness_print_plan_is_json_and_side_effect_free() {
    let output_dir = unique_output_dir("nightly-stability-print-plan");
    let output = Command::new(harness_path())
        .arg("--base-url")
        .arg("http://127.0.0.1:9337")
        .arg("--models")
        .arg("auto,mesh")
        .arg("--attempts")
        .arg("2")
        .arg("--agent-smokes")
        .arg("opencode,pi,goose")
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--print-plan")
        .output()
        .expect("harness print-plan should execute");

    assert!(
        output.status.success(),
        "print-plan failed: status={:?}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("print-plan should emit valid JSON");
    assert_eq!(
        plan.get("name").and_then(serde_json::Value::as_str),
        Some("nightly-stability")
    );
    assert_eq!(
        plan.get("endpoint").and_then(serde_json::Value::as_str),
        Some("http://127.0.0.1:9337/v1")
    );
    let steps = plan
        .get("steps")
        .and_then(serde_json::Value::as_array)
        .expect("print-plan should include steps");
    assert_eq!(
        steps.len(),
        5,
        "OpenAI surface probe, tool-call probe, and three agent smokes"
    );
    assert!(
        !output_dir.exists(),
        "print-plan must not create evidence directories"
    );
}

#[test]
fn nightly_stability_harness_help_documents_contract() {
    let output = Command::new(harness_path())
        .arg("--help")
        .output()
        .expect("harness help should execute");

    assert!(
        output.status.success(),
        "harness --help failed: status={:?}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in [
        "--base-url",
        "--models",
        "--attempts",
        "--agent-smokes",
        "--output-dir",
        "--print-plan",
        "commands.jsonl",
        "results.jsonl",
        "summary.json",
    ] {
        assert!(
            stdout.contains(expected),
            "harness help should document {expected:?}; stdout was:\n{stdout}"
        );
    }
}

#[test]
fn nightly_stability_harness_rejects_unknown_agent_smokes() {
    let output = Command::new(harness_path())
        .arg("--agent-smokes")
        .arg("unknown")
        .arg("--print-plan")
        .output()
        .expect("harness argument validation should execute");

    assert!(!output.status.success(), "unknown agent smoke should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown agent smoke"),
        "unknown agent smoke error should be explicit; stderr was:\n{stderr}"
    );
}
