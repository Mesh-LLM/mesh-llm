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
    repo_root()
        .join("scripts")
        .join("qa-agent-tool-call-reliability.py")
}

fn unique_output_path(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "mesh-llm-{test_name}-{}-{nanos}.jsonl",
        std::process::id()
    ))
}

#[test]
fn agent_tool_call_harness_exists_and_is_executable() {
    let path = harness_path();
    assert!(
        path.is_file(),
        "agent tool-call harness is missing at {}",
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
fn agent_tool_call_harness_help_documents_contract() {
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
        "--timeout",
        "--output",
        "--skip-streaming",
        "--print-plan",
        "tool-call",
        "tool-result",
    ] {
        assert!(
            stdout.contains(expected),
            "harness help should document {expected:?}; stdout was:\n{stdout}"
        );
    }
}

#[test]
fn agent_tool_call_harness_print_plan_is_json_and_side_effect_free() {
    let output_path = unique_output_path("agent-tool-call-print-plan");
    let output = Command::new(harness_path())
        .arg("--base-url")
        .arg("http://127.0.0.1:9337")
        .arg("--models")
        .arg("auto,mesh")
        .arg("--attempts")
        .arg("2")
        .arg("--output")
        .arg(&output_path)
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
        Some("agent-tool-call-reliability")
    );
    assert_eq!(
        plan.get("endpoint").and_then(serde_json::Value::as_str),
        Some("http://127.0.0.1:9337/v1")
    );
    let checks = plan
        .get("checks")
        .and_then(serde_json::Value::as_array)
        .expect("print-plan should include checks");
    assert_eq!(
        checks.len(),
        4,
        "two models x two attempts should be planned"
    );
    assert!(
        !output_path.exists(),
        "print-plan must not create result files"
    );
}

#[test]
fn agent_tool_call_harness_rejects_unknown_arguments() {
    let output = Command::new(harness_path())
        .arg("--definitely-unknown")
        .output()
        .expect("harness argument validation should execute");

    assert!(!output.status.success(), "unknown arguments should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--definitely-unknown"),
        "unknown-argument error should be explicit; stderr was:\n{stderr}"
    );
}
