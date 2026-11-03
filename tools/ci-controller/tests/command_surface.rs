use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn run(args: &[&str], input: &[u8]) -> Result<Output, Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child = Command::new(env!("CARGO_BIN_EXE_ci-controller"))
        .current_dir(root)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.take().ok_or("stdin")?.write_all(input)?;
    Ok(child.wait_with_output()?)
}

#[test]
fn plan_rejects_malformed_json_when_input_is_invalid() -> TestResult {
    // Given: invalid JSON on stdin.
    let input = b"not json";
    // When: the narrow planner handles the plan command.
    let output = run(&["ci", "plan"], input)?;
    // Then: the existing planner refusal status and stream split survive.
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(
        output
            .stderr
            .starts_with(b"ERROR: unable to build CI plan: ")
    );
    Ok(())
}

#[test]
fn exhaustive_plans_keep_coverage_while_restricting_actual_policy_changes() -> TestResult {
    let plan = |profile: &str,
                event: &str,
                base: &str,
                path: &str|
     -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let request = serde_json::json!({
            "profile": profile,
            "event_name": event,
            "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "base_sha": base,
            "changed_files": [path],
            "workspace_packages": [{"name": "mesh-llm", "path": "mesh/crates/mesh-llm"}]
        });
        let result = run(&["ci", "plan"], &serde_json::to_vec(&request)?)?;
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        Ok(serde_json::from_slice(&result.stdout)?)
    };
    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let main = plan("main", "push", base, "README.md")?;
    let manual = plan("manual-full", "workflow_dispatch", base, "README.md")?;
    let controlled = plan("main", "push", base, ".github/workflows/main_linux.yml")?;
    let fallback = plan("main", "push", "", "__force_all__")?;
    for value in [&main, &manual, &controlled, &fallback] {
        assert_eq!(value["signals"]["runner_contract_required"], true);
        assert_eq!(value["required_slices"], main["required_slices"]);
        assert_eq!(value["matrices"], main["matrices"]);
    }
    assert_eq!(main["signals"]["force_hosted"], false);
    assert_eq!(manual["signals"]["force_hosted"], false);
    assert_eq!(controlled["signals"]["force_hosted"], true);
    assert_eq!(fallback["signals"]["force_hosted"], true);
    Ok(())
}

#[test]
fn force_all_rejects_partial_pr_crate_coverage() -> TestResult {
    let request = serde_json::json!({
        "profile": "pr-ready",
        "event_name": "pull_request",
        "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "base_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "changed_files": ["__force_all__"],
        "workspace_packages": [
            {"name": "mesh-llm", "path": "mesh/crates/mesh-llm"},
            {"name": "mesh-llm-cli", "path": "mesh/crates/mesh-llm-cli"}
        ],
        "affected_crates": ["mesh-llm"]
    });
    let result = run(&["ci", "plan"], &serde_json::to_vec(&request)?)?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&result.stdout)?;
    assert_eq!(
        plan["affected_crates"],
        serde_json::json!(["mesh-llm", "mesh-llm-cli"])
    );
    let tested = plan["matrices"]["rust_tests"]
        .as_array()
        .ok_or("test batches")?;
    for name in ["mesh-llm", "mesh-llm-cli"] {
        assert!(tested.iter().any(|batch| {
            batch["crates"]
                .as_array()
                .is_some_and(|crates| crates.iter().any(|crate_name| crate_name == name))
        }));
    }
    assert_eq!(plan["signals"]["force_hosted"], true);
    Ok(())
}

#[test]
fn lane_succeeds_when_unselected_jobs_are_skipped() -> TestResult {
    // Given: an unselected website lane with its only job skipped.
    let plan = r#"{"lane":"website","required":false,"required_slices":[]}"#;
    let needs = r#"{"web":{"result":"skipped"}}"#;
    // When: lane admission runs through the executable.
    let output = run(
        &["ci", "validate-lane", "--lane-plan", plan, "--needs", needs],
        b"",
    )?;
    // Then: admission succeeds without output.
    assert_eq!(output.status.code(), Some(0), "{:?}", output);
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    Ok(())
}

#[test]
fn root_prefix_rejects_missing_value_before_command_dispatch() -> TestResult {
    // Given: the shared root prefix lacks its value.
    let args = ["--repo-root", "--help"];
    // When: the binary parses its prefix.
    let output = run(&args, b"")?;
    // Then: it follows xtask's generic error path.
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr, b"error: missing value for --repo-root\n");
    Ok(())
}

#[test]
fn narrow_binary_rejects_commands_outside_its_scope() -> TestResult {
    // Given: a full-xtask command outside the narrow executable's scope.
    let args = ["ci", "validate-graph"];
    // When: the binary dispatches it.
    let output = run(&args, b"")?;
    // Then: it reports narrow usage with status 2.
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(output.stderr.starts_with(b"usage: ci-controller "));
    Ok(())
}

#[test]
fn conventional_commits_accepts_a_conventional_subject() -> TestResult {
    // Given: a conventional subject passed inline.
    let args = [
        "repository",
        "conventional-commits",
        "--message",
        "fix(ci): keep one planner",
    ];
    // When: the commit convention check runs through the executable.
    let output = run(&args, b"")?;
    // Then: it passes silently.
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert!(output.stdout.is_empty());
    Ok(())
}

#[test]
fn conventional_commits_refuses_a_nonconventional_subject() -> TestResult {
    // Given: a subject with no type prefix.
    let args = [
        "repository",
        "conventional-commits",
        "--message",
        "keep one planner",
    ];
    // When: the commit convention check runs.
    let output = run(&args, b"")?;
    // Then: it fails with the shared release-notes guidance.
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8(output.stderr)?.contains("Conventional Commits:"));
    Ok(())
}

#[test]
fn upstream_pin_requires_both_revisions() -> TestResult {
    // Given: the pin guard without its base and head revisions.
    let args = ["repository", "llama-upstream-pin"];
    // When: it parses its arguments.
    let output = run(&args, b"")?;
    // Then: it reports the shared argparse-style usage error.
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(
        String::from_utf8(output.stderr)?
            .contains("the following arguments are required: base_revision, head_revision")
    );
    Ok(())
}

#[test]
fn affected_crates_routes_controller_sources_to_the_controller() -> TestResult {
    // Given: a changed controller source file.
    let input = b"tools/ci-controller/src/main.rs\n";
    // When: Cargo-metadata routing runs in-process.
    let output = run(&["repository", "affected-crates", "--stdin"], input)?;
    // Then: the controller and its reverse dependent xtask are affected.
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let crates = report["affected"].as_array().ok_or("affected")?;
    for name in ["ci-controller", "xtask"] {
        assert!(crates.iter().any(|value| value == name), "{report}");
    }
    Ok(())
}

#[test]
fn bootstrap_reports_the_controller_with_no_other_workspace_crate() -> TestResult {
    // Given: the test's own Cargo-built controller executable.
    // When: it proves its bootstrap identity against the real workspace graph.
    let output = run(&["automation", "bootstrap"], b"")?;
    // Then: the boundary admits it and the three report fields are printed.
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let stdout = String::from_utf8(output.stdout)?;
    let binary = stdout
        .lines()
        .find_map(|line| line.strip_prefix("binary_path="))
        .ok_or("binary_path")?;
    assert!(Path::new(binary).ends_with(format!("ci-controller{}", std::env::consts::EXE_SUFFIX)));
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("target_directory="))
    );
    assert!(stdout.lines().any(|line| line.starts_with("host=")));
    Ok(())
}

#[test]
fn bootstrap_rejects_extra_arguments() -> TestResult {
    // Given: bootstrap with an unexpected argument.
    let output = run(&["automation", "bootstrap", "--release"], b"")?;
    // Then: the controller's own usage is reported through the error path.
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        output.stderr,
        b"error: usage: ci-controller automation bootstrap\n"
    );
    Ok(())
}

#[test]
fn execution_record_cli_preserves_unknowns_and_does_not_echo_runner_name() -> TestResult {
    let trace = include_str!("fixtures/decision-selector-trace.json");
    let input = serde_json::to_vec(&serde_json::json!({
        "selection_trace": trace,
        "context": {"runner": {"name": "private-runner", "os": "Linux", "arch": "X64"}}
    }))?;
    let output = run(&["ci", "execution-record"], &input)?;
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout)?;
    assert!(!text.contains("private-runner"));
    let record: serde_json::Value = serde_json::from_str(&text)?;
    assert_eq!(record["authenticity"], "unverified");
    assert!(record["actual_runner"].is_null());
    assert!(record["effective_cache_mode"].is_null());
    assert_eq!(record["authorizes_execution"], false);
    Ok(())
}

#[test]
fn execution_record_cli_rejects_untrusted_input_without_echoing_it() -> TestResult {
    let output = run(&["ci", "execution-record"], b"private-invalid-input")?;
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!String::from_utf8(output.stderr)?.contains("private-invalid-input"));
    Ok(())
}
