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
