use std::path::Path;
use std::process::{Command, Output};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn run(args: &[&str]) -> Result<Output, Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    Ok(Command::new(env!("CARGO_BIN_EXE_runner-cleanup"))
        .current_dir(root)
        .args(args)
        .output()?)
}

#[test]
fn cleanup_help_matches_the_xtask_command() -> TestResult {
    // Given: the cleanup command's help request.
    let output = run(&["ci-ops", "runner-cleanup", "--help"])?;
    // Then: xtask's usage line is printed on stdout with status 0.
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert!(output.stderr.is_empty());
    assert!(String::from_utf8(output.stdout)?.starts_with("usage: ci-ops runner-cleanup --job "));
    Ok(())
}

#[test]
fn cleanup_refuses_missing_arguments_with_usage_status() -> TestResult {
    // Given: cleanup without its required job and evidence flags.
    let output = run(&["ci-ops", "runner-cleanup"])?;
    // Then: it refuses with xtask's usage status before touching any path.
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr)?.contains("job and evidence-uploaded are required"));
    Ok(())
}

#[test]
fn root_prefix_rejects_missing_value_before_command_dispatch() -> TestResult {
    let output = run(&["--repo-root", "--help"])?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr, b"error: missing value for --repo-root\n");
    Ok(())
}

#[test]
fn narrow_binary_rejects_commands_outside_its_scope() -> TestResult {
    for args in [
        &["ci-ops", "runner-identity"][..],
        &["ci", "plan"],
        &["--help"],
    ] {
        let output = run(args)?;
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty());
        assert!(
            output.stderr.starts_with(b"usage: runner-cleanup "),
            "{args:?}"
        );
    }
    Ok(())
}

#[test]
fn bootstrap_reports_the_executable_with_only_its_admitted_libraries() -> TestResult {
    // Given: the test's own Cargo-built executable against the real workspace graph.
    let output = run(&["automation", "bootstrap"])?;
    // Then: the boundary admits it and prints exactly one binary path.
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let stdout = String::from_utf8(output.stdout)?;
    let binaries: Vec<_> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("binary_path="))
        .collect();
    let [binary] = binaries.as_slice() else {
        return Err(format!("binary_path lines: {binaries:?}").into());
    };
    assert!(Path::new(binary).ends_with(format!("runner-cleanup{}", std::env::consts::EXE_SUFFIX)));
    Ok(())
}
