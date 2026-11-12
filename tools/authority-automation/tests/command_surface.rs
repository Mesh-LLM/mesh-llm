use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn run(args: &[&str], input: &[u8]) -> Result<Output, Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child = Command::new(env!("CARGO_BIN_EXE_authority-automation"))
        .current_dir(root)
        .args(args)
        .env_remove("AUTHORITY_FIXTURE_ENDPOINT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.take().ok_or("stdin")?.write_all(input)?;
    Ok(child.wait_with_output()?)
}

#[test]
fn file_projection_hashes_stdin() -> TestResult {
    // Given: bytes on stdin.
    // When: the digest helper prepare-llama.sh falls back to runs.
    let output = run(&["artifact", "file-projection", "sha256", "-"], b"abc")?;
    // Then: the SHA-256 hex digest is printed with a newline.
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        output.stdout,
        b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
    );
    Ok(())
}

#[test]
fn authority_audit_refuses_a_missing_endpoint() -> TestResult {
    // Given: an endpoint variable that is not set.
    let args = [
        "ci-ops",
        "authority-audit",
        "endpoint",
        "AUTHORITY_FIXTURE_ENDPOINT",
    ];
    // When: the endpoint audit runs.
    let output = run(&args, b"")?;
    // Then: it fails with xtask's stream split and status.
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty());
    assert_eq!(output.stderr, b"AUTHORITY_FIXTURE_ENDPOINT: missing\n");
    Ok(())
}

#[test]
fn models_resolve_requires_manifest_and_cadence() -> TestResult {
    // Given: the resolver without its required arguments.
    // When: it parses its arguments.
    let output = run(&["models", "resolve"], b"")?;
    // Then: it reports the argparse-style usage error with status 2.
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(
        String::from_utf8(output.stderr)?
            .ends_with("the following arguments are required: manifest, --cadence\n")
    );
    Ok(())
}

#[test]
fn root_prefix_rejects_missing_value_before_command_dispatch() -> TestResult {
    // Given: the shared root prefix lacks its value.
    let output = run(&["--repo-root", "--help"], b"")?;
    // Then: it follows xtask's generic error path.
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr, b"error: missing value for --repo-root\n");
    Ok(())
}

#[test]
fn narrow_binary_rejects_commands_outside_its_scope() -> TestResult {
    // Given: full-xtask commands outside the narrow executable's scope.
    for args in [&["models", "generate"][..], &["ci", "plan"], &["--help"]] {
        // When: the binary dispatches it.
        let output = run(args, b"")?;
        // Then: it reports narrow usage with status 2.
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty());
        assert!(
            output.stderr.starts_with(b"usage: authority-automation "),
            "{args:?}"
        );
    }
    Ok(())
}

#[test]
fn bootstrap_reports_the_executable_with_only_the_controller_library() -> TestResult {
    // Given: the test's own Cargo-built executable.
    // When: it proves its bootstrap identity against the real workspace graph.
    let output = run(&["automation", "bootstrap"], b"")?;
    // Then: the boundary admits it and the three report fields are printed.
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let stdout = String::from_utf8(output.stdout)?;
    let binary = stdout
        .lines()
        .find_map(|line| line.strip_prefix("binary_path="))
        .ok_or("binary_path")?;
    assert!(Path::new(binary).ends_with(format!(
        "authority-automation{}",
        std::env::consts::EXE_SUFFIX
    )));
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
    // Then: the executable's own usage is reported through the error path.
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        output.stderr,
        b"error: usage: authority-automation automation bootstrap\n"
    );
    Ok(())
}
