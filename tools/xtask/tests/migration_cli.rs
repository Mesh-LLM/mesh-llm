use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

type TestResult = Result<(), Box<dyn Error>>;

#[path = "migration_cli/copied_verifier.rs"]
mod copied_verifier;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

fn run(cwd: &Path, args: &[&str]) -> Result<Output, Box<dyn Error>> {
    Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(cwd)
        .args(args)
        .output()?)
}

#[test]
fn migration_cli_repo_root_and_nested_invocations_match() -> TestResult {
    // Given: the checkout and a nested directory in the same workspace.
    let root = root();
    // When: the same repository check runs from both directories.
    let top = run(&root, &["repo-consistency", "no-console-print"])?;
    let nested = run(
        &root.join("tools/xtask"),
        &["repo-consistency", "no-console-print"],
    )?;
    // Then: both invocations return the same status and output.
    assert!(
        top.status.success(),
        "{}",
        String::from_utf8_lossy(&top.stderr)
    );
    assert_eq!(top, nested);
    Ok(())
}

#[test]
fn migration_cli_explicit_root_works_outside_checkout() -> TestResult {
    // Given: an unrelated working directory and the real repository root.
    let outside = std::env::temp_dir();
    let root = root();
    // When: the caller supplies the root explicitly.
    let output = run(
        &outside,
        &[
            "--repo-root",
            root.to_str().ok_or("non-UTF8 checkout")?,
            "repo-consistency",
            "no-console-print",
        ],
    )?;
    // Then: the repository check succeeds without relying on the build checkout.
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn migration_cli_relative_explicit_root_resolves_from_cwd() -> TestResult {
    // Given: the xtask directory nested two levels below the repository root.
    let nested = root().join("tools/xtask");
    // When: the root is given as a relative path.
    let output = run(
        &nested,
        &[
            "--repo-root",
            "../..",
            "repo-consistency",
            "no-console-print",
        ],
    )?;
    // Then: root selection succeeds without using the build-time manifest path.
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn migration_cli_outside_checkout_does_not_fall_back_to_build_root() -> TestResult {
    // Given: an unrelated directory without repository markers.
    // When: a repository-owned command runs there without an explicit root.
    let output = run(
        &std::env::temp_dir(),
        &["repo-consistency", "no-console-print"],
    )?;
    // Then: it fails on stderr with no success output.
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("repo root"));
    Ok(())
}

#[test]
fn migration_cli_invalid_explicit_root_never_falls_back() -> TestResult {
    // Given: a valid checkout cwd and an invalid explicit directory.
    let root = root();
    // When: the invalid directory is selected.
    let output = run(
        &root,
        &[
            "--repo-root",
            "/this-xtask-root-does-not-exist",
            "repo-consistency",
            "no-console-print",
        ],
    )?;
    // Then: it fails instead of using the checkout cwd.
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("repo root"));
    Ok(())
}

#[test]
fn migration_cli_malformed_args_preserve_error_channel() -> TestResult {
    // Given: a valid checkout.
    // When: the caller omits the root value.
    let output = run(&root(), &["--repo-root"])?;
    // Then: the CLI rejects it without emitting JSON on stdout.
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).starts_with("error: "));
    Ok(())
}

#[test]
fn migration_cli_unicode_path_with_spaces_is_not_reparsed() -> TestResult {
    // Given: a non-repository directory with a Unicode name and spaces.
    let path = std::env::temp_dir().join(format!("xtask-róot path {}", std::process::id()));
    fs::create_dir_all(&path)?;
    // When: it is passed as one explicit path argument.
    let output = run(
        &root(),
        &[
            "--repo-root",
            path.to_str().ok_or("non-UTF8 fixture")?,
            "repo-consistency",
            "no-console-print",
        ],
    )?;
    // Then: validation reports the exact path, never falling back to cwd.
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains(&path.display().to_string()));
    fs::remove_dir(&path)?;
    Ok(())
}
