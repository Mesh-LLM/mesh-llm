use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

type TestResult = Result<(), Box<dyn Error>>;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

fn bootstrap(cwd: &Path, cargo: Option<&Path>) -> Result<Output, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.current_dir(cwd).args(["automation", "bootstrap"]);
    if let Some(cargo) = cargo {
        command.env("CARGO", cargo);
    }
    Ok(command.output()?)
}

fn field<'a>(stdout: &'a str, key: &str) -> Option<&'a str> {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
}

#[test]
fn migration_bootstrap_reports_absolute_tool_in_cargo_target_root() -> TestResult {
    // Given: the checkout with the xtask test binary built by Cargo.
    // When: bootstrap runs from a nested directory with the active toolchain.
    let output = bootstrap(&root().join("tools/xtask"), None)?;
    // Then: it names this exact binary beneath Cargo's resolved target root.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    let stdout = String::from_utf8(output.stdout)?;
    println!("{stdout}");
    let binary = PathBuf::from(field(&stdout, "binary_path").ok_or("missing binary_path")?);
    let target = PathBuf::from(field(&stdout, "target_directory").ok_or("missing target")?);
    assert!(binary.is_absolute());
    assert_eq!(
        binary,
        Path::new(env!("CARGO_BIN_EXE_xtask")).canonicalize()?
    );
    assert!(binary.starts_with(target.canonicalize()?));
    assert!(field(&stdout, "host").is_some_and(|host| !host.is_empty()));
    Ok(())
}

#[test]
fn migration_bootstrap_missing_toolchain_is_actionable() -> TestResult {
    // Given: a CARGO override naming a toolchain that is not installed.
    let absent = std::env::temp_dir().join("xtask-bootstrap-no-toolchain/cargo");
    // When: bootstrap runs.
    let output = bootstrap(&root(), Some(&absent))?;
    // Then: it fails on stderr, names the missing executable and the fix.
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("{stderr}");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(stderr.contains(&absent.display().to_string()), "{stderr}");
    assert!(
        stderr.contains("install the pinned Rust toolchain"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn migration_bootstrap_outside_checkout_needs_repo_root() -> TestResult {
    // Given: a directory with no workspace markers.
    // When: bootstrap runs there without --repo-root.
    let output = bootstrap(&std::env::temp_dir(), None)?;
    // Then: repository discovery fails before any Cargo work.
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("repo root"));
    Ok(())
}

#[test]
fn migration_bootstrap_copied_verifier_runs_without_cargo_or_source() -> TestResult {
    // Given: the tool copied to an isolated directory with no Cargo reachable.
    let isolated =
        std::env::temp_dir().join(format!("xtask-bootstrap-copied-{}", std::process::id()));
    fs::create_dir_all(&isolated)?;
    let verifier = isolated.join("release-attestation-verifier");
    fs::copy(env!("CARGO_BIN_EXE_xtask"), &verifier)?;
    let host = isolated.join("host.exe");
    fs::write(&host, b"unstamped host")?;
    // When: it inspects a host with PATH and CARGO pointing nowhere.
    let output = Command::new(&verifier)
        .current_dir(&isolated)
        .env("PATH", "")
        .env("CARGO", isolated.join("absent-cargo"))
        .args(["release-attestation", "inspect", "--binary"])
        .arg(&host)
        .arg("--json")
        .output()?;
    // Then: the verifier contract holds without any toolchain or checkout.
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    fs::remove_dir_all(&isolated)?;
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(summary["status"], "missing");
    Ok(())
}
