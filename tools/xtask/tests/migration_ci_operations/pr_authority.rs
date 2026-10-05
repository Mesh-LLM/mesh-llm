//! Actual CLI authority admission without checkout, Python, or credential output.
use std::{path::Path, process::Command};

fn command(directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .env_clear()
        .current_dir(directory)
        .args(["ci-ops", "pr-authority-audit"]);
    for name in ["PATH", "SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .env("INPUT_DEPOT_SELECTED", "true")
        .env("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "false")
        .env("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "false")
        .env("GITHUB_EVENT_NAME", "pull_request")
        .env("DOCKER_CONFIG", directory);
    command
}

#[test]
fn pr_authority_cli_works_before_checkout_and_has_no_success_output() {
    let directory = tempfile::tempdir().unwrap();
    let output = command(directory.path())
        .env(
            "ACTIONS_CACHE_URL",
            "https://actions.githubusercontent.com/cache",
        )
        .env("ACTIONS_RESULTS_URL", "http://[::1]:1234/results")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    let help = command(directory.path())
        .env_remove("INPUT_DEPOT_SELECTED")
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(
        String::from_utf8_lossy(&help.stdout).contains("cargo xtool ci-ops pr-authority-audit")
    );
    assert!(help.stderr.is_empty());
}

#[test]
fn pr_authority_cli_rejects_credentials_malformed_flags_and_deceptive_endpoints_privately() {
    let directory = tempfile::tempdir().unwrap();
    for (name, value) in [
        ("DEPOT_TOKEN", "private-value"),
        ("INPUT_DEPOT_SELECTED", "private-value"),
        (
            "ACTIONS_CACHE_URL",
            "https://user:private-value@actions.githubusercontent.com/cache",
        ),
        (
            "ACTIONS_RUNTIME_URL",
            "https://private-value.example.invalid/cache",
        ),
        ("DOCKER_AUTH_CONFIG", "private-value"),
    ] {
        let output = command(directory.path()).env(name, value).output().unwrap();
        assert!(!output.status.success(), "{name}");
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(name), "{stderr}");
        assert!(!stderr.contains("private-value"));
    }
}

#[test]
fn pr_authority_cli_preserves_native_cache_exception_and_docker_source_checks() {
    let directory = tempfile::tempdir().unwrap();
    let accepted = command(directory.path())
        .env("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "true")
        .env("ACTIONS_CACHE_URL", "https://cache.depot.dev/cache")
        .output()
        .unwrap();
    assert!(accepted.status.success());
    std::fs::write(directory.path().join("config.json"), br#"{"auths":{}}"#).unwrap();
    assert!(!command(directory.path()).output().unwrap().status.success());
    assert!(
        command(directory.path())
            .env("INPUT_DEPOT_SELECTED", "false")
            .output()
            .unwrap()
            .status
            .success()
    );
    std::fs::write(directory.path().join("config.json"), b"{\"auths\":NaN}").unwrap();
    assert!(
        !command(directory.path())
            .env("INPUT_DEPOT_SELECTED", "false")
            .output()
            .unwrap()
            .status
            .success()
    );
}

#[cfg(unix)]
#[test]
fn pr_authority_cli_does_not_treat_nonunicode_authority_as_missing() {
    use std::os::unix::ffi::OsStringExt;
    let directory = tempfile::tempdir().unwrap();
    let output = command(directory.path())
        .env("DEPOT_TOKEN", std::ffi::OsString::from_vec(vec![255]))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("DEPOT_TOKEN: invalid text"));
}
