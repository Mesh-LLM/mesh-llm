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

// Native CLI refusal proof; no execution of the protected external audit action.
fn bounded_authority(command: Command) -> crate::process::RawProcessReport {
    use crate::process::{self, Value};
    let environment = command
        .get_envs()
        .filter_map(|(key, value)| {
            value.map(|value| (key.to_owned(), Value::Public(value.to_owned())))
        })
        .collect();
    let raw = process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: command.get_current_dir().unwrap().to_owned(),
            arguments: command
                .get_args()
                .map(|value| Value::Public(value.to_owned()))
                .collect(),
            environment,
        },
        &process::Limits {
            execution: std::time::Duration::from_secs(3),
            graceful_shutdown: std::time::Duration::from_millis(100),
            forced_shutdown: std::time::Duration::from_millis(100),
            retained_bytes_per_stream: 16384,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(16384),
            stderr: std::num::NonZeroUsize::new(16384),
        },
    )
    .unwrap();
    let p = &raw.process;
    assert_eq!(p.outcome, process::Outcome::Exited);
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
    );
    assert!(p.stdout.line_capture_complete && p.stderr.line_capture_complete);
    assert_eq!(
        raw.stdout.as_ref().unwrap().as_bytes().len() as u64,
        p.stdout.bytes_seen
    );
    assert_eq!(
        raw.stderr.as_ref().unwrap().as_bytes().len() as u64,
        p.stderr.bytes_seen
    );
    assert!(raw.stdout.as_ref().unwrap().as_bytes().is_empty());
    raw
}
#[test]
fn authority_native_cli_escaped_registry_keys_keep_env_file_hosted_and_depot_policy() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("config.json");
    for payload in [
        r#"{"auths":{"registry\u002eDEPOT\u002eDEV":{"auth":"private-fixture-token"}}}"#,
        r#"{"credHelpers":{"REGISTRY\u002eDEPOT\u002eDEV":"private-fixture-token"}}"#,
        r#"{"auths":{"ghcr.io":{"auth":"private-fixture-token"}}}"#,
    ] {
        let depot_key = payload.contains("u002e");
        for depot in [false, true] {
            for file in [false, true] {
                let mut cmd = command(directory.path());
                cmd.env("INPUT_DEPOT_SELECTED", depot.to_string());
                if file {
                    std::fs::write(&config, payload).unwrap();
                } else {
                    cmd.env("DOCKER_AUTH_CONFIG", payload);
                }
                let raw = bounded_authority(cmd);
                let accepted = !(depot || depot_key);
                assert_eq!(raw.process.status.unwrap().success(), accepted);
                let stderr = String::from_utf8_lossy(raw.stderr.as_ref().unwrap().as_bytes());
                assert!(
                    !stderr.contains("private-fixture-token")
                        && !stderr.contains("ghcr.io")
                        && !stderr.contains("REGISTRY")
                        && !stderr.contains("u002e")
                );
                if accepted {
                    assert!(stderr.is_empty());
                } else {
                    assert!(stderr.contains("DOCKER_AUTH_CONFIG/config.json"));
                    assert!(stderr.contains(if depot {
                        "authentication"
                    } else {
                        "depot-authentication"
                    }));
                }
                if file {
                    std::fs::remove_file(&config).unwrap();
                }
            }
        }
    }
    directory.close().unwrap();
}
#[test]
fn authority_native_cli_all_endpoint_variables_refuse_private_classified_candidates() {
    let directory = tempfile::tempdir().unwrap();
    for name in [
        "ACTIONS_CACHE_URL",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_RUNTIME_URL",
    ] {
        for (endpoint, reason) in [
            (
                "https://cache.example.invalid/cache",
                "endpoint must be GitHub-owned HTTPS or an explicit loopback proxy",
            ),
            ("https://cache.depot.dev/cache", "unapproved Depot endpoint"),
            ("https://user@attacker.example/cache", "userinfo"),
            (
                "https://actions.githubusercontent.com:443@attacker.example/",
                "userinfo",
            ),
            (
                "https://cache.example.invalid:8443/cache",
                "endpoint must be GitHub-owned HTTPS or an explicit loopback proxy",
            ),
            (
                "http://actions.githubusercontent.com/cache",
                "endpoint must be GitHub-owned HTTPS or an explicit loopback proxy",
            ),
            ("ftp://cache.example.invalid/cache", "malformed"),
            (
                "http://localhost/cache",
                "endpoint must be GitHub-owned HTTPS or an explicit loopback proxy",
            ),
            (
                "http://127.0.0.1:12345",
                "endpoint must be GitHub-owned HTTPS or an explicit loopback proxy",
            ),
            ("http://[::1]:65536/cache", "malformed"),
        ] {
            let mut cmd = command(directory.path());
            cmd.env(name, endpoint);
            let raw = bounded_authority(cmd);
            assert_eq!(raw.process.status.unwrap().code(), Some(1));
            let stderr = String::from_utf8_lossy(raw.stderr.as_ref().unwrap().as_bytes());
            assert!(stderr.contains(&format!("{name}: {reason}")), "{stderr}");
            for secret in [
                endpoint,
                "cache.example.invalid",
                "cache.depot.dev",
                "attacker.example",
                "actions.githubusercontent.com",
                "/cache",
                "8443",
                "65536",
                "443",
                "12345",
            ] {
                assert!(!stderr.contains(secret), "{stderr}");
            }
        }
    }
    directory.close().unwrap();
}
