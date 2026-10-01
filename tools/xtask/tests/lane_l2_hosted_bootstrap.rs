use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

type TestResult = Result<(), Box<dyn Error>>;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn script(text: &str, step: &str) -> Result<String, Box<dyn Error>> {
    let section = text.split(step).nth(1).ok_or("missing step")?;
    let body = section
        .split("      run: |\n")
        .nth(1)
        .ok_or("missing run")?;
    Ok(body
        .lines()
        .take_while(|line| line.is_empty() || line.starts_with("        "))
        .map(|line| line.strip_prefix("        ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n"))
}

#[test]
fn hosted_profile_rejects_nonhosted_and_unknown_contexts() -> TestResult {
    let action = fs::read_to_string(root().join(".github/actions/prepare-automation/action.yml"))?;
    let body = script(&action, "- name: Validate preparation profile\n")?;
    for (profile, os, environment, accepted) in [
        ("hosted-bare", "Linux", "github-hosted", true),
        ("hosted-bare", "Linux", "self-hosted", false),
        ("hosted-bare", "Windows", "github-hosted", false),
        ("unknown", "Linux", "github-hosted", false),
        ("image", "Linux", "self-hosted", true),
    ] {
        let output = Command::new("/bin/bash")
            .args(["-c", &body])
            .env("AUTOMATION_PROFILE", profile)
            .env("RUNNER_OS", os)
            .env("RUNNER_ENVIRONMENT", environment)
            .output()?;
        assert_eq!(
            output.status.success(),
            accepted,
            "{profile}/{os}/{environment}"
        );
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn hosted_build_exports_locked_output_without_just_or_wrapper() -> TestResult {
    use std::os::unix::fs::PermissionsExt;

    let sandbox = tempfile::tempdir()?;
    let binary = sandbox.path().join("xtask");
    fs::write(&binary, "fixture executable")?;
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o755))?;
    let cargo = sandbox.path().join("cargo");
    fs::write(
        &cargo,
        r#"#!/bin/bash
set -euo pipefail
test -z "$RUSTC_WRAPPER"
test -z "$CARGO_BUILD_RUSTC_WRAPPER"
case "$*" in
  'build --locked --release -p xtask --bin xtask') ;;
  'run --locked --release -p xtask --bin xtask -- automation bootstrap')
    printf 'binary_path=%s/xtask\ntarget_directory=%s\n' "$RUNNER_TEMP" "$RUNNER_TEMP" ;;
  *) exit 99 ;;
esac
"#,
    )?;
    fs::set_permissions(&cargo, fs::Permissions::from_mode(0o755))?;
    let action = fs::read_to_string(root().join(".github/actions/prepare-automation/action.yml"))?;
    let body = script(&action, "- name: Build automation tool\n")?;
    let output_file = sandbox.path().join("output");
    let environment_file = sandbox.path().join("environment");
    let output = Command::new("/bin/bash")
        .args(["-c", &body])
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", sandbox.path().display()),
        )
        .env("AUTOMATION_PROFILE", "hosted-bare")
        .env("RUNNER_TEMP", sandbox.path())
        .env("GITHUB_OUTPUT", &output_file)
        .env("GITHUB_ENV", &environment_file)
        .env("RUSTC_WRAPPER", "missing-sccache")
        .env("CARGO_BUILD_RUSTC_WRAPPER", "missing-sccache")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(environment_file)?,
        format!("MESH_LLM_AUTOMATION_BIN={}\n", binary.display())
    );
    assert!(
        fs::read_to_string(output_file)?.contains(&format!("binary_path={}\n", binary.display()))
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn package_action_consumes_prepared_binary_without_compilation() -> TestResult {
    use std::os::unix::fs::PermissionsExt;

    let sandbox = tempfile::tempdir()?;
    let cargo = sandbox.path().join("cargo");
    fs::write(&cargo, "#!/bin/sh\nexit 99\n")?;
    fs::set_permissions(&cargo, fs::Permissions::from_mode(0o755))?;
    let binary = sandbox.path().join("xtask");
    fs::write(
        &binary,
        r#"#!/bin/bash
set -euo pipefail
test "$1 $2" = 'repository cargo-packages'
shift 2
test "$1 $2 $3 $4" = '--generation current --crates ["fixture"]'
shift 4
test "$1" = --cargo
test "$2" = "$EXPECTED_CARGO"
printf '["fixture"]\n'
"#,
    )?;
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o755))?;
    let text =
        fs::read_to_string(root().join(".github/actions/resolve-cargo-packages/action.yml"))?;
    let body = script(&text, "- id: resolve\n")?;
    let output_path = sandbox.path().join("output");
    let output = Command::new("/bin/bash")
        .args(["-c", &body])
        .env("PATH", sandbox.path())
        .env("MESH_LLM_AUTOMATION_BIN", &binary)
        .env("EXPECTED_CARGO", &cargo)
        .env("REQUESTED_CRATES", "[\"fixture\"]")
        .env("PACKAGE_GENERATION", "current")
        .env("PLANNED_BATCHES", "")
        .env("GITHUB_OUTPUT", &output_path)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(fs::read_to_string(output_path)?, "crates=[\"fixture\"]\n");
    Ok(())
}
