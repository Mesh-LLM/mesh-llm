use std::{fs, os::unix::fs::PermissionsExt, path::Path, process::Command};

const CALLERS: [(&str, &str); 6] = [
    ("skippy/scripts/family-certify.sh", "family_automation"),
    (
        "skippy/scripts/skippy-workload-certify.sh",
        "workload_automation",
    ),
    (
        "skippy/scripts/skippy-workload-oracles-build.sh",
        "workload_automation",
    ),
    ("scripts/ci-pi-smoke.sh", "agent_automation"),
    ("scripts/ci-goose-smoke.sh", "agent_automation"),
    ("skippy/scripts/skippy-openai-smoke.sh", "automation"),
];

fn selector(path: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::read_to_string(root.join(path))
        .unwrap()
        .split("# Frozen automation selection begins.\n")
        .nth(1)
        .unwrap()
        .split("# Frozen automation selection ends.")
        .next()
        .unwrap()
        .to_owned()
}

fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn run_selector(
    directory: &Path,
    path: &str,
    variable: &str,
    configured: Option<&str>,
) -> std::process::Output {
    let script = format!(
        "set -euo pipefail\n{}\nexport HOME='/isolated/client/home'\n\"${{{variable}[@]}}\" automation agent-client-config 'argument with spaces' '' $'line1\\nline2'\n",
        selector(path)
    );
    let mut command = Command::new("/bin/bash");
    command
        .args(["-c", &script])
        .current_dir(directory)
        .env("ROOT", directory.join("repository with spaces"))
        .env("HOME", "/original/toolchain/home")
        .env("PATH", format!("{}:/usr/bin:/bin", directory.display()))
        .env("SKIP_BUILD", "1")
        .env_remove("MESH_LLM_AUTOMATION_BIN");
    if let Some(value) = configured {
        command.env("MESH_LLM_AUTOMATION_BIN", value);
    }
    command.output().unwrap()
}

fn fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("repository with spaces");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("Justfile"), "# case-sensitive facade fixture\n").unwrap();
    executable(
        &directory.path().join("just"),
        "#!/bin/bash\n[[ $1 == --justfile && $2 == */Justfile && -f $2 && $3 == automation-run ]] || exit 92\nprintf '%s\\0' \"$HOME\" \"$@\"\nexit 19\n",
    );
    executable(
        &directory.path().join("cargo"),
        "#!/bin/bash\necho forbidden-cargo >&2\nexit 91\n",
    );
    directory
}

#[test]
fn frozen_controller_preserves_arguments_and_failure_without_just_or_cargo() {
    let directory = fixture();
    let owner = directory.path().join("frozen owner with spaces");
    executable(&owner, "#!/bin/bash\nprintf '%s\\0' \"$@\"\nexit 17\n");
    for (path, variable) in CALLERS {
        let output = run_selector(directory.path(), path, variable, owner.to_str());
        assert_eq!(output.status.code(), Some(17), "{path}");
        assert_eq!(
            output.stdout,
            b"automation\0agent-client-config\0argument with spaces\0\0line1\nline2\0",
            "{path}"
        );
        assert!(output.stderr.is_empty(), "{path}");
    }
}

#[test]
fn absent_controller_uses_case_correct_justfile_and_client_original_home() {
    let directory = fixture();
    for (path, variable) in CALLERS {
        let output = run_selector(directory.path(), path, variable, None);
        assert_eq!(output.status.code(), Some(19), "{path}");
        let home = if variable == "agent_automation" {
            "/original/toolchain/home"
        } else {
            "/isolated/client/home"
        };
        let expected = format!(
            "{home}\0--justfile\0{}\0automation-run\0automation\0agent-client-config\0argument with spaces\0\0line1\nline2\0",
            directory
                .path()
                .join("repository with spaces/Justfile")
                .display()
        );
        assert_eq!(output.stdout, expected.as_bytes(), "{path}");
        assert!(output.stderr.is_empty(), "{path}");
    }
}

#[test]
fn configured_empty_relative_missing_directory_and_nonexecutable_fail_closed() {
    let directory = fixture();
    let nonexecutable = directory.path().join("nonexecutable");
    fs::write(&nonexecutable, "#!/bin/bash\nexit 0\n").unwrap();
    fs::set_permissions(&nonexecutable, fs::Permissions::from_mode(0o600)).unwrap();
    for (path, variable) in CALLERS {
        for configured in [
            "",
            "relative",
            "/absent/frozen-controller",
            directory.path().to_str().unwrap(),
            nonexecutable.to_str().unwrap(),
        ] {
            let output = run_selector(directory.path(), path, variable, Some(configured));
            assert_eq!(output.status.code(), Some(1), "{path}: {configured}");
            assert!(output.stdout.is_empty(), "{path}: {configured}");
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains("absolute executable"), "{path}: {error}");
            assert!(!error.contains("forbidden-cargo"), "{path}: {error}");
        }
    }
}

#[test]
fn generic_config_ports_identity_and_source_calls_use_selected_authority() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (path, _) in CALLERS {
        let source = fs::read_to_string(root.join(path)).unwrap();
        assert!(!source.contains("$ROOT/justfile"), "{path}");
        for command in [
            "agent-client-config",
            "local-ports",
            "family-model-identity",
            "workload-smoke-config",
            "canary-receipts prepared-source",
            "workload-oracle-evidence",
        ] {
            assert!(
                !source.contains(&format!("cargo xtool automation {command}")),
                "{path}: {command}"
            );
        }
    }
}
