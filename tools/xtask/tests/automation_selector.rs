//! Exercise the actual shared selector with finite local tools, never Cargo/bootstrap.
#![cfg(unix)]
use std::{
    fs,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Fixture {
    temporary: tempfile::TempDir,
    repository: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("repository with spaces");
        fs::create_dir(&repository).unwrap();
        fs::write(repository.join("Justfile"), "# fixture bootstrap policy\n").unwrap();
        let fixture = Self {
            temporary,
            repository,
        };
        fixture.executable(
            "just",
            r#"#!/bin/bash
printf called > "$FIXTURE_DIRECTORY/just-called"
[[ "$1" == --justfile && "$2" == "$REPO_ROOT/Justfile" && "$3" == automation-run ]] || exit 92
printf '%s\0' "$HOME" "$@"
exit 19
"#,
        );
        fixture.executable(
            "cargo",
            r#"#!/bin/bash
printf called > "$FIXTURE_DIRECTORY/cargo-called"
echo forbidden-cargo-fallback >&2
exit 91
"#,
        );
        fixture
    }

    fn directory(&self) -> &Path {
        self.temporary.path()
    }

    fn executable(&self, name: &str, bytes: &str) -> PathBuf {
        let path = self.directory().join(name);
        fs::write(&path, bytes).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn invoke(&self, configured: Option<&str>) -> Output {
        self.invoke_platform(configured, None)
    }

    fn invoke_platform(&self, configured: Option<&str>, platform: Option<&str>) -> Output {
        let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/lib/automation.sh");
        let mut command = Command::new("/bin/bash");
        command
            .args([
                "-c",
                r#"
set -euo pipefail
source "$1"
if [[ "${FIXTURE_PLATFORM+set}" == set ]]; then OSTYPE="$FIXTURE_PLATFORM"; fi
HOME='/isolated/client/home'; export HOME
set +e
mesh_automation automation agent-client-config 'argument with spaces' '' $'line1\nline2'
status=$?
set -e
[[ "$HOME" == /isolated/client/home ]] || exit 99
exit "$status"
"#,
                "fixture",
            ])
            .arg(helper)
            .current_dir(self.directory())
            .env_clear()
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.directory().display()),
            )
            .env("HOME", "/original/toolchain/home")
            .env("REPO_ROOT", &self.repository)
            .env("FIXTURE_DIRECTORY", self.directory());
        if let Some(configured) = configured {
            command.env("MESH_LLM_AUTOMATION_BIN", configured);
        }
        if let Some(platform) = platform {
            command.env("FIXTURE_PLATFORM", platform);
        }
        command.output().unwrap()
    }

    fn no_bootstrap(&self) {
        assert!(!self.directory().join("just-called").exists());
        assert!(!self.directory().join("cargo-called").exists());
    }
}

#[test]
fn configured_absolute_executable_preserves_arguments_and_actual_failure() {
    let fixture = Fixture::new();
    let owner = fixture.executable(
        "protected owner with spaces",
        "#!/bin/bash\nprintf '%s\\0' \"$HOME\" \"$@\"\nexit 17\n",
    );
    let output = fixture.invoke(owner.to_str());
    assert_eq!(output.status.code(), Some(17));
    assert_eq!(output.stdout, b"/isolated/client/home\0automation\0agent-client-config\0argument with spaces\0\0line1\nline2\0");
    assert!(output.stderr.is_empty());
    fixture.no_bootstrap();
}

#[test]
fn configured_invalid_empty_relative_missing_directory_or_nonexecutable_fail_closed() {
    let fixture = Fixture::new();
    let relative = fixture.executable("relative-owner", "#!/bin/bash\nexit 18\n");
    let directory = fixture.directory().join("executable directory");
    fs::create_dir(&directory).unwrap();
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
    let nonexecutable = fixture.directory().join("nonexecutable-owner");
    fs::write(&nonexecutable, "#!/bin/bash\nexit 18\n").unwrap();
    fs::set_permissions(&nonexecutable, fs::Permissions::from_mode(0o600)).unwrap();
    let missing = fixture.directory().join("missing-owner");
    for configured in [
        "",
        relative.file_name().unwrap().to_str().unwrap(),
        missing.to_str().unwrap(),
        directory.to_str().unwrap(),
        nonexecutable.to_str().unwrap(),
    ] {
        let output = fixture.invoke(Some(configured));
        assert_eq!(
            output.status.code(),
            Some(1),
            "{configured}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("must be an absolute executable"));
        fixture.no_bootstrap();
    }
}

#[test]
fn absent_configuration_uses_normal_just_and_source_time_home_without_mutating_client_home() {
    let fixture = Fixture::new();
    let output = fixture.invoke(None);
    assert_eq!(output.status.code(), Some(19));
    let arguments = output
        .stdout
        .split(|byte| *byte == 0)
        .map(|part| String::from_utf8(part.to_vec()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        arguments,
        vec![
            "/original/toolchain/home".to_owned(),
            "--justfile".into(),
            fixture
                .repository
                .join("Justfile")
                .to_str()
                .unwrap()
                .to_owned(),
            "automation-run".into(),
            "automation".into(),
            "agent-client-config".into(),
            "argument with spaces".into(),
            "".into(),
            "line1\nline2".into(),
            "".into()
        ]
    );
    assert!(output.stderr.is_empty());
    assert!(fixture.directory().join("just-called").is_file());
    assert!(!fixture.directory().join("cargo-called").exists());
}

#[test]
fn windows_absolute_owner_conversion_preserves_arguments_status_and_admission() {
    let fixture = Fixture::new();
    fixture.executable(
        "normalized owner with spaces",
        r#"#!/bin/bash
printf '%s\0' "$HOME" "$@"
exit 17
"#,
    );
    fixture.executable(
        "cygpath",
        r#"#!/bin/bash
printf '%s\0' "$@" >> "$FIXTURE_DIRECTORY/conversion-arguments"
printf '%s\n' "$FIXTURE_DIRECTORY/normalized owner with spaces"
"#,
    );
    for platform in ["msys", "cygwin"] {
        for configured in [
            r"C:\tools\owner with spaces.exe",
            "C:/tools/owner.exe",
            r"\\server\share\owner.exe",
        ] {
            let trace = fixture.directory().join("conversion-arguments");
            if trace.exists() {
                fs::remove_file(&trace).unwrap();
            }
            let output = fixture.invoke_platform(Some(configured), Some(platform));
            assert_eq!(output.status.code(), Some(17), "{output:?}");
            assert_eq!(output.stdout, b"/isolated/client/home\0automation\0agent-client-config\0argument with spaces\0\0line1\nline2\0");
            assert!(output.stderr.is_empty());
            assert_eq!(
                fs::read(trace).unwrap(),
                format!("-u\0--\0{configured}\0").as_bytes()
            );
            fixture.no_bootstrap();
        }
    }
}

#[test]
fn native_path_conversion_failure_and_wrong_platform_never_fall_back() {
    let fixture = Fixture::new();
    fixture.executable(
        "cygpath",
        r#"#!/bin/bash
printf called > "$FIXTURE_DIRECTORY/conversion-called"
exit 23
"#,
    );
    for (configured, platform, converted) in [
        (r"C:\tools\owner.exe", "linux-gnu", false),
        (r"C:\tools\owner.exe", "msys", true),
        ("C:relative", "msys", false),
    ] {
        let trace = fixture.directory().join("conversion-called");
        if trace.exists() {
            fs::remove_file(&trace).unwrap();
        }
        let output = fixture.invoke_platform(Some(configured), Some(platform));
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("must be an absolute executable"));
        assert_eq!(trace.exists(), converted);
        fixture.no_bootstrap();
    }
}
