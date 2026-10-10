//! Register after applying upload/quality candidate. Executes admission only,
//! with fixture-local git/bootstrap functions; no compiler, product or Git process.
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output},
};

const SOURCE: &str = "0123456789012345678901234567890123456789";

fn block(step: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source =
        fs::read_to_string(root.join(".github/actions/upload-automation/action.yml")).unwrap();
    let body = source
        .split(step)
        .nth(1)
        .unwrap()
        .split("      run: |\n")
        .nth(1)
        .unwrap();
    body.lines()
        .take_while(|line| line.starts_with("        ") || line.trim().is_empty())
        .map(|line| line.strip_prefix("        ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn invoke(
    stage: bool,
    profile: &str,
    source_profile: &str,
    status: &str,
    late_dirty: bool,
    head: &str,
) -> (tempfile::TempDir, Output) {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("host-input")).unwrap();
    fs::write(
        dir.path().join("host-input/host.json"),
        b"generated release output",
    )
    .unwrap();
    fs::write(dir.path().join("status"), status).unwrap();
    let binary = dir.path().join("prepared-xtask");
    fs::write(
        &binary,
        b"#!/bin/bash\necho forbidden > executed\nexit 91\n",
    )
    .unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
    let stubs = r#"
git() {
  case "$1" in
    rev-parse) printf '%s\n' "$FIXTURE_HEAD" ;;
    status) test "$(cat "$FIXTURE_ROOT/status")" != STATUS_ERROR || return 97; cat "$FIXTURE_ROOT/status" ;;
    *) return 95 ;;
  esac
}
just() {
  test "$1" = automation-bootstrap || return 96
  if test "$FIXTURE_LATE_DIRTY" = 1; then printf ' M tools/xtask/Cargo.toml\n' > "$FIXTURE_ROOT/status"; fi
  printf 'binary_path=%s/prepared-xtask\n' "$FIXTURE_ROOT"
}
"#;
    let step = if stage {
        "- name: Build and stage automation\n"
    } else {
        "- name: Admit producer preparation profile\n"
    };
    let body = format!("{stubs}\n{}", block(step));
    let output = Command::new("/bin/bash")
        .args(["-c", &body])
        .current_dir(dir.path())
        .env("FIXTURE_ROOT", dir.path())
        .env("FIXTURE_HEAD", head)
        .env("FIXTURE_LATE_DIRTY", if late_dirty { "1" } else { "0" })
        .env("AUTOMATION_SOURCE_SHA", SOURCE)
        .env("AUTOMATION_PROFILE", profile)
        .env("AUTOMATION_SOURCE_PROFILE", source_profile)
        .env(
            "AUTOMATION_BINARY_PATH",
            if profile == "hosted-bare" {
                binary.to_str().unwrap()
            } else {
                ""
            },
        )
        .env("RUNNER_OS", "Linux")
        .env("RUNNER_TEMP", dir.path())
        .env("GITHUB_OUTPUT", dir.path().join("output"))
        .output()
        .unwrap();
    assert!(!dir.path().join("executed").exists());
    (dir, output)
}

#[test]
fn release_prepared_keeps_generated_and_version_preparation_contract() {
    for status in ["?? host-input/\n", " M Cargo.toml\n?? host-input/\n"] {
        let (dir, output) = invoke(true, "image", "release-prepared", status, false, SOURCE);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read(dir.path().join("immutable-automation/xtask")).unwrap(),
            fs::read(dir.path().join("prepared-xtask")).unwrap()
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("immutable-automation/source.txt")).unwrap(),
            format!("{SOURCE}\n")
        );
        assert!(
            fs::read_to_string(dir.path().join("output"))
                .unwrap()
                .starts_with("binary_sha256=")
        );
    }
}

#[test]
fn protected_clean_rejects_generated_or_tracked_changes_before_prepare_and_stage() {
    for stage in [false, true] {
        for status in [
            "?? host-input/\n",
            " M tools/xtask/Cargo.toml\n",
            "STATUS_ERROR",
        ] {
            let (dir, output) = invoke(
                stage,
                "hosted-bare",
                "protected-clean",
                status,
                false,
                SOURCE,
            );
            assert!(!output.status.success());
            assert!(!dir.path().join("immutable-automation").exists());
            assert!(!dir.path().join("output").exists());
        }
    }
    let (_, output) = invoke(true, "hosted-bare", "protected-clean", "", false, SOURCE);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (dir, output) = invoke(true, "image", "protected-clean", "", true, SOURCE);
    assert!(!output.status.success());
    assert!(!dir.path().join("output").exists());
}

#[test]
fn closed_profiles_and_source_identity_fail_before_publication() {
    for (profile, source_profile, head) in [
        ("hosted-bare", "release-prepared", SOURCE),
        ("unknown", "protected-clean", SOURCE),
        ("image", "", SOURCE),
        ("image", "untracked-allowed", SOURCE),
        (
            "image",
            "release-prepared",
            "1123456789012345678901234567890123456789",
        ),
    ] {
        let (dir, output) = invoke(false, profile, source_profile, "", false, head);
        assert!(!output.status.success());
        assert!(!dir.path().join("output").exists());
    }
}
