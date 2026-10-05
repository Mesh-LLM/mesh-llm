//! Actual source-owned Bash plus a local Git repository; no product build.
use super::{document, job, named, support, text};
use std::{
    fs,
    os::unix::fs::PermissionsExt as _,
    process::{Command, Output},
};
fn command(f: &support::Fixture, program: &str, args: &[&str]) -> Command {
    let mut command = Command::new(program);
    command
        .env_clear()
        .env("GIT_MASTER", "1")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .args(args)
        .current_dir(f.path())
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", f.path().join("home"))
        .env("GIT_CONFIG_GLOBAL", f.path().join("home/gitconfig"))
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("XDG_CONFIG_HOME", f.path().join("home"));
    command
}
fn git(f: &support::Fixture, args: &[&str]) -> Output {
    f.run(command(f, "/usr/bin/git", args))
}
#[test]
fn graph_release_ui_step_admits_exact_source_after_container_ownership_and_rejects_mismatch() {
    let doc = document("ci-ui-artifact-slice.yml");
    let run = text(
        named(job(&doc, "ui_artifact"), "Prepare release UI version"),
        "run",
    )
    .unwrap();
    for mode in ["matching", "mismatching", "malformed"] {
        let f = support::Fixture::new();
        fs::create_dir(f.path().join("home")).unwrap();
        for args in [
            vec!["init", "--quiet"],
            vec![
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "fixture",
            ],
        ] {
            let output = git(&f, &args);
            assert!(output.status.success(), "{output:?}");
        }
        let head = git(&f, &["rev-parse", "HEAD"]);
        assert!(head.status.success());
        let sha = String::from_utf8(head.stdout).unwrap().trim().to_owned();
        fs::create_dir(f.path().join("scripts")).unwrap();
        let script = f.path().join("scripts/release-version.sh");
        fs::write(&script,"#!/bin/bash\nset -euo pipefail\ngit rev-parse HEAD > version-source\nprintf '%s\\n' \"$1\" > version-tag\n").unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let mut untrusted = command(&f, "/usr/bin/git", &["rev-parse", "HEAD"]);
        untrusted.env("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1");
        let output = f.run(untrusted);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("dubious ownership"));
        let mut prepare = command(&f, "/bin/bash", &["-c", run]);
        prepare
            .env("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
            .env("GITHUB_WORKSPACE", f.path())
            .env("GITHUB_ENV", f.path().join("github-env"))
            .env("RELEASE_TAG", "v0.76.0-rc9")
            .env(
                "UI_SOURCE_SHA",
                match mode {
                    "matching" => sha.as_str(),
                    "mismatching" => "0000000000000000000000000000000000000000",
                    _ => "invalid",
                },
            );
        let output = f.run(prepare);
        if mode == "matching" {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                fs::read_to_string(f.path().join("version-source")).unwrap(),
                format!("{sha}\n")
            );
            assert_eq!(
                fs::read_to_string(f.path().join("version-tag")).unwrap(),
                "v0.76.0-rc9\n"
            );
            assert_eq!(
                fs::read_to_string(f.path().join("github-env")).unwrap(),
                "VITE_MESH_LLM_DEBUG_UI=false\n"
            );
            let safe = git(&f, &["config", "--global", "--get-all", "safe.directory"]);
            assert!(safe.status.success());
            assert_eq!(
                String::from_utf8(safe.stdout).unwrap(),
                format!("{}\n", f.path().display())
            );
        } else {
            assert!(!output.status.success());
            assert!(!f.path().join("version-tag").exists());
            assert!(!f.path().join("github-env").exists());
        }
        f.0.close().unwrap();
    }
}
