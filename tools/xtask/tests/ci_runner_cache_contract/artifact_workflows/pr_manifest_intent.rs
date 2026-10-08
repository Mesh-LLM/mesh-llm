//! Source-owned PR admission with actual bounded local Git and no remote fetch.
use super::fixture;
use serde_json::Value;
use std::{fs, os::unix::fs::symlink, process::Command};
fn git(f: &super::support::Fixture, args: &[&str]) -> String {
    let mut command = Command::new("/usr/bin/git");
    command
        .env_clear()
        .env("GIT_MASTER", "1")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", f.path())
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", f.path().join("isolated-gitconfig"))
        .current_dir(f.path())
        .args(args);
    let output = f.run(command);
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}
#[test]
fn graph_pr_plan_admits_only_exact_regular_protected_manifest_source() {
    for mode in ["matching", "draft", "malformed", "catalog-drift", "symlink"] {
        let f = fixture();
        fs::create_dir(f.path().join("ci")).unwrap();
        for name in ["ownership.yml", "slices.yml"] {
            fs::copy(
                super::support::root().join("ci").join(name),
                f.path().join("ci").join(name),
            )
            .unwrap();
        }
        git(&f, &["init", "--quiet"]);
        if mode == "catalog-drift" {
            fs::write(f.path().join("ci/slices.yml"), "{}\n").unwrap();
        }
        if mode == "symlink" {
            fs::remove_file(f.path().join("ci/slices.yml")).unwrap();
            symlink("ownership.yml", f.path().join("ci/slices.yml")).unwrap();
        }
        git(&f, &["add", "ci"]);
        git(
            &f,
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                "fixture",
            ],
        );
        let source = git(&f, &["rev-parse", "HEAD"]);
        if mode == "catalog-drift" {
            fs::copy(
                super::support::root().join("ci/slices.yml"),
                f.path().join("ci/slices.yml"),
            )
            .unwrap();
        }
        let expected_profile = if mode == "draft" {
            "pr-draft"
        } else {
            "pr-ready"
        };
        f.executable("automation",r#"[[ $# == 4 && "$1" == ci && "$2" == plan && "$3" == --manifest-root ]] || exit 91
[[ "$4" != "$GITHUB_WORKSPACE" && "$4" == "$RUNNER_TEMP/mesh-ci-manifests."* ]] || exit 92
[[ -f "$4/ci/ownership.yml" && -f "$4/ci/slices.yml" ]] || exit 93
/usr/bin/cmp "$4/ci/ownership.yml" "$GITHUB_WORKSPACE/ci/ownership.yml"
/usr/bin/cmp "$4/ci/slices.yml" "$GITHUB_WORKSPACE/ci/slices.yml"
/bin/cat > planner-input.json
jq -e --arg profile "$EXPECTED_PROFILE" --arg source "$SOURCE_SHA" '.profile == $profile and .source_sha == $source and .event_name == "pull_request"' planner-input.json >/dev/null
printf '%s\n' "$4" > planner-manifest-root
/bin/cat golden.json"#);
        let action = super::support::action("plan-ci");
        let super::Node::Seq(items) = action.get("runs").unwrap().get("steps").unwrap() else {
            panic!("steps")
        };
        let run = items
            .iter()
            .find(|s| s.get("id").and_then(super::Node::text) == Some("plan"))
            .unwrap()
            .get("run")
            .unwrap()
            .text()
            .unwrap();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .env("GIT_MASTER", "1")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .current_dir(f.path())
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", f.path().join("bin").display()),
            )
            .env("HOME", f.path())
            .env("TMPDIR", f.path())
            .env("GIT_CONFIG_GLOBAL", f.path().join("isolated-gitconfig"))
            .env("GIT_CONFIG_NOSYSTEM", "1");
        for (key, value) in [
            ("EVENT_NAME", "pull_request"),
            ("BASE_SHA", ""),
            ("DRAFT", if mode == "draft" { "true" } else { "false" }),
            ("AFFECTED_CRATES", "[]"),
            ("UPLOAD_ARTIFACT", "false"),
            ("ARTIFACT_NAME", ""),
            ("EXPECTED_PROFILE", expected_profile),
        ] {
            command.env(key, value);
        }
        fs::write(f.path().join("changed.txt"), "docs/MESHES.md\n").unwrap();
        let run = run.replace(
            "/tmp/changed_files.txt",
            &format!("\"{}\"", f.path().join("changed.txt").display()),
        );
        command
            .env(
                "SOURCE_SHA",
                if mode == "malformed" {
                    "invalid"
                } else {
                    source.as_str()
                },
            )
            .env("GITHUB_WORKSPACE", f.path())
            .env("RUNNER_TEMP", f.path())
            .env("GITHUB_OUTPUT", f.path().join("outputs"))
            .env("GITHUB_STEP_SUMMARY", f.path().join("summary"))
            .env("MESH_LLM_AUTOMATION_BIN", f.path().join("bin/automation"))
            .args(["-c", &run]);
        let output = f.run(command);
        if matches!(mode, "matching" | "draft") {
            assert!(output.status.success(), "{output:?}");
            let input: Value =
                serde_json::from_slice(&fs::read(f.path().join("planner-input.json")).unwrap())
                    .unwrap();
            assert_eq!(input["source_sha"], source);
            let manifest = fs::read_to_string(f.path().join("planner-manifest-root")).unwrap();
            let root = std::path::Path::new(manifest.trim());
            assert!(root.join("ci/ownership.yml").is_file());
            assert!(root.join("ci/slices.yml").is_file());
            assert!(!root.join(".git").exists());
            assert_eq!(fs::read_dir(root).unwrap().count(), 1);
            assert_eq!(fs::read_dir(root.join("ci")).unwrap().count(), 2);
        } else {
            assert!(!output.status.success());
            assert!(!f.path().join("planner-input.json").exists());
            assert!(!f.path().join("outputs").exists());
        }
        f.0.close().unwrap();
    }
}
