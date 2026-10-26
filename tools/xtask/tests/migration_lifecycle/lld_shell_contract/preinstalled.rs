//! Persistent canary bootstrap must check tools without invoking installers.
use super::{Fixture, executable, fs, repo, workflow_yaml};

fn action_steps(path: &str) -> Vec<workflow_yaml::Node> {
    let source = fs::read_to_string(repo().join(path)).unwrap();
    let action = workflow_yaml::parse(&source).unwrap();
    let workflow_yaml::Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    steps.clone()
}

fn field<'a>(node: &'a workflow_yaml::Node, key: &str) -> &'a str {
    node.get(key).and_then(workflow_yaml::Node::text).unwrap()
}

#[test]
fn preinstalled_profile_requires_self_hosted_macos_and_each_bootstrap_tool() {
    let steps = action_steps(".github/actions/prepare-automation/action.yml");
    let check = field(&steps[0], "run");
    let tools = ["cargo", "rustc", "just", "sccache", "brew", "xcrun"];
    for missing in std::iter::once(None).chain(tools.iter().copied().map(Some)) {
        let f = Fixture::new();
        for tool in tools {
            if Some(tool) != missing {
                // Presence checks must not invoke the tool or its installer.
                executable(&f.root.join("bin").join(tool), "#!/bin/bash\nexit 93\n");
            }
        }
        let result = f.shell(
            check,
            &[
                ("AUTOMATION_PROFILE", "preinstalled-macos"),
                ("RUNNER_OS", "macOS"),
                ("RUNNER_ENVIRONMENT", "self-hosted"),
                ("GITHUB_ENV", f.root.join("github.env").to_str().unwrap()),
            ],
        );
        assert_eq!(result.code == 0, missing.is_none(), "{}", result.stderr);
        assert!(f.text("github.env").contains("RUSTUP_AUTO_INSTALL=0"));
        if let Some(tool) = missing {
            assert!(result.stderr.contains(&format!(
                "missing preinstalled automation prerequisite: {tool};"
            )));
        }
    }
    for (os, environment) in [
        ("Linux", "self-hosted"),
        ("macOS", "github-hosted"),
        ("Windows", "self-hosted"),
    ] {
        let f = Fixture::new();
        assert_ne!(
            f.shell(
                check,
                &[
                    ("AUTOMATION_PROFILE", "preinstalled-macos"),
                    ("RUNNER_OS", os),
                    ("RUNNER_ENVIRONMENT", environment),
                ]
            )
            .code,
            0
        );
    }
}

#[test]
fn preinstalled_profile_excludes_installers_and_selects_existing_linker_tools() {
    let steps = action_steps(".github/actions/prepare-automation/action.yml");
    for (name, condition) in [
        (
            "Install pinned Rust toolchain",
            "inputs.runner-profile != 'preinstalled-macos' && (runner.os != 'Linux' || inputs.runner-profile == 'hosted-bare')",
        ),
        (
            "Install just",
            "runner.os != 'Linux' && inputs.runner-profile != 'preinstalled-macos'",
        ),
    ] {
        let step = steps
            .iter()
            .find(|step| field(step, "name") == name)
            .unwrap();
        assert_eq!(field(step, "if"), condition);
    }
    let linker = steps
        .iter()
        .find(|step| field(step, "name") == "Prepare macOS linker and compiler cache")
        .unwrap();
    assert_eq!(
        field(linker.get("with").unwrap(), "tool-source"),
        "${{ inputs.runner-profile == 'preinstalled-macos' && 'preinstalled' || 'homebrew' }}"
    );
}

#[test]
fn preinstalled_linker_refuses_missing_tools_and_starts_cache_without_installing() {
    let steps = action_steps(".github/actions/setup-macos-lld/action.yml");
    let script = steps
        .iter()
        .map(|step| field(step, "run"))
        .collect::<Vec<_>>()
        .join("\n");
    for missing in [None, Some("brew"), Some("sccache"), Some("ld64.lld")] {
        let f = Fixture::new();
        if missing != Some("brew") {
            executable(
                &f.root.join("bin/brew"),
                "#!/bin/bash\n[[ \"$*\" == '--prefix lld' ]] || exit 95\nprintf '%s\\n' \"$FIXTURE_ROOT/brew\"\n",
            );
        }
        if missing != Some("sccache") {
            executable(
                &f.root.join("bin/sccache"),
                r#"#!/bin/bash
set -euo pipefail
case "$*" in
 --version) exit 0 ;;
 --start-server) printf '%s' "$SCCACHE_SERVER_UDS" > "$FIXTURE_ROOT/started" ;;
 *) exit 96 ;;
esac
"#,
            );
        }
        if missing != Some("ld64.lld") {
            executable(&f.root.join("brew/bin/ld64.lld"), "#!/bin/bash\nexit 0\n");
        }
        executable(
            &f.root.join("scripts/cargo-linker"),
            r#"#!/bin/bash
[[ "$*" == --mesh-probe ]] || exit 97
[[ "$SCCACHE_SERVER_UDS" == "$(cat "$FIXTURE_ROOT/started")" ]] || exit 98
"#,
        );
        let env = f.root.join("github.env");
        let path = f.root.join("github.path");
        let result = f.shell(
            &script,
            &[
                ("TOOL_SOURCE", "preinstalled"),
                ("SCCACHE_MULTILEVEL_CHAIN", "disk"),
                ("GITHUB_ENV", env.to_str().unwrap()),
                ("GITHUB_PATH", path.to_str().unwrap()),
                ("RUNNER_TEMP", f.root.to_str().unwrap()),
            ],
        );
        assert_eq!(result.code == 0, missing.is_none(), "{}", result.stderr);
        if let Some(tool) = missing {
            assert!(
                result
                    .stderr
                    .contains(&format!("missing preinstalled build prerequisite: {tool};"))
            );
            assert!(!f.root.join("started").exists());
        } else {
            let endpoint = f.text("started");
            assert!(
                f.text("github.env")
                    .contains(&format!("SCCACHE_SERVER_UDS={endpoint}"))
            );
            fs::remove_dir(std::path::Path::new(&endpoint).parent().unwrap()).unwrap();
        }
    }
}
