//! Additional assertions from the stronger overlapping epoch/SDK modules.
use super::super::{
    support::{self, Fixture},
    workflow_yaml::{self, Node},
};
use std::{fs, process::Command};
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn document(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(support::root().join(path)).unwrap()).unwrap()
}
fn epoch_cache_jobs_valid(document: &Node) -> bool {
    let Some(jobs) = document.get("jobs") else {
        return false;
    };
    for (_, job) in jobs.entries() {
        let Some(steps) = job.get("steps") else {
            continue;
        };
        let Node::Seq(steps) = steps else {
            return false;
        };
        let epochs = steps
            .iter()
            .filter(|step| {
                text(step, "uses") == Some("./.github/actions/resolve-native-toolchain-epoch")
                    && text(step, "id") == Some("native_toolchain")
            })
            .count();
        for step in steps {
            if !text(step, "uses").is_some_and(|uses| uses.starts_with("actions/cache@")) {
                continue;
            }
            let Some(inputs) = step.get("with") else {
                return false;
            };
            if epochs != 1
                || !text(inputs, "key")
                    .is_some_and(|key| key.contains("steps.native_toolchain.outputs.epoch"))
            {
                return false;
            }
            if text(inputs, "path")
                .is_some_and(|path| path.contains("${{ env.LLAMA_STAGE_BUILD_DIR }}"))
                && job
                    .get("env")
                    .and_then(|env| env.get("LLAMA_STAGE_BUILD_DIR"))
                    .is_none()
            {
                return false;
            }
        }
    }
    true
}
#[test]
fn every_native_cache_binds_one_epoch_in_its_own_job_and_declares_build_directory() {
    for workflow in [
        "ci-linux-runtime-slice",
        "static-abi-artifact",
        "native-sdk-artifact",
        "swift-sdk-artifact",
        "release",
    ] {
        assert!(
            epoch_cache_jobs_valid(&document(&format!(".github/workflows/{workflow}.yml"))),
            "{workflow}"
        );
    }
    let valid = "jobs:\n  native:\n    env:\n      LLAMA_STAGE_BUILD_DIR: build\n    steps:\n      - uses: ./.github/actions/resolve-native-toolchain-epoch\n        id: native_toolchain\n      - uses: actions/cache@fixture\n        with:\n          key: ${{ steps.native_toolchain.outputs.epoch }}\n          path: ${{ env.LLAMA_STAGE_BUILD_DIR }}\n";
    assert!(epoch_cache_jobs_valid(
        &workflow_yaml::parse(valid).unwrap()
    ));
    for (from, to) in [
        ("id: native_toolchain", "id: unrelated"),
        (
            "steps.native_toolchain.outputs.epoch",
            "steps.other.outputs.epoch",
        ),
        ("LLAMA_STAGE_BUILD_DIR: build", "OTHER_DIR: build"),
        (
            "      - uses: actions/cache@fixture",
            "      - uses: ./.github/actions/resolve-native-toolchain-epoch\n        id: native_toolchain\n      - uses: actions/cache@fixture",
        ),
    ] {
        assert!(
            !epoch_cache_jobs_valid(&workflow_yaml::parse(&valid.replace(from, to)).unwrap()),
            "{from}"
        );
    }
    let cross_job = "jobs:\n  epoch_owner:\n    steps:\n      - uses: ./.github/actions/resolve-native-toolchain-epoch\n        id: native_toolchain\n  cache_owner:\n    steps:\n      - uses: actions/cache@fixture\n        with:\n          key: ${{ steps.native_toolchain.outputs.epoch }}\n          path: build\n";
    assert!(!epoch_cache_jobs_valid(
        &workflow_yaml::parse(cross_job).unwrap()
    ));
}
#[test]
fn epoch_resolver_declares_one_selected_run_and_all_exact_input_projections() {
    let resolver = document(".github/actions/resolve-native-toolchain-epoch/action.yml");
    let Node::Seq(steps) = resolver.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    let selected = steps
        .iter()
        .filter(|step| {
            text(step, "id") == Some("resolve")
                && text(step, "run").is_some_and(|run| run.contains("echo \"epoch=$epoch\""))
        })
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 1);
    let env = selected[0].get("env").unwrap();
    for (key, value) in [
        ("INPUT_PINNED_EPOCH", "${{ inputs.pinned_epoch }}"),
        (
            "INPUT_INCLUDE_TOOL_VERSIONS",
            "${{ inputs.include_tool_versions }}",
        ),
        ("RUNNER_OS_VALUE", "${{ runner.os }}"),
        ("RUNNER_ARCH_VALUE", "${{ runner.arch }}"),
    ] {
        assert_eq!(text(env, key), Some(value));
    }
    assert_eq!(
        text(
            resolver.get("outputs").unwrap().get("epoch").unwrap(),
            "value"
        ),
        Some("${{ steps.resolve.outputs.epoch }}")
    );
}
#[test]
fn swift_host_architecture_branch_admits_apple_silicon_and_refuses_intel_before_native_build() {
    let source = fs::read_to_string(
        support::root().join("mesh/sdk/swift/scripts/build-host-macos-xcframework.sh"),
    )
    .unwrap();
    assert!(!source.contains("x86_64-apple-darwin"));
    let workflow =
        fs::read_to_string(support::root().join(".github/workflows/swift-sdk-artifact.yml"))
            .unwrap();
    assert!(!workflow.contains("x86_64-apple-darwin"));
    let start = "case \"$HOST_ARCH\" in";
    assert_eq!(source.matches(start).count(), 1);
    let branch = source
        .split_once(start)
        .unwrap()
        .1
        .split_once("\nesac")
        .unwrap()
        .0;
    let script =
        format!("{start}{branch}\nesac\nprintf '%s %s\\n' \"$RUST_TARGET\" \"$CMAKE_ARCH\"\n");
    for arch in ["arm64", "aarch64", "x86_64"] {
        let fixture = Fixture::new();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOST_ARCH", arch)
            .args(["-euo", "pipefail", "-c", &script]);
        let output = fixture.run(command);
        assert_eq!(output.status.success(), arch != "x86_64");
        if arch == "x86_64" {
            assert!(output.stdout.is_empty());
            assert!(
                String::from_utf8(output.stderr)
                    .unwrap()
                    .contains("Unsupported macOS host architecture")
            );
        } else {
            assert_eq!(output.stdout, b"aarch64-apple-darwin arm64\n");
        }
        fixture.0.close().unwrap();
    }
}
