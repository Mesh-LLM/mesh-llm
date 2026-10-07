//! Parsed declarations only: executable native owners separately qualify behavior.
use super::super::{
    support,
    workflow_yaml::{self, Node},
};
use std::{collections::BTreeMap, fs};
fn source(path: &str) -> String {
    fs::read_to_string(support::root().join(path)).unwrap()
}
fn doc(path: &str) -> Node {
    workflow_yaml::parse(&source(path)).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(items) = node.get("steps").unwrap() else {
        panic!("steps sequence")
    };
    items
}
fn action_steps(node: &Node) -> &[Node] {
    steps(node.get("runs").unwrap())
}
fn input<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get("with").and_then(|v| text(v, key))
}
fn required_tokens(source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(source.contains(token), "missing declared contract: {token}");
    }
}
fn step_id<'a>(items: &'a [Node], id: &str) -> &'a Node {
    let matches = items
        .iter()
        .filter(|s| text(s, "id") == Some(id))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    matches[0]
}
fn workflow_steps(node: &Node) -> Vec<&Node> {
    node.get("jobs")
        .unwrap()
        .entries()
        .iter()
        .flat_map(|(_, job)| {
            job.get("steps").map_or(&[][..], |value| {
                let Node::Seq(items) = value else {
                    panic!("job steps")
                };
                items.as_slice()
            })
        })
        .collect()
}
fn action_calls(action: &str) -> BTreeMap<String, Vec<Node>> {
    let mut observed = BTreeMap::new();
    let marker = format!("./.github/actions/{action}");
    let mut paths = fs::read_dir(support::root().join(".github/workflows"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|v| v == "yml"))
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let document = workflow_yaml::parse(&fs::read_to_string(&path).unwrap()).unwrap();
        let calls = workflow_steps(&document)
            .into_iter()
            .filter(|step| text(step, "uses") == Some(marker.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if !calls.is_empty() {
            observed.insert(
                path.file_name().unwrap().to_str().unwrap().to_owned(),
                calls,
            );
        }
    }
    observed
}
#[test]
fn windows_abi_cache_declares_exact_compatibility_identity_and_restore_outputs() {
    let action = doc(".github/actions/restore-windows-abi-cache/action.yml");
    let inputs = action.get("inputs").unwrap();
    for key in [
        "backend",
        "build_dir",
        "toolchain_epoch",
        "architecture_set",
        "cuda_toolchain_version",
        "vulkan_toolchain_version",
        "rocm_toolchain_version",
    ] {
        assert!(inputs.get(key).is_some());
    }
    let identity = step_id(action_steps(&action), "identity");
    assert_eq!(text(identity, "shell"), Some("pwsh"));
    let env = identity.get("env").unwrap();
    for (key, input) in [
        ("INPUT_BACKEND", "backend"),
        ("INPUT_BUILD_DIR", "build_dir"),
        ("INPUT_TOOLCHAIN_EPOCH", "toolchain_epoch"),
        ("INPUT_ARCHITECTURE_SET", "architecture_set"),
        ("INPUT_CUDA_TOOLCHAIN_VERSION", "cuda_toolchain_version"),
        ("INPUT_VULKAN_TOOLCHAIN_VERSION", "vulkan_toolchain_version"),
        ("INPUT_ROCM_TOOLCHAIN_VERSION", "rocm_toolchain_version"),
    ] {
        assert_eq!(
            text(env, key),
            Some(format!("${{{{ inputs.{input} }}}}").as_str())
        );
    }
    required_tokens(
        text(env, "CACHE_INPUT_HASH").unwrap(),
        &[
            "restore-windows-abi-cache/action.yml",
            "save-and-verify-actions-cache/action.yml",
            "resolve-native-toolchain-epoch/action.yml",
            "prepare-native-runtime-input/action.yml",
            "setup-windows-rocm-sdk/action.yml",
            "scripts/build-llama.sh",
            "skippy/scripts/build-llama.sh",
            "scripts/prepare-llama.sh",
            "skippy/scripts/prepare-llama.sh",
            "scripts/package-native-runtime.sh",
            "skippy/scripts/package-native-runtime.sh",
            "skippy/llama_cpp/upstream.txt",
            "skippy/llama_cpp/patches/**",
            ".github/cache-version.txt",
        ],
    );
    required_tokens(
        text(identity, "run").unwrap(),
        &[
            "$backend -notin @(\"cpu\", \"cuda\", \"rocm\", \"vulkan\")",
            "$toolchainEpoch -ne $buildStampEpoch",
            "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
            "build_dir must resolve inside GITHUB_WORKSPACE",
            "build_dir must remain outside the replaceable llama.cpp",
            "$resolvedBuildDir -eq $llamaWorktree",
            "$resolvedBuildDir.StartsWith(",
            "$backend -in @(\"cuda\", \"rocm\") -and -not $architectureSet",
            "cuda-$version-Jimver-v0.2.35",
            "vulkan-$version-jakoch-v1.5.2",
            "rocm-$version",
            "mesh-llm-windows-2022-skippy-abi-$backend-$architectureSet-$toolchain-$toolchainEpoch-$inputHash",
            "^[0-9a-f]{64}$",
        ],
    );
    let restore = step_id(action_steps(&action), "restore");
    assert_eq!(
        text(restore, "uses"),
        Some("actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25")
    );
    assert_eq!(
        text(restore, "if"),
        Some("${{ inputs.allow-native-github-cache == 'true' }}")
    );
    assert_eq!(
        input(restore, "path"),
        Some("${{ steps.identity.outputs.build-dir }}")
    );
    assert_eq!(
        input(restore, "key"),
        Some("${{ steps.identity.outputs.cache-key }}")
    );
    assert!(restore.get("with").unwrap().get("restore-keys").is_none());
    for (key, value) in [
        ("cache-hit", "${{ steps.restore.outputs.cache-hit }}"),
        (
            "cache-primary-key",
            "${{ steps.restore.outputs.cache-primary-key }}",
        ),
        ("cache-path", "${{ steps.identity.outputs.build-dir }}"),
    ] {
        assert_eq!(
            text(action.get("outputs").unwrap().get(key).unwrap(), "value"),
            Some(value)
        );
    }
}
#[test]
fn windows_native_cache_callers_declare_fail_closed_defaults_and_exact_opt_in() {
    for (action, expected) in [
        ("restore-windows-abi-cache", [1, 2, 2]),
        ("setup-windows-rocm-sdk", [1, 1, 1]),
    ] {
        let document = doc(&format!(".github/actions/{action}/action.yml"));
        let flag = document
            .get("inputs")
            .unwrap()
            .get("allow-native-github-cache")
            .unwrap();
        assert_eq!(text(flag, "required"), Some("false"));
        assert_eq!(text(flag, "default"), Some("false"));
        let calls = action_calls(action);
        let names = [
            "ci-windows-runtime-slice.yml",
            "release.yml",
            "windows-warm-caches.yml",
        ];
        assert_eq!(calls.len(), names.len());
        for (name, count) in names.into_iter().zip(expected) {
            let observed = calls.get(name).unwrap();
            assert_eq!(observed.len(), count, "{name} {action}");
            for call in observed {
                assert_eq!(
                    input(call, "allow-native-github-cache"),
                    Some(if name == "ci-windows-runtime-slice.yml" {
                        "${{ needs.runner_policy.outputs.allow_native_github_cache }}"
                    } else {
                        "true"
                    })
                );
            }
        }
    }
}
#[test]
fn epoch_consumers_declare_same_build_stamp_and_cache_identity() {
    for name in [
        "ci-linux-runtime-slice",
        "static-abi-artifact",
        "native-sdk-artifact",
        "swift-sdk-artifact",
        "release",
        "windows-warm-caches",
    ] {
        let document = doc(&format!(".github/workflows/{name}.yml"));
        let steps = workflow_steps(&document);
        assert!(
            steps.iter().any(|step| text(step, "uses")
                == Some("./.github/actions/resolve-native-toolchain-epoch")),
            "{name}"
        );
        for step in steps {
            if !text(step, "uses").is_some_and(|value| value.starts_with("actions/cache@")) {
                continue;
            }
            if input(step, "path").is_some_and(|path| path.contains("LLAMA_STAGE_BUILD_DIR")) {
                assert!(
                    input(step, "key")
                        .unwrap()
                        .contains("native_toolchain.outputs.epoch"),
                    "{name}"
                );
            }
        }
    }
    let resolver = doc(".github/actions/resolve-native-toolchain-epoch/action.yml");
    let run = action_steps(&resolver)
        .iter()
        .find_map(|step| text(step, "run"))
        .unwrap();
    required_tokens(
        run,
        &[
            "image_os=\"${ImageOS:-}\"",
            "image_version=\"${ImageVersion:-}\"",
            "epoch=\"runner-${RUNNER_OS_VALUE}-${RUNNER_ARCH_VALUE}\"",
            "echo \"epoch=$epoch\" >> \"$GITHUB_OUTPUT\"",
            "echo \"MESH_LLM_LLAMA_TOOLCHAIN_EPOCH=$epoch\" >> \"$GITHUB_ENV\"",
        ],
    );
}
#[test]
fn release_recipe_declarations_include_all_three_product_primitives() {
    let run = source(".github/actions/compute-changes/derive-outputs.sh");
    let region = run
        .split_once("function is_backend_recipe(name)")
        .unwrap()
        .1
        .split_once('}')
        .unwrap()
        .0;
    let allowed = region
        .split_once("return name ~ /^(")
        .unwrap()
        .1
        .split_once(")$/")
        .unwrap()
        .0
        .split('|')
        .collect::<Vec<_>>();
    for name in [
        "release-host-build",
        "release-runtime-build",
        "skippy-cli-release-build",
    ] {
        assert!(allowed.contains(&name), "{name}");
    }
}
fn smoke_download(name: &str, artifact: &str, kind: &str) {
    let document = doc(".github/workflows/sdk-smoke.yml");
    let matches = workflow_steps(&document)
        .into_iter()
        .filter(|step| text(step, "name") == Some(name))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    let step = matches[0];
    assert_eq!(
        text(step, "uses"),
        Some("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c")
    );
    assert_eq!(input(step, "name"), Some(artifact));
    assert_eq!(
        text(step, "if"),
        Some(format!("${{{{ inputs.sdk_kind == '{kind}' }}}}").as_str())
    );
}
fn compiler_free(script: &str) {
    for forbidden in [
        "cargo ",
        "prepare-llama.sh",
        "build-llama.sh",
        "package-native-sdk.sh",
        "build-xcframework.sh",
        "build-host-macos-xcframework.sh",
    ] {
        assert!(
            !script.contains(forbidden),
            "unexpected native compilation entrypoint {forbidden}"
        );
    }
}
#[test]
fn kotlin_smoke_declares_immutable_download_restore_and_compiler_free_consumer() {
    native_sdk_source_checkout_and_stats();
    smoke_download(
        "Download immutable Kotlin native SDK input",
        "${{ inputs.kotlin_artifact_name }}",
        "kotlin",
    );
    let script = source("scripts/ci-kotlin-sdk-smoke.sh");
    compiler_free(&script);
    required_tokens(
        &script,
        &[
            "scripts/restore-native-sdk-input.sh",
            "prepared-input native-sdk-library-dir",
            "--reuse-from-binary",
        ],
    );
    let restore = source("scripts/restore-native-sdk-input.sh");
    required_tokens(
        &restore,
        &[
            "mesh_automation artifact extract-tar",
            "prepared-input native-sdk-identity",
            "scripts/verify-native-sdk-package.sh",
        ],
    );
    assert!(!restore.contains("tar -x"));
}
#[test]
fn static_abi_restore_declares_typed_admission_and_exact_epoch_before_publication() {
    let restore = source("scripts/restore-static-abi-input.sh");
    required_tokens(
        &restore,
        &[
            "artifact extract-tar",
            "artifact verify-checksum",
            "prepared-input static-abi-manifest verify",
            "prepared-input static-abi-stamp",
            "--toolchain-epoch \"$expected_toolchain_epoch\"",
            "target/runner architecture mismatch",
            "-L \"${extract_entries[0]}\"",
            ".mesh-llm-static-abi-input.json",
            ".mesh-llm-build-stamp",
        ],
    );
    let checksum = restore.find("artifact verify-checksum").unwrap();
    let extraction = restore.find("artifact extract-tar").unwrap();
    let identity = restore
        .find("prepared-input static-abi-manifest verify")
        .unwrap();
    let stamp = restore.find("prepared-input static-abi-stamp").unwrap();
    let publish = restore
        .find("cp -a \"$restored_dir\" \"$build_dir\"")
        .unwrap();
    assert!(checksum < extraction && extraction < identity && identity < stamp && stamp < publish);
    assert!(!restore.contains("tar -x"));
}
#[test]
fn swift_smoke_declares_immutable_binding_safe_destination_and_legacy_store_guard() {
    swift_producer_compiler_cache_and_verifier_declarations();
    smoke_download(
        "Download immutable Swift SDK input",
        "${{ inputs.swift_artifact_name }}",
        "swift",
    );
    smoke_download(
        "Download immutable generated Swift binding",
        "generated-swift-binding-${{ inputs.swift_artifact_name }}",
        "swift",
    );
    let workflow = doc(".github/workflows/sdk-smoke.yml");
    for step in workflow_steps(&workflow) {
        assert!(
            !text(step, "uses").is_some_and(|value| value.starts_with("pnpm/action-setup@")
                || value.starts_with("actions/setup-node@"))
        );
    }
    let script = source("scripts/ci-swift-sdk-smoke.sh");
    compiler_free(&script);
    required_tokens(
        &script,
        &[
            "command -v pnpm",
            "LEGACY_PNPM_STORE=\"$(pnpm store path --silent)\"",
            "mkdir -p \"$LEGACY_PNPM_STORE\"",
            "install -m 0644 \"$SWIFT_INPUT_BINDING\" \"$SWIFT_TRACKED_BINDING\"",
            "cmp \"$SWIFT_INPUT_BINDING\" \"$SWIFT_TRACKED_BINDING\"",
            "mesh_automation artifact extract-zip",
            "[[ -L \"$SWIFT_GENERATED_DIR\" ]]",
            "[[ -e \"$SWIFT_GENERATED_DIR\" && ! -d \"$SWIFT_GENERATED_DIR\" ]]",
            "--reuse-from-binary",
        ],
    );
    assert!(
        script.find("mkdir -p \"$SWIFT_GENERATED_DIR\"").unwrap()
            < script
                .find("mv \"$SWIFT_EXTRACT_DIR/MeshLLMFFI.xcframework\" \"$SWIFT_XCFRAMEWORK\"")
                .unwrap()
    );
}
#[test]
fn swift_host_builder_declares_target_specific_cache_and_all_cmake_platform_arguments() {
    let script = source("mesh/sdk/swift/scripts/build-host-macos-xcframework.sh");
    required_tokens(
        &script,
        &[
            ".deps/llama-build/build-stage-abi-$RUST_TARGET-metal",
            "-DCMAKE_OSX_SYSROOT=macosx",
            "-DCMAKE_OSX_ARCHITECTURES=\"$CMAKE_ARCH\"",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=\"$MACOSX_DEPLOYMENT_TARGET\"",
        ],
    );
    assert!(!script.contains("build-stage-abi-host-metal"));
}
fn assert_hosted_cpu_selector(step: &Node) {
    assert_eq!(text(step, "id"), Some("cpu_policy"));
    let bindings = BTreeMap::from([
        ("event_name", "${{ github.event_name }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        ("repository", "${{ github.repository }}"),
        (
            "head_repository",
            "${{ github.event.pull_request.head.repo.full_name }}",
        ),
        (
            "head_sha",
            "${{ github.event.pull_request.head.sha || github.sha }}",
        ),
        ("ref", "${{ github.ref }}"),
        (
            "depot_main_enabled",
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
        ),
        (
            "depot_pr_enabled",
            "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
        ),
        ("pr_canary_ref", "${{ vars.DEPOT_PR_CANARY_REF }}"),
        ("force_hosted", "true"),
    ]);
    let actual = step
        .get("with")
        .unwrap()
        .entries()
        .iter()
        .map(|(key, value)| (key.as_str(), value.text().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(actual, bindings);
}
#[test]
fn selector_call_census_declares_exact_head_and_paired_approval_inputs() {
    let calls = action_calls("select-ci-runners");
    let total = calls.values().map(Vec::len).sum::<usize>();
    assert_eq!(total, 20);
    let expected = BTreeMap::from([
        ("ci-quality-slice.yml", 2),
        ("ci-linux-product-slice.yml", 1),
        ("native-sdk-artifact.yml", 1),
        ("ci-windows-runtime-slice.yml", 1),
        ("ci-linux-host-slice.yml", 1),
        ("ci-ui-artifact-slice.yml", 1),
        ("release.yml", 1),
        ("ci-macos-runtime-slice.yml", 1),
        ("ci-windows-host-slice.yml", 1),
        ("ci-platform-checks-slice.yml", 1),
        ("static-abi-artifact.yml", 1),
        ("swift-sdk-artifact.yml", 1),
        ("ci-macos-host-slice.yml", 1),
        ("ci-macos-product-slice.yml", 1),
        ("ci-linux-runtime-slice.yml", 2),
        ("ci-windows-product-slice.yml", 1),
        ("ci-web-slice.yml", 1),
        ("ci-rust-tests-slice.yml", 1),
    ]);
    assert_eq!(
        calls
            .iter()
            .map(|(name, steps)| (name.as_str(), steps.len()))
            .collect::<BTreeMap<_, _>>(),
        expected
    );
    let mut approval = 0;
    for (workflow, steps) in calls {
        for step in steps {
            assert!(
                input(&step, "head_sha").is_some_and(|head| !head.is_empty()),
                "{workflow}"
            );
            match (
                input(&step, "pr_approved_ref"),
                input(&step, "pr_approved_sha"),
            ) {
                (Some(_), Some(_)) => approval += 1,
                (None, None) if workflow == "ci-linux-runtime-slice.yml" => {
                    assert_hosted_cpu_selector(&step);
                }
                (None, None) => assert_eq!(workflow, "release.yml"),
                _ => panic!("incomplete approval pair: {workflow}"),
            }
        }
    }
    assert_eq!(approval, 18);
}
fn producer_stats(document: &Node, prefix: &str, count: usize) {
    let stats = workflow_steps(document)
        .into_iter()
        .filter(|step| text(step, "uses") == Some("./.github/actions/capture-sccache-stats"))
        .collect::<Vec<_>>();
    assert_eq!(stats.len(), count);
    for step in stats {
        assert_eq!(text(step, "if"), Some("${{ !cancelled() }}"));
        let name = input(step, "artifact_name").unwrap();
        assert!(
            name.starts_with(prefix) && name.ends_with("-${{ github.run_attempt }}"),
            "{name}"
        );
    }
}
fn native_sdk_source_checkout_and_stats() {
    let document = doc(".github/workflows/native-sdk-artifact.yml");
    assert_eq!(
        text(document.get("env").unwrap(), "RUSTC_WRAPPER"),
        Some("sccache")
    );
    producer_stats(&document, "sccache-native-sdk-", 2);
    let linux = document
        .get("jobs")
        .unwrap()
        .get("linux_native_sdk_artifact")
        .unwrap();
    let items = steps(linux);
    let checkout = items
        .iter()
        .position(|step| text(step, "uses").is_some_and(|v| v.starts_with("actions/checkout@")))
        .unwrap();
    let trust = items
        .iter()
        .position(|step| text(step, "name") == Some("Trust checkout directory"))
        .unwrap();
    let release = items
        .iter()
        .position(|step| text(step, "name") == Some("Prepare dispatched release version"))
        .unwrap();
    assert_eq!(
        text(&items[trust], "run"),
        Some("git config --global --add safe.directory \"$GITHUB_WORKSPACE\"")
    );
    assert!(checkout < trust && trust < release);
}
fn swift_producer_compiler_cache_and_verifier_declarations() {
    let document = doc(".github/workflows/swift-sdk-artifact.yml");
    producer_stats(&document, "sccache-swift-sdk-", 2);
    for name in ["swift_sdk_target", "swift_sdk_artifact"] {
        let job = document.get("jobs").unwrap().get(name).unwrap();
        let env = job.get("env").unwrap();
        assert_eq!(text(env, "RUSTC_WRAPPER"), Some("sccache"));
        required_tokens(
            text(env, "SCCACHE_GHA_RW_MODE").unwrap(),
            &[
                "pull_request",
                "pull_request_target",
                "original_event_name",
                "READ_ONLY",
                "READ_WRITE",
            ],
        );
        assert!(
            steps(job)
                .iter()
                .any(|step| text(step, "uses") == Some("./.github/actions/configure-sccache-gha"))
        );
    }
    assert!(
        source("scripts/verify-swift-release-artifact.sh")
            .contains("/mesh/scripts/verify-swift-release-artifact.sh\" \"$@\"")
    );
    let verifier = source("mesh/scripts/verify-swift-release-artifact.sh");
    required_tokens(
        &verifier,
        &[
            "source \"$REPO_ROOT/scripts/lib/automation.sh\"",
            "mesh_automation release swift-xcframework \\\n  \"${xcframework_args[@]}\"",
        ],
    );
}
#[test]
fn quality_declares_noninteractive_dependency_install_and_refuses_unverified_actionlint_shims() {
    let document = doc(".github/workflows/ci-quality-slice.yml");
    let quality = document
        .get("jobs")
        .unwrap()
        .get("quality_contracts")
        .unwrap();
    let items = steps(quality);
    assert!(items.iter().all(|step| {
        !text(step, "uses").is_some_and(|value| value.starts_with("actions/setup-python@"))
            && !text(step, "run").is_some_and(|run| {
                run.contains("requirements-ci-python.txt") || run.contains("pip install")
            })
    }));
    for step in items {
        assert!(
            !input(step, "tool").is_some_and(|tool| tool.starts_with("actionlint@")),
            "unverified tool shim"
        );
    }
}
