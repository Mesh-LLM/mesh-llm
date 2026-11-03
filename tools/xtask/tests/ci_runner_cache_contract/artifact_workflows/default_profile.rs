//! Classify default and nested automation preparation paths from checked-in source.
//!
//! This is not an observation of a runner or proof of provider qualification.
use super::{Node, document, input, job, named, steps, text, workflow_yaml};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

const PREFIX: &str = "./.github/actions/";
const PREPARE: &str = "prepare-automation";
const CPU_IMAGE: &str = "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:f499b79bc52dc7492d57397fdbec9f890c6f6bb1d8c1fcde9c1c97d45c0541a7";
const STABILITY_IMAGE: &str = "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:ff53048ee7a87b3244ccbfa1b2888fde5db31a73d0dfa42f283d25dde0115307";

type ActionMap = BTreeMap<String, Node>;
type Calls = BTreeMap<String, BTreeSet<String>>;
type Edge = (String, String, String, String);

fn local_action(step: &Node) -> Option<&str> {
    text(step, "uses")?.strip_prefix(PREFIX)
}

fn actions() -> ActionMap {
    let directory = super::support::root().join(".github/actions");
    let mut result = BTreeMap::new();
    for entry in fs::read_dir(directory).unwrap() {
        let entry = entry.unwrap();
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name().into_string().unwrap();
        let yaml = entry.path().join("action.yaml");
        let yml = entry.path().join("action.yml");
        assert!(
            !(yaml.is_file() && yml.is_file()),
            "ambiguous action {name}"
        );
        let path = if yml.is_file() { yml } else { yaml };
        if !path.is_file() {
            continue;
        }
        let source = fs::read_to_string(path).unwrap();
        if source.contains(PREFIX) || name == PREPARE {
            result.insert(name, workflow_yaml::parse(&source).unwrap());
        }
    }
    result
}

fn action_steps(action: &Node) -> &[Node] {
    match action.get("runs").and_then(|run| run.get("steps")) {
        Some(Node::Seq(items)) => items,
        _ => &[],
    }
}

fn closure(actions: &ActionMap) -> BTreeSet<String> {
    let mut closure = BTreeSet::from([PREPARE.to_owned()]);
    loop {
        let before = closure.len();
        for (owner, action) in actions {
            if action_steps(action)
                .iter()
                .filter_map(local_action)
                .any(|target| closure.contains(target))
            {
                closure.insert(owner.clone());
            }
        }
        if before == closure.len() {
            return closure;
        }
    }
}

fn edges(actions: &ActionMap, closure: &BTreeSet<String>) -> BTreeSet<Edge> {
    let mut result = BTreeSet::new();
    for (owner, action) in actions {
        if !closure.contains(owner) {
            continue;
        }
        for step in action_steps(action) {
            let Some(target) = local_action(step).filter(|name| closure.contains(*name)) else {
                continue;
            };
            let row = (
                owner.clone(),
                target.to_owned(),
                text(step, "if").unwrap_or("").to_owned(),
                input(step, "runner-profile").unwrap_or("").to_owned(),
            );
            assert!(
                result.insert(row),
                "duplicate local profile edge in {owner}"
            );
        }
    }
    result
}

fn expected_edges() -> BTreeSet<Edge> {
    [
        ("compose-product-input", PREPARE, "runner.os != 'Linux' && env.MESH_LLM_AUTOMATION_BIN == ''", ""),
        ("prepare-native-runtime-input", PREPARE, "runner.os != 'Linux' && env.MESH_LLM_AUTOMATION_BIN == ''", ""),
        ("restore-smoke-inputs", PREPARE, "runner.os != 'Linux' || job.container.id != ''", ""),
        ("restore-smoke-inputs", PREPARE, "runner.os == 'Linux' && job.container.id == '' && runner.environment == 'github-hosted' && env.MESH_LLM_AUTOMATION_BIN == ''", "hosted-bare"),
        ("run-laya-product-smoke", "restore-smoke-inputs", "${{ inputs.device != 'Vulkan0' || inputs.enable_vulkan_inference == 'true' }}", ""),
        ("upload-automation", PREPARE, "runner.os == 'Windows' || inputs.runner-profile == 'hosted-bare' || (runner.os == 'macOS' && (inputs.source-profile == 'protected-clean' || inputs.source-profile == 'product-clean'))", "${{ inputs.runner-profile }}"),
    ]
    .into_iter()
    .map(|(a, b, c, d)| (a.to_owned(), b.to_owned(), c.to_owned(), d.to_owned()))
    .collect()
}

fn workflow_calls(closure: &BTreeSet<String>) -> (Calls, BTreeSet<String>) {
    let mut calls = Calls::new();
    let mut defaults = BTreeSet::new();
    let directory = super::support::root().join(".github/workflows");
    for entry in fs::read_dir(directory).unwrap() {
        let file = entry.unwrap().file_name().into_string().unwrap();
        if !file.ends_with(".yml") && !file.ends_with(".yaml") {
            continue;
        }
        let workflow = document(&file);
        for (name, definition) in workflow.get("jobs").unwrap().entries() {
            let Some(Node::Seq(job_steps)) = definition.get("steps") else {
                continue;
            };
            for step in job_steps {
                let Some(action) = local_action(step).filter(|action| closure.contains(*action))
                else {
                    continue;
                };
                let path = format!("{file}/{name}");
                assert!(
                    calls
                        .entry(action.to_owned())
                        .or_default()
                        .insert(path.clone()),
                    "duplicate {action} in {path}"
                );
                if action == PREPARE && input(step, "runner-profile").is_none() {
                    defaults.insert(path);
                }
            }
        }
    }
    (calls, defaults)
}

fn paths(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn default_image_paths_have_explicit_platform_and_container_boundaries() {
    let actions = actions();
    let profile = actions[PREPARE]
        .get("inputs")
        .unwrap()
        .get("runner-profile")
        .unwrap();
    assert_eq!(text(profile, "default"), Some("image"));
    let verification = named(
        actions[PREPARE].get("runs").unwrap(),
        "Verify prebuilt Linux automation environment",
    );
    assert_eq!(
        text(verification, "if"),
        Some("runner.os == 'Linux' && inputs.runner-profile == 'image'")
    );
    assert_eq!(
        text(verification, "run"),
        Some("verify-runner-image public cpu")
    );

    let (_, defaults) = workflow_calls(&closure(&actions));
    assert_eq!(
        defaults,
        paths(&[
            "ci-platform-checks-slice.yml/platform_checks",
            "ci-ui-artifact-slice.yml/ui_stamp",
            "native-sdk-artifact.yml/macos_native_sdk_artifact",
            "nightly-stability-run.yml/stability",
            "release.yml/windows_host_input",
            "smoke.yml/smoke_tests_macos",
            "static-abi-artifact.yml/static_abi_artifact",
            "swift-sdk-artifact.yml/swift_sdk_target",
            "windows-warm-caches.yml/warm_windows_cpu",
            "windows-warm-caches.yml/warm_windows_gpu",
        ])
    );

    for (file, name, runner, image) in [
        (
            "ci-ui-artifact-slice.yml",
            "ui_stamp",
            "${{ needs.runner_policy.outputs.runner_4 }}",
            Some(CPU_IMAGE),
        ),
        (
            "nightly-stability-run.yml",
            "stability",
            "ubuntu-24.04",
            Some(STABILITY_IMAGE),
        ),
        (
            "static-abi-artifact.yml",
            "static_abi_artifact",
            "${{ needs.runner_policy.outputs.runner }}",
            Some(CPU_IMAGE),
        ),
        (
            "native-sdk-artifact.yml",
            "macos_native_sdk_artifact",
            "${{ needs.runner_policy.outputs.runner }}",
            None,
        ),
        (
            "swift-sdk-artifact.yml",
            "swift_sdk_target",
            "${{ needs.runner_policy.outputs.runner_macos }}",
            None,
        ),
        ("smoke.yml", "smoke_tests_macos", "macos-15", None),
        ("release.yml", "windows_host_input", "windows-2022", None),
        (
            "windows-warm-caches.yml",
            "warm_windows_cpu",
            "windows-2022",
            None,
        ),
        (
            "windows-warm-caches.yml",
            "warm_windows_gpu",
            "windows-2022",
            None,
        ),
    ] {
        let workflow = document(file);
        let job = job(&workflow, name);
        assert_eq!(text(job, "runs-on"), Some(runner), "{file}/{name}");
        assert_eq!(
            job.get("container").and_then(|value| text(value, "image")),
            image,
            "{file}/{name}"
        );
    }
    let platform = document("ci-platform-checks-slice.yml");
    let platform_job = job(&platform, "platform_checks");
    let prep = steps(platform_job)
        .iter()
        .find(|step| local_action(step) == Some(PREPARE))
        .unwrap();
    assert_eq!(
        text(prep, "if"),
        Some("${{ matrix.check.kind == 'unit' && matrix.check.platform == 'windows' }}")
    );
    assert_eq!(
        text(platform_job, "runs-on"),
        Some(
            "${{ fromJSON(needs.runner_policy.outputs.runner_by_platform)[matrix.check.platform] }}"
        )
    );
    let mapping = named(
        job(&platform, "runner_policy"),
        "Map platform runners explicitly",
    );
    assert_eq!(
        text(mapping.get("env").unwrap(), "RUNNER_WINDOWS"),
        Some("${{ steps.policy.outputs.runner_windows }}")
    );
    assert!(text(mapping, "run").unwrap().contains("\"windows\":\"%s\""));
    assert!(
        text(mapping, "run")
            .unwrap()
            .contains("\"$RUNNER_MACOS\" \"$RUNNER_WINDOWS\"")
    );
    let native = document("native-sdk-artifact.yml");
    assert_eq!(
        text(job(&native, "macos_native_sdk_artifact"), "if"),
        Some(
            "${{ needs.runner_policy.result == 'success' && endsWith(inputs.target, '-apple-darwin') }}"
        )
    );
    let resolver = named(
        job(&native, "runner_policy"),
        "Resolve runner size and target",
    );
    let script = text(resolver, "run").unwrap();
    assert!(script.contains("aarch64-apple-darwin)"));
    assert!(script.contains("runner=\"$RUNNER_MACOS\""));
    assert!(script.contains("runner=macos-15"));
}

#[test]
fn nested_profile_edges_and_local_action_callers_are_declared() {
    let actions = actions();
    let closure = closure(&actions);
    assert_eq!(
        closure,
        paths(&[
            PREPARE,
            "compose-product-input",
            "prepare-native-runtime-input",
            "restore-smoke-inputs",
            "run-laya-product-smoke",
            "upload-automation"
        ])
    );
    assert_eq!(edges(&actions, &closure), expected_edges());
    let (mut calls, _) = workflow_calls(&closure);
    calls.remove(PREPARE); // Direct explicit and default paths have their own checks.
    let expected = [
        (
            "compose-product-input",
            &[
                "ci-linux-product-slice.yml/linux_product",
                "ci-macos-product-slice.yml/macos_product",
                "ci-windows-product-slice.yml/windows_product",
                "release.yml/compose_cpu_products",
                "release.yml/compose_linux_aarch64_cuda",
                "release.yml/compose_linux_arm64_cpu",
                "release.yml/compose_linux_cuda",
                "release.yml/compose_linux_rocm",
                "release.yml/compose_linux_vulkan",
                "release.yml/compose_windows_cpu",
                "release.yml/compose_windows_gpu",
            ][..],
        ),
        (
            "prepare-native-runtime-input",
            &[
                "ci-linux-runtime-slice.yml/linux_runtime",
                "ci-macos-runtime-slice.yml/macos_runtime",
                "ci-windows-runtime-slice.yml/windows_runtime",
                "depot-canary.yml/runtime_seed",
                "release.yml/build_native_runtime",
                "release.yml/build_native_runtime_linux_aarch64_cuda",
                "release.yml/build_native_runtime_linux_x86_64_cuda",
                "release.yml/build_native_runtime_linux_x86_64_rocm",
                "release.yml/build_native_runtime_linux_x86_64_vulkan",
                "release.yml/build_native_runtime_windows_cpu",
                "release.yml/build_native_runtime_windows_gpu",
                "windows-warm-caches.yml/warm_windows_cpu",
                "windows-warm-caches.yml/warm_windows_gpu",
            ][..],
        ),
        (
            "restore-smoke-inputs",
            &[
                "scripted-binary-smoke.yml/scripted_binary_smoke",
                "sdk-smoke.yml/sdk_smoke",
                "smoke.yml/smoke_tests",
                "smoke.yml/smoke_tests_macos",
            ][..],
        ),
        (
            "run-laya-product-smoke",
            &[
                "ci-linux-product-smoke-slice.yml/laya_cpu",
                "ci-linux-product-smoke-slice.yml/laya_cuda",
                "ci-linux-product-smoke-slice.yml/laya_rocm",
                "ci-linux-product-smoke-slice.yml/laya_vulkan",
                "ci-macos-product-smoke-slice.yml/laya_metal",
                "ci-windows-product-smoke-slice.yml/laya_cpu",
            ][..],
        ),
        (
            "upload-automation",
            &[
                "ci-governance-administration.yml/prepare",
                "ci-governance-publisher.yml/resolve",
                "ci-quality-slice.yml/runner_policy",
                "depot-canary.yml/authority_automation",
                "product-automation-artifact.yml/producer",
                "protected-automation-artifact.yml/producer",
                "release.yml/build",
                "release.yml/build_linux_arm64",
                "release.yml/windows_host_input",
            ][..],
        ),
    ]
    .into_iter()
    .map(|(name, values)| (name.to_owned(), paths(values)))
    .collect::<Calls>();
    assert_eq!(
        calls, expected,
        "new local consumers require a profile decision"
    );

    // Bare self-hosted smoke may fall back to a locally provisioned Cargo.
    // Keep this explicit; it is not proof those tools or the runner are qualified.
    let smoke = document("smoke.yml");
    let gpu = job(&smoke, "smoke_tests");
    assert!(text(gpu, "runs-on").unwrap().contains("gpu-nvidia"));
    assert!(
        text(gpu.get("container").unwrap(), "image")
            .unwrap()
            .contains("inputs.runner != 'gpu-nvidia'")
    );
    let restore = &actions["restore-smoke-inputs"];
    let extract = named(
        restore.get("runs").unwrap(),
        "Extract and verify composed product",
    );
    assert!(
        text(extract, "run")
            .unwrap()
            .contains("automation=(cargo xtool)")
    );

    let laya = document("ci-linux-product-smoke-slice.yml");
    for name in ["laya_cuda", "laya_rocm", "laya_vulkan"] {
        let items = steps(job(&laya, name));
        let bootstrap = items
            .iter()
            .position(|step| {
                text(step, "name")
                    == Some("Prepare source-bound native cleanup before managed work")
            })
            .unwrap();
        let admission = items
            .iter()
            .position(|step| {
                text(step, "name") == Some("Admit cleanup controller before managed work")
            })
            .unwrap();
        let smoke = items
            .iter()
            .position(|step| local_action(step) == Some("run-laya-product-smoke"))
            .unwrap();
        assert!(bootstrap < admission && admission < smoke, "{name}");
        let script = text(&items[bootstrap], "run").unwrap();
        assert!(script.contains("just automation-bootstrap"));
        assert!(script.contains("sha256sum"));
        assert!(script.contains("MESH_LLM_AUTOMATION_BIN="));
        assert!(
            text(&items[admission], "run")
                .unwrap()
                .contains("MESH_NATIVE_CLEANUP_SHA")
        );
    }

    let original = fs::read_to_string(
        super::support::root().join(".github/actions/restore-smoke-inputs/action.yml"),
    )
    .unwrap();
    let changed = original.replace(
        "runner.environment == 'github-hosted'",
        "runner.environment == 'self-hosted'",
    );
    assert_ne!(original, changed);
    let mut mutated = actions.clone();
    mutated.insert(
        "restore-smoke-inputs".to_owned(),
        workflow_yaml::parse(&changed).unwrap(),
    );
    assert_ne!(
        edges(&mutated, &closure),
        expected_edges(),
        "hosted-bare must reject a self-hosted mutation"
    );
}

#[test]
fn reusable_profile_entrypoints_have_declared_source_boundaries() {
    let actions = actions();
    let (calls, _) = workflow_calls(&closure(&actions));
    let mut affected = calls
        .values()
        .flat_map(|values| values.iter())
        .map(|path| path.split('/').next().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    let mut workflows = BTreeMap::new();
    for entry in fs::read_dir(super::support::root().join(".github/workflows")).unwrap() {
        let file = entry.unwrap().file_name().into_string().unwrap();
        if file.ends_with(".yml") || file.ends_with(".yaml") {
            workflows.insert(file.clone(), document(&file));
        }
    }
    loop {
        let before = affected.len();
        for (file, doc) in &workflows {
            for (_, definition) in doc.get("jobs").unwrap().entries() {
                if text(definition, "uses")
                    .and_then(|s| s.strip_prefix("./.github/workflows/"))
                    .is_some_and(|target| affected.contains(target))
                {
                    affected.insert(file.clone());
                }
            }
        }
        if affected.len() == before {
            break;
        }
    }
    let mut actual = BTreeSet::new();
    for (file, doc) in &workflows {
        for (name, definition) in doc.get("jobs").unwrap().entries() {
            let Some(uses) = text(definition, "uses") else {
                continue;
            };
            if let Some(target) = uses.strip_prefix("./.github/workflows/") {
                if affected.contains(target) {
                    assert!(actual.insert(format!("{file}/{name} -> {target}")));
                }
            } else if uses
                .to_ascii_lowercase()
                .starts_with("mesh-llm/mesh-llm/.github/workflows/")
            {
                let target = &uses["mesh-llm/mesh-llm/.github/workflows/".len()..];
                let (path, revision) = target
                    .split_once('@')
                    .expect("repository workflow revision");
                if affected.contains(path) {
                    assert!(actual.insert(format!("{file}/{name} -> {path}@{revision}")));
                }
            }
        }
    }
    let roster = include_str!("profile_reusable_paths.txt");
    let expected = roster.lines().map(str::to_owned).collect::<BTreeSet<_>>();
    assert!(
        !expected.is_empty(),
        "profile path declarations cannot be empty"
    );
    assert_eq!(
        expected.len(),
        roster.lines().count(),
        "duplicate profile path declaration"
    );
    assert_eq!(
        actual, expected,
        "new reusable paths require a profile decision"
    );
}
