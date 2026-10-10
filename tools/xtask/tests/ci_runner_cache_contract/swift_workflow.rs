//! Swift workflow handoff, admission and binding fixtures; no Xcode or SDK build.
use super::{
    support::{Fixture, root},
    workflow_yaml::{self, Node},
};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    process::{Command, Output},
};
const TARGETS: [&str; 4] = [
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "aarch64-apple-ios-macabi",
    "aarch64-apple-darwin",
];
fn source() -> String {
    fs::read_to_string(root().join(".github/workflows/swift-sdk-artifact.yml")).unwrap()
}
fn workflow() -> Node {
    workflow_yaml::parse(&source()).unwrap()
}
fn job<'a>(document: &'a Node, name: &str) -> &'a Node {
    document.get("jobs").unwrap().get(name).unwrap()
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(values) = node.get("steps").unwrap() else {
        panic!("steps must be a sequence")
    };
    values
}
fn named<'a>(node: &'a Node, name: &str) -> (usize, &'a Node) {
    let matches = steps(node)
        .iter()
        .enumerate()
        .filter(|(_, step)| step.get("name").and_then(Node::text) == Some(name))
        .collect::<Vec<_>>();
    let [(index, step)] = matches.as_slice() else {
        panic!("one {name} step required")
    };
    (*index, *step)
}
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn input(node: &Node, key: &str, expected: &str) -> bool {
    node.get("with").and_then(|n| text(n, key)) == Some(expected)
}
fn handoff(document: &Node) -> bool {
    let target = job(document, "swift_sdk_target");
    let assembly = job(document, "swift_sdk_artifact");
    let matrix = target
        .get("strategy")
        .and_then(|n| n.get("matrix"))
        .and_then(|n| n.get("target"))
        .unwrap()
        .list();
    let (_, upload) = named(target, "Upload immutable Swift target");
    let (_, download) = named(assembly, "Download immutable Swift targets");
    let mut needs = assembly.get("needs").unwrap().list();
    needs.sort_unstable();
    matrix == TARGETS
        && needs == ["runner_policy", "swift_sdk_target"]
        && text(assembly, "if")
            == Some(
                "${{ !cancelled() && needs.runner_policy.result == 'success' && (inputs.mode == 'host-only' || needs.swift_sdk_target.result == 'success') }}",
            )
        && text(target, "if") == Some("${{ inputs.mode == 'full' }}")
        && target.get("strategy").and_then(|n| text(n, "max-parallel"))
            == Some("${{ inputs.max_parallel }}")
        && target.get("strategy").and_then(|n| text(n, "fail-fast"))
            == Some("${{ inputs.fail_fast }}")
        && text(target, "runs-on") == Some("${{ needs.runner_policy.outputs.runner_macos }}")
        && text(assembly, "runs-on") == Some("${{ needs.runner_policy.outputs.runner_macos }}")
        && input(
            upload,
            "name",
            "swift-sdk-target-${{ matrix.target }}-${{ github.run_attempt }}",
        )
        && input(upload, "path", "dist/swift-targets")
        && input(upload, "if-no-files-found", "error")
        && text(download, "if") == Some("${{ inputs.mode == 'full' }}")
        && text(download, "uses").is_some_and(|v| v.starts_with("actions/download-artifact@"))
        && input(download, "pattern", "swift-sdk-target-*")
        && input(download, "path", "dist/swift-targets")
        && input(download, "merge-multiple", "true")
}
fn publication(document: &Node) -> bool {
    let assembly = job(document, "swift_sdk_artifact");
    let (stage_index, _) = named(assembly, "Stage immutable generated Swift binding");
    let (package_index, _) = named(assembly, "Package Swift SDK input");
    let (verify_index, verify) = named(assembly, "Verify immutable Swift SDK input");
    let (sdk_index, sdk) = named(assembly, "Upload immutable Swift SDK input");
    let (binding_index, binding) = named(assembly, "Upload immutable generated Swift binding");
    stage_index < binding_index
        && package_index < verify_index
        && verify_index < sdk_index
        && verify_index < binding_index
        && verify.get("if").is_none()
        && verify
            .get("env")
            .and_then(|env| text(env, "SWIFT_SDK_MODE"))
            == Some("${{ inputs.mode }}")
        && input(sdk, "name", "${{ inputs.artifact_name }}")
        && input(sdk, "path", "dist/MeshLLMFFI.xcframework.zip")
        && input(
            binding,
            "name",
            "generated-swift-binding-${{ inputs.artifact_name }}",
        )
        && input(binding, "path", "dist/swift-generated-binding")
        && [sdk, binding].into_iter().all(|step| {
            text(step, "uses").is_some_and(|value| value.starts_with("actions/upload-artifact@"))
                && input(step, "if-no-files-found", "error")
                && input(step, "retention-days", "${{ inputs.retention_days }}")
                && step.get("if").is_none()
        })
}
#[test]
fn swift_workflow_publishes_verified_zip_and_staged_binding_with_exact_consumed_arguments() {
    assert!(publication(&workflow()));
    for (valid, invalid) in [
        (
            "path: dist/MeshLLMFFI.xcframework.zip",
            "path: dist/unverified.zip",
        ),
        (
            "path: dist/swift-generated-binding",
            "path: sdk/swift/Sources",
        ),
        (
            "name: generated-swift-binding-${{ inputs.artifact_name }}",
            "name: unrelated-binding",
        ),
        ("if-no-files-found: error", "if-no-files-found: ignore"),
    ] {
        let original = source();
        assert!(original.contains(valid));
        let changed = format!("{}\n# {valid}\n", original.replace(valid, invalid));
        assert!(!publication(&workflow_yaml::parse(&changed).unwrap()));
    }
    let document = workflow();
    let (_, verify) = named(
        job(&document, "swift_sdk_artifact"),
        "Verify immutable Swift SDK input",
    );
    for mode in ["host-only", "full"] {
        for fail in ["false", "true"] {
            let fixture = Fixture::new();
            fs::create_dir_all(fixture.path().join("scripts")).unwrap();
            fixture.executable("verify-observer", r#"[[ "$#" == 2 && "$1" == dist/MeshLLMFFI.xcframework.zip && "$2" == "$SWIFT_SDK_MODE" ]] || exit 97
printf 'verify:%s\n' "$2" > events
[[ "$FAIL_VERIFY" == false ]] || exit 97
"#);
            fs::copy(
                fixture.path().join("bin/verify-observer"),
                fixture
                    .path()
                    .join("scripts/verify-swift-release-artifact.sh"),
            )
            .unwrap();
            let result = run_step(
                &fixture,
                verify,
                &[("SWIFT_SDK_MODE", mode), ("FAIL_VERIFY", fail)],
                None,
            );
            assert_eq!(result.status.success(), fail == "false");
            assert_eq!(
                fs::read_to_string(fixture.path().join("events")).unwrap(),
                format!("verify:{mode}\n")
            );
        }
    }
}
fn cache_target(node: &Node, target: &str, cache_name: &str) -> bool {
    let (_, cache) = named(node, cache_name);
    let prefix = format!(
        "${{{{ format('mesh-llm-swift-sdk-target-{{0}}-{{1}}-{{2}}-{{3}}-{{4}}', {target}, runner.os, runner.arch, steps.native_toolchain.outputs.epoch, hashFiles("
    );
    let expected_path = if target == "matrix.target" {
        ".deps/llama-build/build-stage-abi-${{ matrix.target }}-metal"
    } else {
        ".deps/llama-build/build-stage-abi-aarch64-apple-darwin-metal"
    };
    let Some(inputs) = cache.get("with") else {
        return false;
    };
    let Some(key) = text(inputs, "key") else {
        return false;
    };
    let rust = steps(node)
        .iter()
        .filter(|n| text(n, "uses").is_some_and(|v| v.starts_with("Swatinem/rust-cache@")))
        .collect::<Vec<_>>();
    let [rust] = rust.as_slice() else {
        return false;
    };
    let shared = if target == "matrix.target" {
        "swift-sdk-${{ matrix.target }}"
    } else {
        "swift-sdk-aarch64-apple-darwin"
    };
    text(inputs, "path") == Some(expected_path)
        && key.starts_with(&prefix)
        && !key.contains("inputs.mode")
        && inputs.get("restore-keys").is_none()
        && input(rust, "shared-key", shared)
        && input(rust, "key", "${{ steps.native_toolchain.outputs.epoch }}")
        && input(rust, "add-job-id-key", "false")
}
#[test]
fn swift_workflow_preserves_complete_attempt_aware_target_handoff() {
    assert!(handoff(&workflow()));
    for (valid, invalid) in [
        (
            "pattern: swift-sdk-target-*",
            "pattern: swift-sdk-target-*-${{ github.run_attempt }}",
        ),
        (
            "name: swift-sdk-target-${{ matrix.target }}-${{ github.run_attempt }}",
            "name: swift-sdk-target-${{ matrix.target }}",
        ),
        ("- aarch64-apple-ios-sim", "- unsupported-target"),
    ] {
        let original = source();
        assert!(original.contains(valid));
        let changed = format!("{}\n# {valid}\n", original.replace(valid, invalid));
        assert!(!handoff(&workflow_yaml::parse(&changed).unwrap()));
    }
}
#[test]
fn swift_workflow_cache_identity_is_target_specific_and_mode_independent() {
    let check = |document: &Node| {
        cache_target(
            job(document, "swift_sdk_target"),
            "matrix.target",
            "Restore exact Swift target native ABI cache",
        ) && cache_target(
            job(document, "swift_sdk_artifact"),
            "'aarch64-apple-darwin'",
            "Restore exact Swift native ABI cache",
        )
    };
    assert!(check(&workflow()));
    for (valid, invalid) in [
        (
            "matrix.target, runner.os, runner.arch, steps.native_toolchain.outputs.epoch, hashFiles(",
            "matrix.target, runner.os, inputs.mode, steps.native_toolchain.outputs.epoch, hashFiles(",
        ),
        (
            "shared-key: swift-sdk-${{ matrix.target }}",
            "shared-key: swift-sdk-shared",
        ),
        (
            "path: .deps/llama-build/build-stage-abi-aarch64-apple-darwin-metal",
            "path: .deps/llama-build/build-stage-abi-host-metal",
        ),
    ] {
        let original = source();
        assert!(original.contains(valid));
        let changed = format!("{}\n# {valid}\n", original.replace(valid, invalid));
        assert!(!check(&workflow_yaml::parse(&changed).unwrap()));
    }
}
fn run_step(fixture: &Fixture, step: &Node, env: &[(&str, &str)], target: Option<&str>) -> Output {
    let mut command = Command::new("/bin/bash");
    command.env_clear().current_dir(fixture.path()).env(
        "PATH",
        format!("{}:/usr/bin:/bin", fixture.path().join("bin").display()),
    );
    for (key, value) in env {
        command.env(key, value);
    }
    let mut run = text(step, "run").unwrap().to_owned();
    if let Some(target) = target {
        run = run.replace("${{ matrix.target }}", target);
    }
    command.args(["-c", &run]);
    fixture.run(command)
}
#[test]
fn swift_workflow_admission_bounds_modes_parallelism_and_release_source_preparation() {
    let document = workflow();
    let (_, validate) = named(
        job(&document, "runner_policy"),
        "Validate typed producer inputs",
    );
    for (mode, event, release, prepare, parallel, ui, sha, expected) in [
        ("host-only", "pull_request", "", "false", "1", "", "", true),
        ("full", "push", "", "false", "4", "", "", true),
        (
            "full",
            "workflow_dispatch",
            "v1.2.3",
            "true",
            "4",
            "ui-input",
            "0123456789abcdef0123456789abcdef01234567",
            true,
        ),
        ("unsupported", "push", "", "false", "1", "", "", false),
        ("full", "push", "", "false", "5", "", "", false),
        ("full", "push", "v1.2.3", "true", "1", "", "", false),
        ("full", "workflow_dispatch", "", "true", "1", "", "", false),
        (
            "full",
            "workflow_dispatch",
            "v1.2.3",
            "false",
            "1",
            "ui-input",
            "short-sha",
            false,
        ),
    ] {
        let fixture = Fixture::new();
        let result = run_step(
            &fixture,
            validate,
            &[
                ("SWIFT_SDK_MODE", mode),
                ("EVENT_NAME", event),
                ("RELEASE_TAG", release),
                ("PREPARE_RELEASE_VERSION", prepare),
                ("UPDATE_RELEASE_MANIFEST", "false"),
                ("MAX_PARALLEL", parallel),
                ("UI_ARTIFACT_NAME", ui),
                ("UI_SOURCE_SHA", sha),
            ],
            None,
        );
        assert_eq!(
            result.status.success(),
            expected,
            "{mode}/{event}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
#[test]
fn swift_workflow_executes_target_and_assembly_build_entrypoints_with_exact_consumed_arguments() {
    let document = workflow();
    for target in TARGETS {
        let fixture = Fixture::new();
        fs::create_dir_all(fixture.path().join("mesh/sdk/swift/scripts")).unwrap();
        fixture.executable("build-observer","[[ \"$#\" == 2 && \"$1\" == --target && \"$2\" == \"$EXPECTED_TARGET\" ]] || exit 97\nprintf 'target:%s\\n' \"$2\" > events");
        fs::copy(
            fixture.path().join("bin/build-observer"),
            fixture
                .path()
                .join("mesh/sdk/swift/scripts/build-xcframework.sh"),
        )
        .unwrap();
        let (_, build) = named(job(&document, "swift_sdk_target"), "Build Swift target");
        let result = run_step(
            &fixture,
            build,
            &[("SDK_DIR", "mesh/sdk"), ("EXPECTED_TARGET", target)],
            Some(target),
        );
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            fs::read_to_string(fixture.path().join("events")).unwrap(),
            format!("target:{target}\n")
        );
    }
    let fixture = Fixture::new();
    fs::create_dir_all(fixture.path().join("mesh/sdk/swift/scripts")).unwrap();
    fixture.executable("build-observer","[[ \"$#\" == 2 && \"$1\" == --assemble-from && \"$2\" == dist/swift-targets ]] || exit 97\nprintf 'assembly\\n' > events");
    fs::copy(
        fixture.path().join("bin/build-observer"),
        fixture
            .path()
            .join("mesh/sdk/swift/scripts/build-xcframework.sh"),
    )
    .unwrap();
    let (_, build) = named(
        job(&document, "swift_sdk_artifact"),
        "Build full Swift SDK input",
    );
    let result = run_step(&fixture, build, &[("SDK_DIR", "mesh/sdk")], None);
    assert!(result.status.success());
    assert_eq!(
        fs::read_to_string(fixture.path().join("events")).unwrap(),
        "assembly\n"
    );
}
#[test]
fn swift_workflow_checks_tracked_binding_changes_and_stages_exact_binding_bytes() {
    let document = workflow();
    let assembly = job(&document, "swift_sdk_artifact");
    let (verify_index, verify) =
        named(assembly, "Verify committed Swift binding source is current");
    let (stage_index, stage) = named(assembly, "Stage immutable generated Swift binding");
    assert!(verify_index < stage_index);
    assert_eq!(
        text(verify, "if"),
        Some("${{ !inputs.prepare_release_version }}")
    );
    assert!(stage.get("if").is_none());
    let binding = "mesh/sdk/swift/Sources/MeshLLM/Generated/mesh_ffi.swift";
    for mode in ["clean", "changed", "untracked"] {
        let fixture = Fixture::new();
        fs::create_dir_all(fixture.path().join(binding).parent().unwrap()).unwrap();
        let bytes = b"immutable generated Swift binding\n";
        fs::write(fixture.path().join(binding), bytes).unwrap();
        fixture.executable("git",r#"case "$1" in
ls-files) [[ "$#" == 3 && "$2" == --error-unmatch && "$3" == "$BINDING" ]] || exit 97; printf 'tracked\n' >> events; [[ "$MODE" != untracked ]];;
diff) [[ "$#" == 4 && "$2" == --exit-code && "$3" == -- && "$4" == "$BINDING" ]] || exit 97; printf 'diff\n' >> events; [[ "$MODE" != changed ]];;
*) exit 97;; esac
"#);
        let env = [
            ("SDK_DIR", "mesh/sdk"),
            ("BINDING", binding),
            ("MODE", mode),
        ];
        let result = run_step(&fixture, verify, &env, None);
        assert_eq!(result.status.success(), mode == "clean");
        if mode == "clean" {
            let staged = run_step(&fixture, stage, &env, None);
            assert!(staged.status.success());
            let output = fixture
                .path()
                .join("dist/swift-generated-binding/mesh_ffi.swift");
            assert_eq!(fs::read(&output).unwrap(), bytes);
            assert_eq!(
                fs::metadata(output).unwrap().permissions().mode() & 0o777,
                0o644
            );
            assert_eq!(
                fs::read_to_string(fixture.path().join("events")).unwrap(),
                "tracked\ndiff\ntracked\n"
            );
        } else if mode == "changed" {
            assert!(
                String::from_utf8_lossy(&result.stderr)
                    .contains("generated Swift UniFFI bindings changed")
            );
        } else {
            let staged = run_step(&fixture, stage, &env, None);
            assert!(!staged.status.success());
            assert!(!fixture.path().join("dist/swift-generated-binding").exists());
        }
    }
}
