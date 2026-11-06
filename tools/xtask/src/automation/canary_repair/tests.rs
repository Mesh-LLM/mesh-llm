use super::*;
#[cfg(unix)]
use std::collections::BTreeMap;

fn fixture(phase: Phase) -> (tempfile::TempDir, Input) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = Input {
        root: root.join("source"),
        controller: root.join("controller"),
        state: root.join("state"),
        build: root.join("native"),
        backend: "metal".into(),
        phase,
        seconds: 10,
    };
    for path in [
        &input.root,
        &input.controller,
        &input.state.join("checks"),
        &input.build,
    ] {
        fs::create_dir_all(path).unwrap();
    }
    (temp, input)
}

#[cfg(unix)]
#[test]
fn output_paths_resolve_symlinked_existing_ancestors() {
    let (temp, input) = fixture(Phase::Fast);
    let alias = temp.path().join("alias");
    std::os::unix::fs::symlink(&input.root, &alias).unwrap();
    assert_eq!(
        resolved_output(&alias.join("nested/checks")).unwrap(),
        input.root.join("nested/checks")
    );
}

#[test]
fn candidate_checks_cannot_write_github_workflow_control_files() {
    let environment = runner::inherited([
        ("GITHUB_ENV".into(), "/runner/env".into()),
        ("GITHUB_OUTPUT".into(), "/runner/output".into()),
        ("GITHUB_STEP_SUMMARY".into(), "/runner/summary".into()),
        ("GITHUB_PATH".into(), "/runner/path".into()),
        ("GITHUB_STATE".into(), "/runner/state".into()),
        ("GH_ENTERPRISE_TOKEN".into(), "credential".into()),
        ("PATH".into(), "/usr/bin".into()),
    ]);
    assert_eq!(environment.len(), 1);
    assert!(environment.contains_key(std::ffi::OsStr::new("PATH")));
}

#[test]
fn focused_and_full_plans_share_exact_environment_compile_and_runtime_checks() {
    let (_temp, mut input) = fixture(Phase::Fast);
    let fast = plan::steps(&input);
    let prepare = fast
        .iter()
        .position(|s| s.label == "native-prepare")
        .unwrap();
    let compile = fast
        .iter()
        .position(|s| s.label == "native-compile")
        .unwrap();
    assert!(prepare < compile);
    assert!(
        fast[compile]
            .arguments
            .windows(2)
            .any(|args| args == ["--target", "test-jinja"])
    );
    assert_eq!(fast[prepare].arguments.last().unwrap(), "pinned");
    assert!(
        !fast
            .iter()
            .any(|s| s.label == "generated-family-and-native-suite")
    );
    let focused = fast
        .iter()
        .find(|s| s.label == "focused-split-state-smoke")
        .unwrap();
    assert_eq!(focused.environment["SKIPPY_SMOKE_SKIP_BUILD"], "1");
    assert_eq!(focused.environment["SMOKE_N_GPU_LAYERS"], "999");
    assert_eq!(focused.environment["SMOKE_FLASH_ATTN"], "enabled");
    assert!(
        fast.iter().position(|s| s.identity).unwrap()
            < fast.iter().position(|s| s.label == focused.label).unwrap()
    );
    input.phase = Phase::Full;
    let full = plan::steps(&input);
    assert_eq!(
        serde_json::to_value(&fast).unwrap(),
        serde_json::to_value(&full[..fast.len()]).unwrap()
    );
    assert_eq!(
        full.iter()
            .filter(|s| s.label == "generated-family-and-native-suite")
            .count(),
        1
    );
    for label in [
        "canonical-skippy-smoke",
        "workload-oracles",
        "system-one-smoke",
        "laya-smoke",
    ] {
        assert!(full.iter().any(|s| s.label == label));
    }
    let complete = full
        .iter()
        .position(|s| s.label == "complete-native-compile")
        .unwrap();
    assert!(complete >= fast.len());
    assert!(!full[complete].arguments.iter().any(|arg| arg == "--target"));
    let canonical = full
        .iter()
        .find(|s| s.label == "canonical-skippy-smoke")
        .unwrap();
    assert!(
        !canonical
            .environment
            .contains_key("SKIPPY_SMOKE_FOCUSED_ONLY")
    );
    input.phase = Phase::Environment;
    assert_eq!(plan::steps(&input).len(), 1);
}

#[test]
fn cpu_focused_plan_does_not_request_unavailable_gpu_flash_attention() {
    let (_temp, mut input) = fixture(Phase::Fast);
    input.backend = "cpu".into();
    let steps = plan::steps(&input);
    let smoke = steps
        .iter()
        .find(|s| s.label == "focused-split-state-smoke")
        .unwrap();
    assert_eq!(smoke.environment["SMOKE_N_GPU_LAYERS"], "0");
    assert_eq!(smoke.environment["SMOKE_FLASH_ATTN"], "disabled");
}

fn native_fixture(input: &Input) {
    let native = input.root.join(".deps/llama.cpp");
    fs::create_dir_all(&native).unwrap();
    fs::create_dir_all(input.root.join("skippy/llama_cpp")).unwrap();
    fs::create_dir_all(input.root.join("target/debug")).unwrap();
    fs::write(
        input.root.join("skippy/llama_cpp/upstream.txt"),
        "a".repeat(40),
    )
    .unwrap();
    let rust = input.root.join("skippy/crates/skippy-ffi/src");
    fs::create_dir_all(&rust).unwrap();
    fs::create_dir_all(native.join("include/skippy")).unwrap();
    fs::write(rust.join("lib.rs"), "pub const ABI_VERSION_MAJOR: u32 = 0;\npub const ABI_VERSION_MINOR: u32 = 1;\npub const ABI_VERSION_PATCH: u32 = 79;\n").unwrap();
    fs::write(native.join("include/skippy/common.h"), "#define SKIPPY_ABI_VERSION_MAJOR 0\n#define SKIPPY_ABI_VERSION_MINOR 1\n#define SKIPPY_ABI_VERSION_PATCH 79\n").unwrap();
    for name in [
        ".mesh-llm-upstream-sha",
        ".mesh-llm-patch-digest",
        ".mesh-llm-patched-sha",
        ".mesh-llm-prepare-schema",
    ] {
        fs::write(native.join(name), "a".repeat(40)).unwrap();
    }
    for name in ["skippy", "skippy-correctness", "skippy-package-builder"] {
        fs::write(input.root.join("target/debug").join(name), name).unwrap();
    }
    fs::write(input.root.join(".gitignore"), ".deps/\ntarget/\n").unwrap();
    for repository in [&native, &input.root] {
        for arguments in [
            vec!["init", "-q"],
            vec!["add", "."],
            vec![
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
        ] {
            assert!(
                std::process::Command::new("git")
                    .args(arguments)
                    .current_dir(repository)
                    .status()
                    .unwrap()
                    .success()
            );
        }
    }
}

#[test]
fn identity_rejects_changed_child_binary_native_recipe_and_requested_pin() {
    let (_temp, input) = fixture(Phase::Fast);
    native_fixture(&input);
    let identity = identity::capture(&input).unwrap();
    identity.verify(&input).unwrap();
    let header = input.root.join(".deps/llama.cpp/include/skippy/common.h");
    let original = fs::read_to_string(&header).unwrap();
    fs::write(&header, original.replace("PATCH 79", "PATCH 78")).unwrap();
    assert!(identity.verify(&input).is_err());
    fs::write(&header, original).unwrap();
    let child = input.root.join("target/debug/skippy");

    fs::write(&child, "stale child").unwrap();
    assert!(identity.verify(&input).is_err());
    fs::write(child, "skippy").unwrap();
    let marker = input.root.join(".deps/llama.cpp/.mesh-llm-patched-sha");
    fs::write(&marker, "b".repeat(40)).unwrap();
    assert!(identity.verify(&input).is_err());
    fs::write(marker, "a".repeat(40)).unwrap();
    let model = input.root.join(".deps/llama.cpp/src/model.cpp");
    fs::create_dir_all(model.parent().unwrap()).unwrap();
    fs::write(&model, "candidate changed after compile").unwrap();
    assert!(identity.verify(&input).is_err());
    let dirty_identity = identity::capture(&input).unwrap();
    fs::write(&model, "candidate changed again").unwrap();
    assert!(dirty_identity.verify(&input).is_err());
    fs::remove_file(model).unwrap();
    fs::write(input.root.join("skippy/llama_cpp/new.patch"), "new patch").unwrap();
    assert!(identity.verify(&input).is_err());
    fs::remove_file(input.root.join("skippy/llama_cpp/new.patch")).unwrap();
    fs::write(
        input.root.join("skippy/llama_cpp/upstream.txt"),
        "b".repeat(40),
    )
    .unwrap();
    assert!(identity.verify(&input).is_err());
}

#[cfg(unix)]
#[test]
fn missing_executable_keeps_reproduction_details_and_returns_infrastructure_status() {
    let (_temp, input) = fixture(Phase::Fast);
    let step = plan::Step {
        label: "missing-executable".into(),
        arguments: vec!["/missing/repair-prerequisite".into(), "--check".into()],
        environment: BTreeMap::new(),
        identity: false,
    };
    let report = runner::execute(&input, &[step]).unwrap();
    assert_eq!(report.code, 125);
    let row: serde_json::Value =
        serde_json::from_slice(&fs::read(input.state.join("last-check.json")).unwrap()).unwrap();
    assert_eq!(row["command"][0], "/missing/repair-prerequisite");
    assert_eq!(row["environment"]["LLAMA_STAGE_LINK_MODE"], "static");
    assert_eq!(row["failure_class"], "infrastructure");
    assert_eq!(row["rerun_repair_check"][0], "cargo");
    assert!(
        row["rerun_repair_check"]
            .as_array()
            .unwrap()
            .contains(&serde_json::Value::String(
                input.state.to_string_lossy().into_owned()
            ))
    );
}

#[cfg(unix)]
fn shell_step(label: &str, body: &str) -> plan::Step {
    let sh = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|p| p.join("sh"))
        .find(|p| p.is_file())
        .unwrap()
        .canonicalize()
        .unwrap();
    plan::Step {
        label: label.into(),
        arguments: vec![sh.to_string_lossy().into_owned(), "-c".into(), body.into()],
        environment: BTreeMap::new(),
        identity: false,
    }
}

#[cfg(unix)]
#[test]
fn owned_commands_stop_at_first_failure_and_retain_reproduction_and_both_streams() {
    let (_temp, input) = fixture(Phase::Fast);
    let steps = [
        shell_step(
            "broken-native",
            r#"test "$LLAMA_STAGE_LINK_MODE" = static || exit 99
test "$LLAMA_STAGE_UPSTREAM_TESTS" = OFF || exit 99
test "$STAGE_SERVER_BIN" = "$PWD/target/debug/skippy" || exit 99
test "$CARGO_TARGET_DIR" = "$PWD/target" || exit 99
test -z "${CARGO_BUILD_TARGET+x}" || exit 99
case "$PATH" in "$UV_PROJECT_ENVIRONMENT/bin:"*) ;; *) exit 99;; esac
echo compiler-output; echo exact-failure >&2; exit 73"#,
        ),
        shell_step("must-not-start", "touch forbidden"),
    ];
    let report = runner::execute(&input, &steps).unwrap();
    assert_eq!(report.code, 73);
    assert!(!input.root.join("forbidden").exists());
    let row: serde_json::Value =
        serde_json::from_slice(&fs::read(input.state.join("last-check.json")).unwrap()).unwrap();
    assert_eq!(row["failure_class"], "candidate");
    assert_eq!(row["environment"]["LLAMA_STAGE_LINK_MODE"], "static");
    assert_eq!(
        row["environment"]["STAGE_SERVER_BIN"],
        input
            .root
            .join("target/debug/skippy")
            .to_string_lossy()
            .as_ref()
    );
    assert!(
        row["removed_environment"]
            .as_array()
            .unwrap()
            .iter()
            .any(|key| key == "CARGO_BUILD_TARGET")
    );
    assert_eq!(
        row["command"],
        serde_json::to_value(&steps[0].arguments).unwrap()
    );
    assert_eq!(
        fs::read_to_string(row["log"].as_str().unwrap()).unwrap(),
        "compiler-output\n"
    );
    let directory = Path::new(row["log"].as_str().unwrap()).parent().unwrap();
    assert_eq!(
        fs::read_to_string(directory.join("broken-native.stderr.log")).unwrap(),
        "exact-failure\n"
    );
}

#[cfg(unix)]
#[test]
fn missing_dependency_and_owned_timeout_stop_without_requesting_code_repair() {
    let (_temp, mut input) = fixture(Phase::Fast);
    let report = runner::execute(
        &input,
        &[shell_step(
            "upstream-python-tests",
            "echo 'ModuleNotFoundError: numpy' >&2; exit 1",
        )],
    )
    .unwrap();
    assert_eq!(report.code, 125);
    let row: serde_json::Value =
        serde_json::from_slice(&fs::read(input.state.join("last-check.json")).unwrap()).unwrap();
    assert_eq!(row["failure_class"], "dependency");
    let report = runner::execute(&input, &[shell_step("missing-tool", "exit 127")]).unwrap();
    assert_eq!(report.code, 125);
    let row: serde_json::Value =
        serde_json::from_slice(&fs::read(input.state.join("last-check.json")).unwrap()).unwrap();
    assert_eq!(row["failure_class"], "infrastructure");
    input.seconds = 1;

    let report = runner::execute(&input, &[shell_step("blocked-child", "sleep 60")]).unwrap();
    assert_eq!(report.code, 125);
    let row: serde_json::Value =
        serde_json::from_slice(&fs::read(input.state.join("last-check.json")).unwrap()).unwrap();
    assert_eq!(row["failure_class"], "infrastructure");
}

#[test]
fn only_an_isolated_fixture_bus_error_can_request_a_retry_and_full_native_rerun() {
    let (_temp, mut input) = fixture(Phase::Full);
    let directory = input.build.join("Testing/Temporary");
    fs::create_dir_all(&directory).unwrap();
    let failed = directory.join("LastTestsFailed.log");
    let log = input.state.join("native.log");
    fs::write(
        &log,
        "98/371 Test #203: skippy_generate_contract_ultravox ...Bus error***Exception: 0.04 sec\n",
    )
    .unwrap();
    fs::write(
        &failed,
        "203:skippy_generate_contract_ultravox\n204:skippy_graph_contract_ultravox\n",
    )
    .unwrap();
    let steps = retry::steps(&input, &log);
    if cfg!(target_os = "macos") {
        let steps = steps.unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(
            steps[0].arguments.last().unwrap(),
            "^skippy_generate_contract_ultravox$"
        );
        assert!(
            !steps[1]
                .arguments
                .iter()
                .any(|s| s == "-R" || s == "--rerun-failed")
        );
    } else {
        assert!(steps.is_none());
    }
    fs::write(
        &log,
        "clang: Bus error while compiling; stale LastTestsFailed is unrelated\n",
    )
    .unwrap();
    assert!(retry::steps(&input, &log).is_none());
    fs::write(
        &log,
        "98/371 Test #203: skippy_generate_contract_ultravox ...Bus error***Exception: 0.04 sec\n",
    )
    .unwrap();
    input.backend = "cpu".into();
    assert!(retry::steps(&input, &log).is_none());
    input.backend = "metal".into();
    fs::write(
        &failed,
        "203:skippy_generate_contract_ultravox\n341:test-conversion-fp8\n",
    )
    .unwrap();
    assert!(retry::steps(&input, &log).is_none());
    fs::write(&failed, "203:skippy_generate_contract_ultravox\n").unwrap();
    fs::write(log, "ordinary assertion failed\n").unwrap();
    assert!(retry::steps(&input, &input.state.join("native.log")).is_none());
}
