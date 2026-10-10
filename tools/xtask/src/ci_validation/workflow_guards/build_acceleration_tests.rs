use super::*;

fn document(source: &str) -> Node {
    super::super::workflow_yaml::parse(source).unwrap()
}

fn isolated_step() -> Node {
    document(
        "name: Prepare isolated automation before measurement\nshell: bash\nenv:\n  CARGO_TARGET_DIR: ${{ runner.temp }}/runtime-seed-automation-target\n  CARGO_HOME: ${{ runner.temp }}/runtime-seed-automation-cargo\n  RUSTC_WRAPPER: ''\n  CARGO_BUILD_RUSTC_WRAPPER: ''\nrun: |\n  set -euo pipefail\n  cargo build --locked --release -p xtask --bin xtask\n  echo \"MESH_LLM_AUTOMATION_BIN=$CARGO_TARGET_DIR/release/xtask\" >> \"$GITHUB_ENV\"\n",
    )
}

fn replace(node: &mut Node, key: &str, value: &str) {
    let Node::Map(entries) = node else {
        panic!("mapping required")
    };
    let (_, field) = entries.iter_mut().find(|(name, _)| name == key).unwrap();
    *field = Node::Scalar(value.into());
}

#[test]
fn build_acceleration_actual_workflows_and_actions_preserve_repository_defaults() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    check(&root).unwrap();
}

#[test]
fn build_acceleration_environment_cannot_disable_cache_or_replace_target_driver() {
    for source in [
        "env:\n  RUSTC_WRAPPER: ''\n",
        "env:\n  RUSTC_WRAPPER: \"\"\n",
        "env:\n  CARGO_BUILD_RUSTC_WRAPPER: ''\n",
        "env:\n  LLAMA_STAGE_USE_SCCACHE: '0'\n",
        "env:\n  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER: other-driver\n",
    ] {
        assert!(
            check_node("workflow.yml", &document(source)).is_err(),
            "{source}"
        );
    }
    check_node(
        "workflow.yml",
        &document("env:\n  RUSTC_WRAPPER: sccache\n  LLAMA_STAGE_USE_SCCACHE: '1'\n"),
    )
    .unwrap();
}

#[test]
fn build_acceleration_nested_jobs_and_composite_actions_reject_overrides() {
    for source in [
        "jobs:\n  compile:\n    steps:\n      - env:\n          RUSTC_WRAPPER: ''\n        run: cargo build\n",
        "runs:\n  using: composite\n  steps:\n    - env:\n        RUSTC_WRAPPER: ''\n      run: cargo build\n",
    ] {
        assert!(check_node("definition.yml", &document(source)).is_err());
    }
}

#[test]
fn build_acceleration_unprobed_linker_flags_fail_and_comments_do_not_override_policy() {
    for run in [
        "cargo build -C linker=other",
        "env RUSTFLAGS=-Clinker=other cargo build",
        "cc -fuse-ld=lld object.o",
        "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=other cargo build",
    ] {
        assert!(linker_defaults(run).is_err(), "{run}");
    }
    linker_defaults("# do not use -fuse-ld=lld here\njust release-host-build\n").unwrap();
}

#[test]
fn build_acceleration_exception_requires_the_named_definition_and_unmeasured_roots() {
    let step = isolated_step();
    check_node(".github/workflows/depot-canary.yml", &step).unwrap();
    assert!(check_node(".github/workflows/another.yml", &step).is_err());
    for key in ["CARGO_HOME", "CARGO_TARGET_DIR"] {
        let mut step = isolated_step();
        let Node::Map(entries) = &mut step else {
            unreachable!()
        };
        let environment = &mut entries
            .iter_mut()
            .find(|(name, _)| name == "env")
            .unwrap()
            .1;
        replace(environment, key, "measured-cache");
        assert!(check_node(".github/workflows/depot-canary.yml", &step).is_err());
    }
}

#[test]
fn build_acceleration_exception_admits_semantic_option_order_but_never_product_builds() {
    automation_build("build --bin xtask --release -p xtask --locked").unwrap();
    for arguments in [
        "build --release --locked -p mesh-llm --bin mesh-llm",
        "build --release --locked --workspace",
        "build --release -p xtask --bin xtask",
        "build --locked --release -p xtask --bin xtask --locked",
    ] {
        assert!(automation_build(arguments).is_err(), "{arguments}");
    }
    let mut step = isolated_step();
    let run = field(&step, "run").unwrap().to_owned() + "just release-host-build\n";
    replace(&mut step, "run", &run);
    assert!(check_node(".github/workflows/depot-canary.yml", &step).is_err());
}
