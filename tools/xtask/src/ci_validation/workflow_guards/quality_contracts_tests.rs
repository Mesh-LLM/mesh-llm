use super::*;
fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn inputs() -> (String, Vec<String>) {
    let root = root();
    (
        std::fs::read_to_string(root.join("just/ci.just")).unwrap(),
        serde_json::from_slice(&std::fs::read(root.join(REGISTRY)).unwrap()).unwrap(),
    )
}
#[test]
fn current_quality_and_local_recipes_select_owning_portable_targets() {
    let (source, targets) = inputs();
    recipe(&source, &targets).unwrap();
    local(&source).unwrap();
    let workflow = super::super::workflow_yaml::parse(
        &std::fs::read_to_string(root().join(".github/workflows/ci-quality-slice.yml")).unwrap(),
    )
    .unwrap();
    quality(&workflow).unwrap();
}
#[test]
fn missing_extra_duplicate_or_filtered_cargo_coverage_is_rejected() {
    let (source, targets) = inputs();
    for changed in [
        source.replace("--test migration_lifecycle \\", ""),
        source.replace(
            "--test migration_lifecycle \\",
            "--test migration_lifecycle --test migration_lifecycle \\",
        ),
        source.replace("--test-threads=1", "--test-threads=1 --ignored"),
        source.replace("--bin xtask", "--bin another_tool"),
        source.replace("cargo test --locked", "cargo test"),
        source.replace("--test-threads=1", "--test-threads=2"),
        source.replace("-p xtask --bin", "-p mesh-llm --bin"),
    ] {
        assert!(recipe(&changed, &targets).is_err());
    }
}
#[test]
fn bootstrap_fixture_build_and_required_environment_cannot_be_omitted_or_reordered() {
    let (source, targets) = inputs();
    for changed in [
        source.replace(
            "target_directory=\"$(printf",
            "guessed_directory=\"$(printf",
        ),
        source.replace(
            "export CARGO_TARGET_DIR=\"$target_directory\"",
            "export CARGO_TARGET_DIR=target",
        ),
        source.replace("--bins --examples", "--bins"),
        source.replace(
            "export SPV_ADAPTER_FIXTURE=",
            "# export SPV_ADAPTER_FIXTURE=",
        ),
        source.replace("export MIGRATION_TEST_GIT=", "# export MIGRATION_TEST_GIT="),
        source.replace(
            "bootstrap=\"$(just automation-bootstrap)\"",
            "echo bootstrap=\"$(just automation-bootstrap)\"",
        ),
    ] {
        assert!(recipe(&changed, &targets).is_err());
    }
    let build = "    just with-lld cargo build --locked -p xtask --bins --examples\n";
    let late = source.replace(build, "").replace(
        "    just with-lld cargo test",
        &format!("{build}    just with-lld cargo test"),
    );
    assert!(recipe(&late, &targets).is_err());
}
#[test]
fn quality_calls_are_required_direct_and_ordered_before_retained_legacy() {
    for run in [
        "echo just ci-automation-contracts\njust ci-legacy-contracts",
        "just ci-legacy-contracts\njust ci-automation-contracts",
        "just ci-automation-contracts || true\njust ci-legacy-contracts",
        "# just ci-automation-contracts\njust ci-legacy-contracts",
    ] {
        let yaml = format!(
            "jobs:\n  quality_contracts:\n    steps:\n      - name: Test CI, packaging, and SDK contracts\n        run: |\n{}",
            run.lines()
                .map(|l| format!("          {l}\n"))
                .collect::<String>()
        );
        let workflow = super::super::workflow_yaml::parse(&yaml).unwrap();
        assert!(quality(&workflow).is_err());
    }
    let (source, _) = inputs();
    assert!(
        local(&source.replace(
            "    just ci-automation-contracts\n",
            "    echo just ci-automation-contracts\n"
        ))
        .is_err()
    );
}

#[test]
fn canonical_recipe_preserves_operator_controller_and_normal_absent_fallback() {
    let (source, targets) = inputs();
    recipe(&source, &targets).unwrap();
    for assignment in [
        "export MESH_LLM_AUTOMATION_BIN=\"$binary\"",
        "MESH_LLM_AUTOMATION_BIN=\"$binary\"",
        "unset MESH_LLM_AUTOMATION_BIN",
    ] {
        let changed = source.replace(
            "    just with-lld cargo test",
            &format!("    {assignment}\n    just with-lld cargo test"),
        );
        assert!(recipe(&changed, &targets).is_err());
    }
}

#[test]
fn required_cargo_test_cannot_be_looped_or_have_errexit_disabled_or_status_replaced() {
    let (source, targets) = inputs();
    let looped = source.replace(
        "    just with-lld cargo test",
        "    while false; do\n    just with-lld cargo test",
    ) + "    done\n";
    for changed in [
        source.replace("set -euo pipefail", "set -euo pipefail\n    set +e"),
        source.replace("set -euo pipefail", "set -euo pipefail\n    set +o errexit"),
        source.clone() + "    echo success\n",
        looped,
    ] {
        assert!(recipe(&changed, &targets).is_err());
    }
}

#[test]
fn only_fixture_suffix_case_is_admitted_and_control_transfer_cannot_skip_tests() {
    let (source, targets) = inputs();
    recipe(&source, &targets).unwrap();
    for transfer in [
        "case x in x) exit 0 ;; esac",
        "case x in x) return 0 ;; esac",
        "trap 'exit 0' EXIT",
        "exec true",
    ] {
        let changed = source.replace(
            "    just with-lld cargo test",
            &format!("    {transfer}\n    just with-lld cargo test"),
        );
        assert!(recipe(&changed, &targets).is_err(), "{transfer}");
    }
}
