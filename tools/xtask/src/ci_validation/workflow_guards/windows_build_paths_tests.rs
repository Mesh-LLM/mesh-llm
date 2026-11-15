use super::*;
fn job(steps: &str) -> Node {
    workflow_yaml::parse(&format!("steps:\n{steps}")).unwrap()
}
#[test]
fn windows_build_paths_actual_managed_jobs_have_branch_valid_setup_before_cache() {
    check(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
}
#[test]
fn windows_build_paths_missing_duplicate_and_skipped_setup_fail() {
    for steps in [
        "  - run: cargo build\n",
        "  - uses: ./.github/actions/setup-windows-short-paths\n  - uses: ./.github/actions/setup-windows-short-paths\n",
        "  - uses: ./.github/actions/setup-windows-short-paths\n    if: false\n",
    ] {
        assert!(job_paths(&job(steps), None).is_err(), "{steps}");
    }
    assert!(
        job_paths(
            &workflow_yaml::parse("name: omitted steps\n").unwrap(),
            None
        )
        .is_err()
    );
}
#[test]
fn windows_build_paths_compiler_and_cache_cannot_run_before_setup() {
    for prior in [
        "uses: mozilla-actions/sccache-action@finite-pin",
        "uses: Swatinem/rust-cache@finite-pin",
        "uses: ./.github/actions/restore-windows-abi-cache",
        "run: cargo check --locked -p mesh-llm",
        "run: cmake --build native",
        "run: scripts/build-windows.ps1 -Backend cpu",
    ] {
        let steps = format!("  - {prior}\n  - uses: ./.github/actions/setup-windows-short-paths\n");
        assert!(job_paths(&job(&steps), None).is_err(), "{prior}");
    }
    job_paths(&job("  - uses: actions/checkout@finite-pin\n  - uses: ./.github/actions/setup-windows-short-paths\n  - run: cargo build\n"), None).unwrap();
}
#[test]
fn windows_build_paths_sccache_must_pin_the_argfile_safe_release() {
    let setup = "  - uses: ./.github/actions/setup-windows-short-paths\n";
    for install in [
        "  - uses: mozilla-actions/sccache-action@finite-pin\n",
        "  - uses: mozilla-actions/sccache-action@finite-pin\n    with:\n      version: v0.19.0\n",
    ] {
        let error = job_paths(&job(&format!("{setup}{install}")), None).unwrap_err();
        assert!(
            error.to_string().contains("argfile-safe release"),
            "{install}: {error}"
        );
    }
    let pinned =
        "  - uses: mozilla-actions/sccache-action@finite-pin\n    with:\n      version: v0.16.0\n";
    job_paths(&job(&format!("{setup}{pinned}")), None).unwrap();
}
#[test]
fn windows_build_paths_composite_action_sccache_must_pin_the_argfile_safe_release() {
    let Some(Node::Seq(steps)) = job(
        "  - uses: mozilla-actions/sccache-action@finite-pin\n    if: runner.os == 'Windows'\n",
    )
    .get("steps")
    .cloned() else {
        panic!("fixture steps");
    };
    let error = sccache_pinned(&steps).unwrap_err();
    assert!(
        error.to_string().contains("argfile-safe release"),
        "{error}"
    );
}
#[test]
fn windows_build_paths_multi_platform_job_uses_windows_condition() {
    let steps = format!(
        "  - uses: ./.github/actions/setup-windows-short-paths\n    if: {WINDOWS_MATRIX}\n  - run: cargo check\n"
    );
    job_paths(&job(&steps), Some(WINDOWS_MATRIX)).unwrap();
    assert!(job_paths(&job(&steps), None).is_err());
    assert!(
        job_paths(
            &job("  - uses: ./.github/actions/setup-windows-short-paths\n"),
            Some(WINDOWS_MATRIX)
        )
        .is_err()
    );
}

#[test]
fn windows_build_paths_host_target_keeps_provider_fallback_and_actual_cache_root() {
    let source = "  - uses: ./.github/actions/setup-windows-short-paths\n    id: short_paths\n    with:\n      cargo_target: host\n  - uses: Swatinem/rust-cache@finite-pin\n    with:\n      workspaces: . -> target\n      cache-targets: ${{ steps.short_paths.outputs.cargo_target == '' }}\n      cache-directories: ${{ steps.short_paths.outputs.cargo_target }}\n";
    host_target_mode(&job(source), true).unwrap();
    assert!(host_target_mode(&job(source), false).is_err());
    for (before, after) in [
        ("cargo_target: host", "cargo_target: arbitrary"),
        ("id: short_paths", "id: missing_output"),
        ("workspaces: . -> target", "workspaces: . -> D:\\b"),
        (
            "${{ steps.short_paths.outputs.cargo_target == '' }}",
            "true",
        ),
        (
            "cache-directories: ${{ steps.short_paths.outputs.cargo_target }}",
            "cache-directories: target",
        ),
    ] {
        assert!(
            host_target_mode(&job(&source.replace(before, after)), true).is_err(),
            "{before}"
        );
    }
    host_target_mode(
        &job("  - uses: ./.github/actions/setup-windows-short-paths\n"),
        false,
    )
    .unwrap();
}

#[test]
fn windows_build_paths_short_target_cannot_become_the_default_or_run_on_other_providers() {
    let root = tempfile::tempdir().unwrap();
    let directory = root
        .path()
        .join(".github/actions/setup-windows-short-paths");
    fs::create_dir_all(&directory).unwrap();
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.github/actions/setup-windows-short-paths/action.yml"),
    )
    .unwrap();
    for candidate in [
        source.replace("default: workspace", "default: host"),
        source.replace(" && runner.environment == 'github-hosted'", ""),
        source.replace("inputs.cargo_target == 'host' && ", ""),
    ] {
        fs::write(directory.join("action.yml"), candidate).unwrap();
        assert!(host_target_action(root.path()).is_err());
    }
}
