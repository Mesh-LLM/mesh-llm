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
