use super::*;

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source() -> String {
    std::fs::read_to_string(root().join(".github/workflows").join(WORKFLOW)).unwrap()
}

fn action(name: &str) -> DynResult<Node> {
    let path = root().join(format!(".github/actions/{name}/action.yml"));
    Ok(workflow_yaml::parse(&std::fs::read_to_string(path)?)?)
}

fn verdict(source: &str) -> Result<(), String> {
    let document = workflow_yaml::parse(source)?;
    check_document(&document, &action).map_err(|error| error.to_string())
}

fn mutated(before: &str, after: &str) -> Result<(), String> {
    let source = source();
    assert!(source.contains(before), "missing mutation source: {before}");
    verdict(&source.replacen(before, after, 1))
}

const UPLOAD_STEP: &str = "      - name: Upload measurement evidence\n";
const RESTORE_IF: &str = "        if: ${{ inputs.condition != 'cold' }}\n";

#[test]
fn linux_cache_measurement_actual_workflow_is_manual_read_only_and_save_free() {
    check(&root()).unwrap();
}

#[test]
fn linux_cache_measurement_refuses_any_non_dispatch_trigger() {
    for event in [
        "push:\n    branches: [main]\n",
        "pull_request:\n",
        "schedule:\n    - cron: '0 0 * * *'\n",
        "workflow_call:\n",
        "workflow_run:\n    workflows: [Main]\n",
    ] {
        let error = mutated(
            "on:\n  workflow_dispatch:\n",
            &format!("on:\n  {event}  workflow_dispatch:\n"),
        )
        .unwrap_err();
        assert!(error.contains("only workflow_dispatch"), "{event}: {error}");
    }
}

#[test]
fn linux_cache_measurement_refuses_cache_save_prune_and_publication() {
    let save = "      - name: Save measured entry\n        uses: actions/cache/save@caa296126883cff596d87d8935842f9db880ef25 # v5.1.0\n        with:\n          path: runtime-input\n          key: x\n";
    let combined = "      - name: Combined cache\n        uses: actions/cache@caa296126883cff596d87d8935842f9db880ef25 # v5.1.0\n        with:\n          path: runtime-input\n          key: x\n";
    let verified =
        "      - name: Publish\n        uses: ./.github/actions/save-and-verify-actions-cache\n";
    let controller = "      - uses: ./.github/actions/prepare-ci-controller\n";
    let automation = "      - uses: ./.github/actions/prepare-automation\n        with:\n          runner-profile: hosted-bare\n";
    let prune = "      - name: Prune\n        run: gh cache delete --all\n";
    let artifact = "      - name: Publish package\n        uses: actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f # v6.0.0\n        with:\n          name: runtime\n          path: runtime-input\n";
    for step in [
        save, combined, verified, controller, automation, prune, artifact,
    ] {
        let error = mutated(UPLOAD_STEP, &format!("{step}{UPLOAD_STEP}")).unwrap_err();
        assert!(
            error.contains("denied") || error.contains("only the evidence artifact"),
            "{step}: {error}"
        );
    }
}

#[test]
fn linux_cache_measurement_refuses_reached_local_action_writers() {
    let error = mutated(
        "        uses: ./.github/actions/resolve-native-toolchain-epoch\n",
        "        uses: ./.github/actions/restore-test-model\n",
    )
    .unwrap_err();
    assert!(error.contains("restore-test-model"), "{error}");
    let error = mutated(
        "          allow_native_github_cache: \"false\"\n",
        "          allow_native_github_cache: \"true\"\n",
    )
    .unwrap_err();
    assert!(error.contains("literal false"), "{error}");
}

#[test]
fn linux_cache_measurement_refuses_write_permissions_and_extra_grants() {
    for (before, after) in [
        (
            "permissions:\n  contents: read\n\nenv:",
            "permissions:\n  contents: write\n\nenv:",
        ),
        (
            "    permissions:\n      contents: read\n      packages: read\n",
            "    permissions:\n      contents: read\n      packages: write\n",
        ),
        (
            "    permissions:\n      contents: read\n    steps:\n",
            "    permissions:\n      contents: read\n      id-token: write\n    steps:\n",
        ),
        (
            "    runs-on: ubuntu-24.04\n",
            "    runs-on: depot-ubuntu-24.04\n",
        ),
    ] {
        assert!(mutated(before, after).is_err(), "accepted: {after}");
    }
}

#[test]
fn linux_cache_measurement_restores_only_the_exact_production_key_in_warm_arms() {
    for (before, after) in [
        (RESTORE_IF, "        if: ${{ always() }}\n"),
        (
            "          path: runtime-input\n",
            "          path: runtime-input\n          restore-keys: skippy-runtime-linux-v2-\n",
        ),
        (
            "          key: skippy-runtime-linux-v2-${{ matrix.runtime.target }}",
            "          key: skippy-runtime-linux-v3-${{ matrix.runtime.target }}",
        ),
        (
            "          path: target/release/xtask\n",
            "          path: target/release/xtask\n          lookup-only: true\n",
        ),
    ] {
        assert!(mutated(before, after).is_err(), "accepted: {after}");
    }
}

#[test]
fn linux_cache_measurement_jobs_end_with_retained_evidence() {
    let error = mutated(
        "        if: ${{ !cancelled() }}\n        uses: actions/upload-artifact",
        "        if: ${{ success() }}\n        uses: actions/upload-artifact",
    )
    .unwrap_err();
    assert!(error.contains("evidence"), "{error}");
}

#[test]
fn linux_cache_measurement_writes_receipts_for_failed_preparations() {
    let guard = "        if: ${{ !cancelled() && steps.admit.outcome == 'success' }}\n";
    for (before, after) in [
        (
            "      - name: Write measurement receipt\n",
            "      - name: Write measurement receipt\n        if: ${{ success() }}\n",
        ),
        (
            "      - name: Admit measurement sample\n        id: admit\n",
            "      - name: Admit measurement sample\n",
        ),
    ] {
        let source = source();
        let mutated = if before.contains("Write") {
            source.replacen(&format!("{before}{guard}"), after, 1)
        } else {
            source.replacen(before, after, 1)
        };
        assert_ne!(mutated, source, "mutation did not apply: {before}");
        let error = verdict(&mutated).unwrap_err();
        assert!(
            error.contains("admitted sample") || error.contains("id admit"),
            "{error}"
        );
    }
    let error = mutated(
        "          PREPARE_OUTCOME: ${{ steps.prepare.outcome }}\n",
        "          PREPARE_OUTCOME: success\n",
    )
    .unwrap_err();
    assert!(error.contains("preparation outcome"), "{error}");
}
