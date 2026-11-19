use super::*;

fn source(path: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path),
    )
    .unwrap()
}

#[test]
fn windows_abi_publication_actual_workflows_have_one_shared_producer() {
    check(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
}

#[test]
fn windows_abi_publication_rejects_automatic_or_duplicate_maintenance_production() {
    let original = source(WARMER);
    for (before, after, diagnostic) in [
        (
            "on:\n",
            "on:\n  push:\n    branches: [main]\n",
            "manual-only",
        ),
        (
            "      - name: Verify Windows CPU ABI cache\n",
            "      - uses: ./.github/actions/prepare-native-runtime-input\n      - name: Verify Windows CPU ABI cache\n",
            "must not build",
        ),
        (
            "      - name: Verify Windows CPU ABI cache\n",
            "      - uses: ./.github/actions/save-and-verify-actions-cache\n      - name: Verify Windows CPU ABI cache\n",
            "must not build",
        ),
    ] {
        let changed = original.replacen(before, after, 1);
        assert_ne!(changed, original);
        let document = workflow_yaml::parse(&changed).unwrap();
        assert!(
            maintenance(&document)
                .unwrap_err()
                .to_string()
                .contains(diagnostic)
        );
    }
}

#[test]
fn windows_abi_publication_rejects_untrusted_save_or_missing_recovery_path() {
    let original = source(RUNTIME);
    let (before_save, save) = original
        .split_once("      - name: Save exact shared Windows ABI build from trusted main\n")
        .unwrap();
    for clause in [
        "github.repository == 'Mesh-LLM/mesh-llm'",
        "github.ref == 'refs/heads/main'",
        "github.event_name == 'push'",
        "inputs.original_event_name == 'push'",
        "needs.runner_policy.outputs.allow_native_github_cache == 'true'",
        "steps.llama_cache.outputs.cache-hit != 'true'",
    ] {
        let changed = format!(
            "{before_save}      - name: Save exact shared Windows ABI build from trusted main\n{}",
            save.replacen(clause, "true", 1)
        );
        assert_ne!(changed, original);
        let document = workflow_yaml::parse(&changed).unwrap();
        assert!(
            publication(&document)
                .unwrap_err()
                .to_string()
                .contains(clause)
        );
    }
    for (before, after, diagnostic) in [
        (
            "uses: actions/cache/save@caa296126883cff596d87d8935842f9db880ef25 # v5.1.0\n        with:\n          path: ${{ steps.llama_cache.outputs.cache-path }}\n          key: ${{ steps.llama_cache.outputs.cache-primary-key }}",
            "uses: actions/cache/save@fixture\n        with:\n          path: ${{ steps.llama_cache.outputs.cache-path }}\n          key: ${{ steps.llama_cache.outputs.cache-primary-key }}",
            "require one exact",
        ),
        (
            "if: ${{ github.repository == 'Mesh-LLM/mesh-llm'",
            "if: ${{ always() && github.repository == 'Mesh-LLM/mesh-llm'",
            "implicit success",
        ),
        (
            "if: ${{ github.repository == 'Mesh-LLM/mesh-llm'",
            "if: ${{ false && github.repository == 'Mesh-LLM/mesh-llm'",
            // A never-true save satisfies every required clause, including the PR-only one.
            "require one exact",
        ),
        (
            "      - name: Prepare immutable Windows native runtime\n",
            "      - name: Prepare immutable Windows native runtime\n        if: false\n",
            "successful package",
        ),
        (
            "      - name: Save exact shared Windows ABI build from trusted main\n",
            "      - name: Save exact shared Windows ABI build from trusted main\n        continue-on-error: true\n",
            "successful package",
        ),
        (
            "      - name: Prepare immutable Windows native runtime\n",
            "      - name: Prepare immutable Windows native runtime\n        continue-on-error: true\n",
            "successful package",
        ),
        (
            "key: ${{ steps.llama_cache.outputs.cache-primary-key }}\n      - name: Save exact PR-scoped",
            "key: broad-key\n      - name: Save exact PR-scoped",
            "exact key",
        ),
        (
            "path: ${{ steps.llama_cache.outputs.cache-path }}\n          key: ${{ steps.llama_cache.outputs.cache-primary-key }}\n      - name: Save exact PR-scoped",
            "path: other-tree\n          key: ${{ steps.llama_cache.outputs.cache-primary-key }}\n      - name: Save exact PR-scoped",
            "exact path",
        ),
    ] {
        let changed = original.replacen(before, after, 1);
        assert_ne!(changed, original);
        let document = workflow_yaml::parse(&changed).unwrap();
        let error = publication(&document).unwrap_err().to_string();
        assert!(error.contains(diagnostic), "{after}: {error}");
    }
}

#[test]
fn windows_abi_publication_counts_a_save_that_merely_mentions_pull_request() {
    let original = source(RUNTIME);
    let extra = "      - name: Second shared Windows ABI save\n        if: ${{ inputs.original_event_name != 'pull_request' }}\n        uses: actions/cache/save@caa296126883cff596d87d8935842f9db880ef25 # v5.1.0\n        with:\n          path: ${{ steps.llama_cache.outputs.cache-path }}\n          key: ${{ steps.llama_cache.outputs.cache-primary-key }}\n";
    let changed = original.replacen(
        "      - name: Save exact PR-scoped Windows ABI build\n",
        &format!("{extra}      - name: Save exact PR-scoped Windows ABI build\n"),
        1,
    );
    assert_ne!(changed, original);
    let document = workflow_yaml::parse(&changed).unwrap();
    assert!(
        publication(&document)
            .unwrap_err()
            .to_string()
            .contains("require one exact")
    );
}

#[test]
fn windows_abi_publication_rejects_saving_before_package_verification() {
    let original = source(RUNTIME);
    let start = original
        .find("      - name: Save exact shared Windows ABI build from trusted main\n")
        .unwrap();
    let end = original
        .find("      - name: Save exact PR-scoped Windows ABI build\n")
        .unwrap();
    let save = &original[start..end];
    let without_save = format!("{}{}", &original[..start], &original[end..]);
    let changed = without_save.replacen(
        "      - name: Prepare immutable Windows native runtime\n",
        &format!("{save}      - name: Prepare immutable Windows native runtime\n"),
        1,
    );
    let document = workflow_yaml::parse(&changed).unwrap();
    assert!(
        publication(&document)
            .unwrap_err()
            .to_string()
            .contains("successful package")
    );
}

#[test]
fn windows_abi_publication_release_cannot_race_main_cpu_or_vulkan() {
    let original = source(".github/workflows/release.yml");
    for scope in [
        "github.ref != 'refs/heads/main' && steps.llama_cache.outputs.cache-hit != 'true'",
        "(github.ref != 'refs/heads/main' || matrix.backend != 'vulkan') && steps.llama_cache.outputs.cache-hit != 'true'",
    ] {
        let changed = original.replacen(scope, "steps.llama_cache.outputs.cache-hit != 'true'", 1);
        assert_ne!(changed, original);
        let document = workflow_yaml::parse(&changed).unwrap();
        assert!(
            release_ownership(&document)
                .unwrap_err()
                .to_string()
                .contains("shared main keys")
        );
    }
}
