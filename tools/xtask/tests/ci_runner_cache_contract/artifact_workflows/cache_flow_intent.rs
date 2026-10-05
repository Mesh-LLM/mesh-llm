//! Workflow-specific exact restore, verification and publication declarations.
use super::{document, input, job, named, steps, text};
fn position(node: &super::Node, name: &str) -> usize {
    steps(node)
        .iter()
        .position(|s| text(s, "name") == Some(name))
        .unwrap()
}
#[test]
fn graph_skippy_seed_restore_verify_publish_keeps_exact_trusted_revision_and_no_hit_bypass() {
    let doc = document("ci-rust-tests-slice.yml");
    let rust = job(&doc, "rust_tests");
    let restore_name = "Restore Skippy correctness model cache when required";
    let download_name = "Download Skippy correctness model when required";
    let verify_name = "Verify Skippy correctness model when required";
    let save_name = "Save trusted Skippy correctness model cache";
    let restore = named(rust, restore_name);
    let download = named(rust, download_name);
    let verify = named(rust, verify_name);
    let save = named(rust, save_name);
    assert!(position(rust, restore_name) < position(rust, download_name));
    assert!(position(rust, download_name) < position(rust, verify_name));
    assert!(position(rust, verify_name) < position(rust, save_name));
    assert!(
        text(restore, "uses")
            .unwrap()
            .starts_with("actions/cache/restore@")
    );
    assert!(
        text(restore, "if")
            .unwrap()
            .contains("needs.runner_policy.outputs.allow_native_github_cache == 'true'")
    );
    assert!(input(restore, "restore-keys").is_none());
    assert!(
        input(restore, "key")
            .unwrap()
            .contains("steps.skippy_correctness_model.outputs.sha256")
    );
    let download = text(download, "run").unwrap();
    assert!(
        download.contains("--revision \"${{ steps.skippy_correctness_model.outputs.revision }}\"")
    );
    let guard = text(verify, "if").unwrap();
    assert!(!guard.contains("cache-hit"));
    let run = text(verify, "run").unwrap();
    assert!(run.contains("cargo xtool models resolve"));
    assert!(run.contains("ci/model-artifacts/manifests/skippy-correctness.json"));
    assert!(run.contains("--verify-root \"$RUNNER_TEMP/skippy-correctness-model\""));
    let guard = text(save, "if").unwrap();
    for required in [
        "contains(steps.resolve_batch_crates.outputs.crates, 'skippy-runtime')",
        "needs.runner_policy.outputs.allow_native_github_cache == 'true'",
        "github.ref == 'refs/heads/main'",
        "inputs.original_event_name != 'pull_request'",
        "inputs.original_event_name != 'pull_request_target'",
        "steps.skippy_correctness_model_cache.outputs.cache-hit != 'true'",
    ] {
        assert!(
            crate::cache_predicate::requires(guard, required),
            "{required}"
        );
    }
    assert!(
        text(save, "uses")
            .unwrap()
            .starts_with("actions/cache/save@")
    );
    assert_eq!(
        input(save, "key"),
        Some("${{ steps.skippy_correctness_model_cache.outputs.cache-primary-key }}")
    );
    assert!(
        !steps(rust)
            .iter()
            .any(|s| text(s, "uses").is_some_and(|u| u.starts_with("Swatinem/rust-cache@")))
    );
}
#[test]
fn graph_windows_vulkan_cache_save_is_denied_for_dispatched_pull_requests() {
    let doc = document("ci-windows-runtime-slice.yml");
    let runtime = job(&doc, "windows_runtime");
    let candidates = steps(runtime)
        .iter()
        .filter(|s| input(s, "cache_save_if").is_some())
        .collect::<Vec<_>>();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        input(candidates[0], "cache"),
        Some("${{ needs.runner_policy.outputs.allow_native_github_cache == 'true' }}")
    );
    assert_eq!(
        input(candidates[0], "cache_save_if"),
        Some(
            "${{ inputs.original_event_name != 'pull_request' && inputs.original_event_name != 'pull_request_target' }}"
        )
    );
}
#[test]
fn graph_windows_pr_abi_saves_require_exact_restored_identity_and_closed_scope() {
    let runtime = document("ci-windows-runtime-slice.yml");
    let runtime = job(&runtime, "windows_runtime");
    let platform = document("ci-platform-checks-slice.yml");
    let matches = platform
        .get("jobs")
        .unwrap()
        .entries()
        .iter()
        .filter_map(|(_, j)| j.get("steps").map(|_| j))
        .filter(|j| {
            steps(j)
                .iter()
                .any(|s| text(s, "name") == Some("Save static Metal ABI build"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    for (owner, name, scope) in [
        (
            runtime,
            "Save exact PR-scoped Windows ABI build",
            "inputs.original_event_name == 'pull_request'",
        ),
        (
            matches[0],
            "Save static Metal ABI build",
            "inputs.original_event_name == 'pull_request' || (github.ref == 'refs/heads/main' && inputs.original_event_name != 'pull_request_target')",
        ),
    ] {
        let save = named(owner, name);
        assert!(
            text(save, "uses")
                .unwrap()
                .starts_with("actions/cache/save@")
        );
        assert_eq!(
            input(save, "key"),
            Some("${{ steps.llama_cache.outputs.cache-primary-key }}")
        );
        assert!(input(save, "restore-keys").is_none());
        let condition = text(save, "if").unwrap();
        for clause in [
            "needs.runner_policy.outputs.allow_native_github_cache == 'true'",
            "steps.llama_cache.outputs.cache-hit != 'true'",
            scope,
        ] {
            assert!(
                crate::cache_predicate::requires(condition, clause),
                "{name}:{clause}"
            );
        }
        for s in steps(owner) {
            if text(s, "uses").is_some_and(|u| u.starts_with("actions/cache")) {
                assert!(input(s, "restore-keys").is_none());
            }
        }
    }
}
#[test]
fn graph_rust_trusted_compiler_seed_restores_forward_the_central_decision() {
    let doc = document("ci-rust-tests-slice.yml");
    let mut restores = 0;
    for (_, j) in doc.get("jobs").unwrap().entries() {
        if j.get("steps").is_none() {
            continue;
        }
        for s in steps(j) {
            let uses = text(s, "uses").unwrap_or("");
            assert!(!uses.starts_with("Swatinem/rust-cache@"));
            if uses == "./.github/actions/restore-sccache-seed" {
                restores += 1;
                assert_eq!(
                    input(s, "allow_trusted_seed"),
                    Some("${{ needs.runner_policy.outputs.allow_trusted_sccache_seed }}")
                );
                assert!(
                    input(s, "cache_key")
                        .unwrap()
                        .starts_with("mesh-llm-sccache-seed-linux-x86_64-img-")
                );
                assert!(input(s, "cache_key").unwrap().contains("-epoch-"));
            }
        }
    }
    assert!(restores > 0);
}
