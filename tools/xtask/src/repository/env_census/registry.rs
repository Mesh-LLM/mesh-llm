//! The reviewable census from `scripts/check-env-mutation-contract.py`. Every
//! entry is an explicit review decision; changing one needs the same review
//! as changing the legacy tables it mirrors.

pub(super) const TODO: &str =
    "// TODO: Audit that the environment access only happens in single-threaded code.";

/// The complete census of the 128 TODO comments this check was introduced to
/// audit. These files receive the strict serial-test/deferred-site checks.
pub(super) const AUDITED_FILES: &[&str] = &[
    "skippy/crates/skippy-protocol/build.rs",
    "mesh/crates/mesh-llm-plugin/build.rs",
    "mesh/crates/mesh-llm-config/src/env_overrides.rs",
    "mesh/crates/mesh-llm-host-runtime/src/plugin/config/tests.rs",
    "mesh/crates/mesh-llm-host-runtime/src/capture.rs",
    "mesh/crates/mesh-llm-membership/src/identity_persistence.rs",
    "mesh/crates/mesh-llm-host-runtime/src/mesh/public_identity_tests.rs",
    "mesh/crates/mesh-llm-host-runtime/src/network/nostr/keys.rs",
    "mesh/crates/mesh-llm-host-runtime/src/runtime/instance.rs",
    "mesh/crates/mesh-llm-host-runtime/src/models/maintenance.rs",
    "mesh/crates/mesh-llm-host-runtime/src/models/remote_catalog.rs",
    "mesh/crates/mesh-llm-host-runtime/src/models/artifact_transfer.rs",
    "mesh/crates/mesh-llm-host-runtime/src/models/delete_tests.rs",
    "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs",
    "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/materialization/package_download.rs",
    "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/materialization/cache_management.rs",
    "skippy/crates/skippy-model-hf/src/store/local.rs",
    "skippy/crates/skippy-model-hf/src/remote_catalog/tests.rs",
    "mesh/crates/mesh-llm-host-runtime/tests/membership_test_home_isolation.rs",
    "mesh/crates/mesh-llm-system/src/autoupdate.rs",
    "mesh/crates/mesh-llm-system/src/autoupdate/release_fetch/tests.rs",
    "mesh/crates/mesh-llm-system/src/benchmark/tests.rs",
    "skippy/crates/skippy-runtime/src/logging.rs",
    "mesh/crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
];

/// Mutations that predate the audit, frozen by exact file and count so a new
/// call or a new mutation-bearing file requires explicit review.
pub(super) const KNOWN_UNAUDITED_MUTATION_COUNTS: &[(&str, usize)] = &[
    (
        "mesh/crates/mesh-llm-host-runtime/src/api/routes/plugins.rs",
        3,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/api/tests/apply_config_diagnostics.rs",
        6,
    ),
    ("mesh/crates/mesh-llm-host-runtime/src/api/tests/mod.rs", 6),
    (
        "mesh/crates/mesh-llm-host-runtime/src/api/tests/runtime_config_validation_authority.rs",
        3,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/mesh/tests/admission/requirements.rs",
        6,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/mesh/tests/owner_control.rs",
        5,
    ),
    ("skippy/crates/skippy-model-hf/src/inventory.rs", 12),
    (
        "mesh/crates/mesh-llm-host-runtime/src/models/resolve/tests.rs",
        4,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/network/nostr/auto.rs",
        6,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/runtime/config_state_tests/support.rs",
        3,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/runtime/tests/auto_join.rs",
        1,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/runtime/tests/mod.rs",
        2,
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/runtime/tests/startup_models.rs",
        2,
    ),
    ("skippy/crates/skippy-runtime-install/src/lib.rs", 16),
    ("mesh/crates/mesh-llm/src/commands/plugin_cli.rs", 3),
    ("skippy/crates/skippy-model-hf/src/cache_paths.rs", 2),
];

/// Runtime sites that may already have Tokio worker threads; they keep an
/// explicit TODO until an ordering guarantee is established.
pub(super) const DEFERRED_FILES: &[&str] = &[
    "skippy/crates/skippy-runtime/src/logging.rs",
    "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "mesh/crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
];

/// Production mutations behind a real synchronous bootstrap boundary, with
/// the exact owning function.
pub(super) const SYNCHRONOUS_BOOTSTRAP_FILES: &[(&str, &str)] = &[(
    "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs",
    "configure_metal_pipeline_cache",
)];

/// Helpers that own scoped mutations for manually verified serial tests.
pub(super) const SERIAL_TEST_HELPERS: &[(&str, &[&str])] = &[
    (
        "mesh/crates/mesh-llm-host-runtime/tests/membership_test_home_isolation.rs",
        &["install", "drop"],
    ),
    (
        "mesh/crates/mesh-llm-config/src/env_overrides.rs",
        &["drop", "set", "with_env_override_for_test"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/capture.rs",
        &["drop"],
    ),
    (
        "mesh/crates/mesh-llm-membership/src/identity_persistence.rs",
        &["drop", "set"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/mesh/public_identity_tests.rs",
        &["drop", "set_home"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/network/nostr/keys.rs",
        &["drop", "set"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/materialization/cache_management.rs",
        &["restore_env"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/inference/skippy/materialization/package_download.rs",
        &["restore_env"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/models/artifact_transfer.rs",
        &["restore_env"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/models/delete_tests.rs",
        &["restore_env"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/models/maintenance.rs",
        &["restore_env"],
    ),
    (
        "mesh/crates/mesh-llm-host-runtime/src/runtime/instance.rs",
        &["drop", "save_and_remove", "save_and_set"],
    ),
    (
        "mesh/crates/mesh-llm-system/src/benchmark/tests.rs",
        &["drop", "with_benchmark_child_override"],
    ),
    (
        "skippy/crates/skippy-model-hf/src/store/local.rs",
        &["restore_env"],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The legacy script's docstring promises these totals; drift here means
    /// the two census copies disagree.
    #[test]
    fn migration_repository_census_tables_are_complete() {
        assert_eq!(AUDITED_FILES.len(), 25);
        assert_eq!(KNOWN_UNAUDITED_MUTATION_COUNTS.len(), 16);
        let frozen: usize = KNOWN_UNAUDITED_MUTATION_COUNTS.iter().map(|(_, n)| n).sum();
        assert_eq!(frozen, 80);
        assert!(
            DEFERRED_FILES
                .iter()
                .all(|file| AUDITED_FILES.contains(file))
        );
        assert!(
            SERIAL_TEST_HELPERS
                .iter()
                .all(|(file, _)| AUDITED_FILES.contains(file))
        );
    }
}
