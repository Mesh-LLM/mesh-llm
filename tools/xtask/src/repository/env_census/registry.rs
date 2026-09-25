//! The reviewable census from `scripts/check-env-mutation-contract.py`. Every
//! entry is an explicit review decision; changing one needs the same review
//! as changing the legacy tables it mirrors.

pub(super) const TODO: &str =
    "// TODO: Audit that the environment access only happens in single-threaded code.";

/// The complete census of the 128 TODO comments this check was introduced to
/// audit. These files receive the strict serial-test/deferred-site checks.
pub(super) const AUDITED_FILES: &[&str] = &[
    "crates/skippy-protocol/build.rs",
    "crates/mesh-llm-plugin/build.rs",
    "crates/mesh-llm-config/src/env_overrides.rs",
    "crates/mesh-llm-host-runtime/src/plugin/config/tests.rs",
    "crates/mesh-llm-host-runtime/src/capture.rs",
    "crates/mesh-llm-host-runtime/src/mesh/identity_persistence.rs",
    "crates/mesh-llm-host-runtime/src/mesh/public_identity_tests.rs",
    "crates/mesh-llm-host-runtime/src/network/nostr/keys.rs",
    "crates/mesh-llm-host-runtime/src/runtime/instance.rs",
    "crates/mesh-llm-host-runtime/src/models/maintenance.rs",
    "crates/mesh-llm-host-runtime/src/models/remote_catalog.rs",
    "crates/mesh-llm-host-runtime/src/models/artifact_transfer.rs",
    "crates/mesh-llm-host-runtime/src/models/delete_tests.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/package_download.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/cache_management.rs",
    "crates/model-hf/src/store/local.rs",
    "crates/mesh-llm-system/src/autoupdate.rs",
    "crates/mesh-llm-system/src/autoupdate/release_fetch.rs",
    "crates/mesh-llm-system/src/benchmark/tests.rs",
    "crates/skippy-runtime/src/logging.rs",
    "crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
];

/// Mutations that predate the audit, frozen by exact file and count so a new
/// call or a new mutation-bearing file requires explicit review.
pub(super) const KNOWN_UNAUDITED_MUTATION_COUNTS: &[(&str, usize)] = &[
    ("crates/mesh-llm-host-runtime/src/api/routes/plugins.rs", 3),
    (
        "crates/mesh-llm-host-runtime/src/api/tests/apply_config_diagnostics.rs",
        6,
    ),
    ("crates/mesh-llm-host-runtime/src/api/tests/mod.rs", 6),
    (
        "crates/mesh-llm-host-runtime/src/api/tests/runtime_config_validation_authority.rs",
        3,
    ),
    (
        "crates/mesh-llm-host-runtime/src/mesh/tests/admission/requirements.rs",
        6,
    ),
    (
        "crates/mesh-llm-host-runtime/src/mesh/tests/owner_control.rs",
        5,
    ),
    ("crates/mesh-llm-host-runtime/src/models/inventory.rs", 13),
    (
        "crates/mesh-llm-host-runtime/src/models/resolve/tests.rs",
        4,
    ),
    ("crates/mesh-llm-host-runtime/src/network/nostr/auto.rs", 6),
    (
        "crates/mesh-llm-host-runtime/src/runtime/config_state_tests/support.rs",
        3,
    ),
    (
        "crates/mesh-llm-host-runtime/src/runtime/tests/auto_join.rs",
        1,
    ),
    ("crates/mesh-llm-host-runtime/src/runtime/tests/mod.rs", 2),
    (
        "crates/mesh-llm-host-runtime/src/runtime/tests/startup_models.rs",
        2,
    ),
    ("crates/mesh-llm-runtime-install/src/lib.rs", 16),
    ("crates/mesh-llm/src/commands/plugin_cli.rs", 3),
    ("crates/model-hf/src/cache_paths.rs", 2),
];

/// Runtime sites that may already have Tokio worker threads; they keep an
/// explicit TODO until an ordering guarantee is established.
pub(super) const DEFERRED_FILES: &[&str] = &[
    "crates/skippy-runtime/src/logging.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
];

/// Production mutations behind a real synchronous bootstrap boundary, with
/// the exact owning function.
pub(super) const SYNCHRONOUS_BOOTSTRAP_FILES: &[(&str, &str)] = &[(
    "crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs",
    "configure_metal_pipeline_cache",
)];

/// Helpers that own scoped mutations for manually verified serial tests.
pub(super) const SERIAL_TEST_HELPERS: &[(&str, &[&str])] = &[
    (
        "crates/mesh-llm-config/src/env_overrides.rs",
        &["drop", "set", "with_env_override_for_test"],
    ),
    ("crates/mesh-llm-host-runtime/src/capture.rs", &["drop"]),
    (
        "crates/mesh-llm-host-runtime/src/mesh/identity_persistence.rs",
        &["drop", "set"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/mesh/public_identity_tests.rs",
        &["drop", "set_home"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/network/nostr/keys.rs",
        &["drop", "set"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/cache_management.rs",
        &["restore_env"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/package_download.rs",
        &["restore_env"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/models/artifact_transfer.rs",
        &["restore_env"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/models/delete_tests.rs",
        &["restore_env"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/models/maintenance.rs",
        &["restore_env"],
    ),
    (
        "crates/mesh-llm-host-runtime/src/runtime/instance.rs",
        &["drop", "save_and_remove", "save_and_set"],
    ),
    (
        "crates/mesh-llm-system/src/benchmark/tests.rs",
        &["drop", "with_benchmark_child_override"],
    ),
    ("crates/model-hf/src/store/local.rs", &["restore_env"]),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The legacy script's docstring promises these totals; drift here means
    /// the two census copies disagree.
    #[test]
    fn migration_repository_census_tables_are_complete() {
        assert_eq!(AUDITED_FILES.len(), 23);
        assert_eq!(KNOWN_UNAUDITED_MUTATION_COUNTS.len(), 16);
        let frozen: usize = KNOWN_UNAUDITED_MUTATION_COUNTS.iter().map(|(_, n)| n).sum();
        assert_eq!(frozen, 81);
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
