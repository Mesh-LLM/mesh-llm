//! Behavioural parity tests for `cargo xtool ci-ops ...` ports.
//!
//! `runner-identity` replaces `scripts/runner-image-identity.py` (and the
//! evidence helpers it loads). Each case runs on a staged copy of the
//! checkout inputs and compares exit status, stdout and stderr with a golden
//! under `fixtures/ci_operations/runner_identity/`, captured from the legacy
//! script under Python 3.13 (`{root}` stands for the stage path). Default
//! runs start no Python. Set `MIGRATION_CI_OPERATIONS_LEGACY_PYTHON` to an
//! interpreter to also run the legacy script on identical inputs and require
//! identical bytes; add `MIGRATION_CI_OPERATIONS_CAPTURE=1` to rewrite the
//! goldens from those legacy runs.
//!
//! `build-cache` replaces `scripts/manage-build-cache.py`; its cases run on
//! fake cache trees with stub `ps` and `just`, goldens under
//! `fixtures/ci_operations/build_cache/` (`{root}` is the temp tree).
//!
//! `sccache-stats` replaces `.github/actions/capture-sccache-stats/capture.py`;
//! its cases run a stub `sccache` and compare the evidence and step-output
//! files too, goldens under `fixtures/ci_operations/sccache/`.
//!
//! `collect-metrics` replaces the offline JSON core of
//! `scripts/collect-ci-metrics.py`; its cases run on synthetic run JSON and
//! compare the written report too, goldens under
//! `fixtures/ci_operations/ci_metrics/` (`generated_at` masked).

#[path = "migration_ci_operations/build_cache.rs"]
mod build_cache;
#[path = "migration_ci_operations/ci_metrics.rs"]
mod ci_metrics;
#[path = "migration_ci_operations/ci_metrics_compare.rs"]
mod ci_metrics_compare;
#[path = "migration_ci_operations/ci_metrics_github.rs"]
mod ci_metrics_github;
#[path = "migration_ci_operations/ci_metrics_github_stub.rs"]
mod ci_metrics_github_stub;
#[path = "migration_ci_operations/runner_evidence.rs"]
mod runner_evidence;
#[path = "migration_ci_operations/runner_identity.rs"]
mod runner_identity;
#[path = "migration_ci_operations/sccache_stats.rs"]
mod sccache_stats;
#[path = "migration_ci_operations/support.rs"]
mod support;
