//! `ci-ops`: Rust owners of the runner, cache and CI metrics helper scripts.
//! Each command keeps its legacy script's argv, streams and exit statuses.

mod authority_audit;
mod authority_command;
mod build_cache;
mod build_cache_argv;
mod build_cache_cargo;
mod build_cache_help;
pub(crate) mod build_cache_options;
mod build_cache_prune;
mod build_cache_tree;
mod build_cache_values;
mod cache_configuration;
mod cache_configuration_command;
mod cache_home;
mod catalog_contracts;
mod catalog_release_pair;
mod catalog_validation;
mod chat_display;
mod ci_metrics;
mod ci_metrics_aggregate;
mod ci_metrics_analyze;
mod ci_metrics_argv;
mod ci_metrics_calendar;
mod ci_metrics_compare;
mod ci_metrics_github;
mod ci_metrics_input;
pub(crate) mod ci_metrics_int;
mod ci_metrics_markdown;
mod ci_metrics_markdown_format;
mod ci_metrics_markdown_jobs;
mod ci_metrics_normalize;
mod ci_metrics_observe;
mod ci_metrics_report;
mod ci_metrics_rollup;
mod ci_metrics_runner;
mod ci_metrics_stats;
pub(crate) mod ci_metrics_time;
pub(crate) mod ci_metrics_value;
mod evidence_binding;
mod evidence_catalog;
mod evidence_input;
mod evidence_platforms;
mod evidence_timestamp;
mod identity_text;
pub(crate) mod json_access;
pub(crate) mod json_decode;
mod registry_pulls;
mod runner_cleanup;
mod runner_identity;
pub(crate) mod runner_identity_argv;
mod runner_identity_check;
mod runner_identity_help;
mod runner_identity_subargs;
mod runtime_seed;
mod runtime_seed_io;
mod runtime_seed_preflight;
mod runtime_seed_stats;
mod runtime_seed_summary;
#[cfg(test)]
mod runtime_seed_tests;
mod runtime_seed_types;
mod sccache_argv;
mod sccache_evidence;
mod sccache_render;
mod sccache_stats;
mod sdk_census;
mod seed_census;
mod workflow_bindings;
mod workflow_census;
mod workflow_text;

use crate::command::DynResult;
use std::path::PathBuf;

/// A `ci-ops` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum CiOperationsCommand {
    RunnerIdentity,
    RunnerCleanup,
    BuildCache,
    SccacheStats,
    CollectMetrics,
    AuthorityAudit,
    RegistryPulls,
    ChatDisplay,
    RuntimeSeed,
    ConfigureCanaryCache,
}

impl CiOperationsCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "runner-identity" => Some(Self::RunnerIdentity),
            "runner-cleanup" => Some(Self::RunnerCleanup),
            "build-cache" => Some(Self::BuildCache),
            "sccache-stats" => Some(Self::SccacheStats),
            "collect-metrics" => Some(Self::CollectMetrics),
            "authority-audit" => Some(Self::AuthorityAudit),
            "registry-pulls" => Some(Self::RegistryPulls),
            "chat-display" => Some(Self::ChatDisplay),
            "runtime-seed" => Some(Self::RuntimeSeed),
            "configure-canary-cache" => Some(Self::ConfigureCanaryCache),
            _ => None,
        }
    }
}

/// `root` resolves the default checkout lazily, as the legacy scripts used
/// their own location only when no explicit root was given.
pub(crate) fn run(
    command: CiOperationsCommand,
    args: &[String],
    root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    if matches!(command, CiOperationsCommand::ConfigureCanaryCache) {
        return cache_configuration_command::run(args);
    }
    if matches!(command, CiOperationsCommand::ChatDisplay) {
        return chat_display::run(args);
    }
    if matches!(command, CiOperationsCommand::RunnerCleanup) {
        return runner_cleanup::command::run(args);
    }
    let report = match command {
        CiOperationsCommand::ConfigureCanaryCache => unreachable!(),
        CiOperationsCommand::RunnerCleanup => unreachable!(),
        CiOperationsCommand::ChatDisplay => unreachable!(),
        CiOperationsCommand::AuthorityAudit => authority_command::run(args),
        CiOperationsCommand::RuntimeSeed => runtime_seed::run(args),
        CiOperationsCommand::RegistryPulls => {
            let output = registry_pulls::run(args)?;
            crate::repository::check_report::CheckReport {
                stdout: output.stdout,
                stderr: String::new(),
                code: output.code,
            }
        }
        CiOperationsCommand::BuildCache => build_cache::run(args),
        CiOperationsCommand::SccacheStats => sccache_stats::run(args),
        CiOperationsCommand::CollectMetrics => ci_metrics::run(args),
        CiOperationsCommand::RunnerIdentity => {
            let mut failure = None;
            let report = runner_identity::run(args, || {
                root().unwrap_or_else(|error| {
                    let fallback = std::env::current_dir().unwrap_or_default();
                    failure = Some(error);
                    fallback
                })
            });
            if let Some(error) = failure {
                return Err(error);
            }
            report
        }
    };
    report.emit()
}
