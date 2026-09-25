//! `ci-ops`: Rust owners of the runner, cache and CI metrics helper scripts.
//! Each command keeps its legacy script's argv, streams and exit statuses.

mod build_cache;
mod build_cache_argv;
mod build_cache_cargo;
mod build_cache_help;
mod build_cache_options;
mod build_cache_prune;
mod build_cache_tree;
mod build_cache_values;
mod catalog_contracts;
mod catalog_release_pair;
mod catalog_validation;
mod ci_metrics;
mod ci_metrics_aggregate;
mod ci_metrics_analyze;
mod ci_metrics_argv;
mod ci_metrics_calendar;
mod ci_metrics_compare;
mod ci_metrics_github;
mod ci_metrics_input;
mod ci_metrics_int;
mod ci_metrics_markdown;
mod ci_metrics_markdown_format;
mod ci_metrics_markdown_jobs;
mod ci_metrics_normalize;
mod ci_metrics_observe;
mod ci_metrics_report;
mod ci_metrics_rollup;
mod ci_metrics_runner;
mod ci_metrics_stats;
mod ci_metrics_time;
mod ci_metrics_value;
mod evidence_binding;
mod evidence_catalog;
mod evidence_input;
mod evidence_platforms;
mod evidence_timestamp;
mod identity_text;
mod python_access;
mod python_json_decode;
mod python_json_strings;
mod runner_identity;
mod runner_identity_argv;
mod runner_identity_check;
mod runner_identity_help;
mod runner_identity_subargs;
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
    BuildCache,
    SccacheStats,
    CollectMetrics,
}

impl CiOperationsCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "runner-identity" => Some(Self::RunnerIdentity),
            "build-cache" => Some(Self::BuildCache),
            "sccache-stats" => Some(Self::SccacheStats),
            "collect-metrics" => Some(Self::CollectMetrics),
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
    let report = match command {
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
