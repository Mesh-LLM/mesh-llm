mod action_pins;
mod authority_delivery_body;
mod authority_sources;
mod build_acceleration;
mod cache_authority;
mod cache_boundaries;
mod cache_callers;
mod cache_consumers;
mod cache_evidence;
mod cache_identity;
mod cache_marker;
mod cache_predicate;
mod canary_build;
mod canary_execution;
mod canary_graph;
mod canary_sdk;
mod cancellation;
mod claude_clients;
mod compute_changes_budget;
mod cpu_runtime_cache;
mod crates_recovery;
mod handoffs;
mod laya;
mod nightly_stability;
mod permissions;
mod pr_canary;
mod quality_contracts;
mod registry_canary;
mod replay;
mod replay_admission;
mod replay_environment;
mod runner_finalization;
mod runtime_events;
mod selected_ref;
mod shell;
mod windows_build_paths;

use super::lane_results::workflow_yaml::{self, Node};
use crate::command::DynResult;
use std::collections::BTreeMap;
use std::path::Path;

pub(super) fn check(root: &Path) -> DynResult<()> {
    build_acceleration::check(root)?;
    windows_build_paths::check(root)?;
    compute_changes_budget::check(root)?;
    action_pins::check(root)?;
    let mut workflows = BTreeMap::new();
    for entry in std::fs::read_dir(root.join(".github/workflows"))? {
        let path = entry?.path();
        if !matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("yml" | "yaml")
        ) {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("invalid workflow name")?
            .to_owned();
        let source = std::fs::read_to_string(&path)?;
        shell::check_expressions(&source).map_err(|error| format!("{name}: {error}"))?;
        let document = if name == "llama-upstream-canary.yml" {
            workflow_yaml::parse_resolved_aliases(&source)
        } else {
            workflow_yaml::parse(&source)
        }
        .map_err(|error| format!("{name}: {error}"))?;
        shell::check_containers(&document).map_err(|error| format!("{name}: {error}"))?;
        workflows.insert(name, document);
    }
    cancellation::check(&workflows)?;
    quality_contracts::check(root, &workflows)?;
    selected_ref::check(&workflows)?;
    claude_clients::check(&workflows)?;
    pr_canary::check(&workflows)?;
    runner_finalization::check(&workflows)?;
    runtime_events::check(&workflows)?;
    replay::check(&workflows)?;
    replay_admission::check(root, &workflows)?;
    replay_environment::check(&workflows)?;
    nightly_stability::check(&workflows)?;
    crates_recovery::check(&workflows)?;
    registry_canary::check(&workflows)?;
    canary_graph::check(&workflows)?;
    canary_build::check(&workflows)?;
    laya::check(root, &workflows)?;
    let quality_jobs = workflows
        .get("ci-quality-slice.yml")
        .and_then(|workflow| workflow.get("jobs"))
        .ok_or("required cache authority workflow missing")?;
    cache_authority::producer(
        quality_jobs
            .get("runner_policy")
            .ok_or("required cache authority producer missing")?,
    )?;
    cache_authority::sentinel(
        quality_jobs
            .get("authority_sentinel")
            .ok_or("required cache authority sentinel missing")?,
    )?;
    cache_marker::check(&workflows)?;
    authority_sources::check(&workflows)?;
    authority_sources::documentation(&std::fs::read_to_string(
        root.join("ci/DEPOT_MIGRATION.md"),
    )?)?;
    cpu_runtime_cache::check(&workflows)?;
    cache_consumers::check(&workflows)?;
    cache_callers::check(&workflows)?;
    cache_boundaries::check(&workflows)?;
    cache_evidence::check(&workflows)?;
    cache_identity::check(&workflows)?;
    let capture =
        std::fs::read_to_string(root.join(".github/actions/capture-sccache-stats/action.yml"))?;
    cache_evidence::capture_action(&workflow_yaml::parse(&capture)?)?;
    let configure =
        std::fs::read_to_string(root.join(".github/actions/configure-sccache-gha/action.yml"))?;
    cache_evidence::configure_action(&workflow_yaml::parse(&configure)?)?;
    for name in ["restore-windows-abi-cache", "setup-windows-rocm-sdk"] {
        let source =
            std::fs::read_to_string(root.join(format!(".github/actions/{name}/action.yml")))?;
        cache_callers::nested_windows(&workflow_yaml::parse(&source)?)?;
    }
    permissions::check(&workflows)
}

fn field<'a>(node: &'a Node, name: &str) -> Option<&'a str> {
    node.get(name).and_then(Node::text)
}

/// Reuse the permission owner's declaration semantics for native entrypoint authority.
pub(super) fn job_permission_is_none(
    document: &Node,
    job: &Node,
    permission: &str,
) -> Result<bool, String> {
    permissions::effective_none(document, job, permission)
}

#[cfg(test)]
mod build_accelerator_sources;
