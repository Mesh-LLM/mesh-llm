use super::ledger::{GithubEdge, MigrationLedgers};
use super::loader_closure;
use super::required_closure;
use super::scan;
use super::selected_process;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::path::Path;

fn check_paths(root: &Path, paths: &[String], ledgers: &MigrationLedgers) -> DynResult<()> {
    let actual = paths
        .iter()
        .filter(|path| path.ends_with(".py"))
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut expected = BTreeSet::new();
    for file in &ledgers.inventory.files {
        if !expected.insert(file.path.clone())
            || file.classification.is_empty()
            || file.replacement_owner.is_empty()
            || file.deletion_condition.is_empty()
        {
            return Err(format!(
                "python inventory: duplicate or incomplete path {}",
                file.path
            )
            .into());
        }
    }
    let missing = actual.difference(&expected).collect::<Vec<_>>();
    let stale = expected.difference(&actual).collect::<Vec<_>>();
    if !missing.is_empty() || !stale.is_empty() {
        return Err(format!("python inventory: missing {missing:?}; stale {stale:?}").into());
    }
    for path in expected {
        if !root.join(&path).is_file() {
            return Err(format!("python inventory: stale file {path}").into());
        }
    }
    Ok(())
}

fn github_edge(observed: &[scan::Candidate], edge: &GithubEdge) -> DynResult<()> {
    if edge.owner.is_empty()
        || edge.argv.is_empty()
        || edge.status_output_effects.is_empty()
        || edge.transitive_boundary.is_empty()
        || edge.transitive_boundary.starts_with("unresolved:")
        || edge.replacement.is_empty()
        || edge.deletion_condition.is_empty()
        || edge.reason.is_empty()
        || !matches!(edge.disposition.as_str(), "execution" | "nonexecution")
    {
        return Err(format!("automation inventory: incomplete GitHub edge {}", edge.id).into());
    }
    let source = observed.iter().find(|row| row.id == edge.id);
    if source.is_some_and(|row| row.source_block != edge.source_block) {
        return Err(format!("automation inventory: changed source {}", edge.id).into());
    }
    if source.is_none() {
        return Err(format!("automation inventory: stale GitHub edge {}", edge.id).into());
    }
    Ok(())
}

fn check_edges(
    root: &Path,
    paths: &[String],
    observed: &[scan::Candidate],
    ledgers: &MigrationLedgers,
    validated: &BTreeSet<String>,
    roots: &[&str],
) -> DynResult<()> {
    let _ = ledgers.invocations.reproducible_candidate_census.acceptance;
    let mut accepted = BTreeSet::new();
    for edge in &ledgers.invocations.source_verified_edges {
        if edge.owner.is_empty()
            || edge.reason.is_empty()
            || edge.target.is_empty()
            || edge.target.starts_with("unresolved:")
            || edge.replacement_task == 0
        {
            return Err(format!("automation inventory: incomplete edge {}", edge.id).into());
        }
        if !accepted.insert(edge.id.clone()) {
            return Err(format!("automation inventory: duplicate edge {}", edge.id).into());
        }
    }
    for edge in &ledgers.invocations.github_source_records {
        github_edge(observed, edge)?;
        if !accepted.insert(edge.id.clone()) {
            return Err(format!("automation inventory: duplicate edge {}", edge.id).into());
        }
    }
    let seen = observed
        .iter()
        .map(|row| row.id.clone())
        .collect::<BTreeSet<_>>();
    let stale = accepted
        .difference(&seen)
        .take(5)
        .cloned()
        .collect::<Vec<_>>();
    let unowned = observed
        .iter()
        .filter(|row| row.executable && !accepted.contains(&row.id) && !validated.contains(&row.id))
        .collect::<Vec<_>>();
    if !stale.is_empty() || !unowned.is_empty() {
        let examples = unowned
            .iter()
            .take(5)
            .map(|row| format!("{} [{}]", row.path, row.id))
            .collect::<Vec<_>>();
        return Err(format!("automation inventory: {} stale edges {stale:?}; {} unclassified executable candidates {examples:?}; task 1 remains open", stale.len(), unowned.len()).into());
    }
    required_closure::check_required_closure(root, paths, observed, validated, roots)?;
    super::required_graph::check_census(root, paths, observed, validated, roots)?;
    selected_process::check_selected_processes(root, &ledgers.invocations.selected_process_calls)?;
    if paths
        .iter()
        .any(|path| path == "scripts/runner-image-identity.py")
    {
        loader_closure::check_runner_image_planner_loader(root, &ledgers.invocations)?;
    }
    if paths
        .iter()
        .any(|path| path == "scripts/llama-canary-family-evidence.py")
    {
        loader_closure::check_family_canary_loaders(root, &ledgers.invocations)?;
    }
    Ok(())
}

pub(super) fn check_inventory(
    root: &Path,
    paths: &[String],
    ledgers: &MigrationLedgers,
    observed: &[scan::Candidate],
    validated: &BTreeSet<String>,
) -> DynResult<()> {
    let roots = required_closure::required_roots(root, paths)?;
    let refs = roots.iter().map(String::as_str).collect::<Vec<_>>();
    check_inventory_from_roots(root, paths, ledgers, observed, validated, &refs)
}

pub(super) fn check_inventory_from_roots(
    root: &Path,
    paths: &[String],
    ledgers: &MigrationLedgers,
    observed: &[scan::Candidate],
    validated: &BTreeSet<String>,
    roots: &[&str],
) -> DynResult<()> {
    ledgers.versions()?;
    check_paths(root, paths, ledgers)?;
    check_edges(root, paths, observed, ledgers, validated, roots)
}

pub(super) fn check_policy(
    root: &Path,
    paths: &[String],
    ledgers: &MigrationLedgers,
    observed: &[scan::Candidate],
    validated: &BTreeSet<String>,
) -> DynResult<()> {
    let roots = required_closure::required_roots(root, paths)?;
    let refs = roots.iter().map(String::as_str).collect::<Vec<_>>();
    check_policy_from_roots(root, paths, ledgers, observed, validated, &refs)
}

pub(super) fn check_policy_from_roots(
    root: &Path,
    paths: &[String],
    ledgers: &MigrationLedgers,
    observed: &[scan::Candidate],
    validated: &BTreeSet<String>,
    roots: &[&str],
) -> DynResult<()> {
    ledgers.versions()?;
    check_paths(root, paths, ledgers)?;
    let actual = paths
        .iter()
        .filter(|path| scan::is_instruction(path))
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut expected = BTreeSet::new();
    for asset in &ledgers.instructions.assets {
        if !expected.insert(asset.path.clone())
            || asset.classification.is_empty()
            || asset.owner.is_empty()
        {
            return Err(format!(
                "automation policy: duplicate or incomplete instruction {}",
                asset.path
            )
            .into());
        }
    }
    if actual != expected {
        return Err(format!(
            "automation policy: instruction path drift; missing {:?}; stale {:?}",
            actual.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&actual).collect::<Vec<_>>()
        )
        .into());
    }
    super::exception_policy::check_exceptions(paths, ledgers)?;
    check_edges(root, paths, observed, ledgers, validated, roots)
}
