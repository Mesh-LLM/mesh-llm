use super::scan::Candidate;
use crate::command::DynResult;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;
mod boundaries;
#[cfg(test)]
mod boundary_tests;
mod boundary_walk;
mod contexts;
#[cfg(test)]
mod contexts_tests;
#[cfg(test)]
mod contract_tests;
mod contracts;
mod edge_push;
mod inline_context;
#[cfg(test)]
mod inline_context_tests;
mod inline_launch;
#[cfg(test)]
mod inline_launch_tests;
mod just_bindings;
mod python_children;
#[cfg(test)]
mod python_children_tests;
mod python_targets;
#[cfg(test)]
mod recipe_binding_tests;
mod recipe_sources;
mod recipes;
mod rooted_scripts;
mod sources;
mod workflow_sources;
use contracts::Contracts;
use sources::{source_lines, tokens};

#[derive(Debug, Serialize)]
pub(super) struct Graph {
    pub(super) schema_version: u32,
    pub(super) roots: Vec<String>,
    pub(super) edges: Vec<boundaries::ClassifiedEdge>,
    pub(super) disposition_counts: BTreeMap<boundaries::EdgeDisposition, usize>,
    pub(super) unresolved: usize,
    pub(super) unbound_boundary_records: Vec<String>,
    pub(super) complete_census: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct Edge {
    pub(super) parent: String,
    pub(super) line: usize,
    pub(super) source_block: String,
    pub(super) trust_revision: &'static str,
    pub(super) status: &'static str,
    pub(super) child: Option<String>,
    pub(super) unresolved_reason: Option<String>,
    pub(super) argv: Option<String>,
    pub(super) status_streams_effects: Option<String>,
    pub(super) contract_source: Option<String>,
}

struct GraphBuilder<'a> {
    known: BTreeSet<&'a str>,
    queue: VecDeque<(String, &'static str)>,
    edges: Vec<Edge>,
    observed: &'a [Candidate],
    contracts: Contracts,
    validated: &'a BTreeSet<String>,
}

pub(super) fn report(
    root: &Path,
    paths: &[String],
    observed: &[Candidate],
    validated: &BTreeSet<String>,
    roots: &[&str],
) -> DynResult<Graph> {
    let known = paths.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let queue = roots
        .iter()
        .map(|path| {
            (
                (*path).to_owned(),
                if path.starts_with(".github/workflows/pr_") {
                    "protected_main_external"
                } else {
                    "same_commit"
                },
            )
        })
        .collect::<VecDeque<_>>();
    let mut visited = BTreeSet::new();
    let mut builder = GraphBuilder {
        known,
        queue,
        edges: Vec::new(),
        observed,
        contracts: Contracts::load(root)?,
        validated,
    };
    let mut recipes = BTreeSet::new();
    if roots.contains(&"Justfile") {
        recipes.insert("default".to_owned());
    }
    if roots.contains(&"just/ci.just") {
        recipes.extend(["ci-validate".to_owned(), "test-all".to_owned()]);
    }
    while let Some((path, trust)) = builder.queue.pop_front() {
        if !builder.known.contains(path.as_str()) || !root.join(&path).is_file() {
            return Err(format!("required graph: missing source {path}").into());
        }
        if !visited.insert((path.clone(), trust)) {
            continue;
        }
        let text = fs::read_to_string(root.join(&path))?;
        if path.ends_with(".py") && trust == "same_commit" {
            python_children::scan(&mut builder, &path, &text)?;
            continue;
        }
        for (line, block) in source_lines(&path, &text) {
            if path == "Justfile" || path.ends_with(".just") {
                for child in tokens(&block) {
                    builder.edges.push(Edge {
                        parent: path.clone(),
                        line,
                        source_block: block.clone(),
                        trust_revision: trust,
                        status: "reference_only",
                        child: Some(child.to_owned()),
                        unresolved_reason: None,
                        argv: None,
                        status_streams_effects: None,
                        contract_source: None,
                    });
                }
                continue;
            }
            if (path.ends_with(".yml") || path.ends_with(".yaml"))
                && let Some(recipe) = block
                    .trim_start_matches("run: ")
                    .strip_prefix("just ")
                    .and_then(|tail| tail.split_whitespace().next())
            {
                recipes.insert(recipe.to_owned());
            }
            if workflow_sources::record(&mut builder, &path, line, &block, trust)? {
                continue;
            }
            if inline_context::record(&mut builder, &path, line, &block, trust) {
                continue;
            }
            if path.ends_with(".py") {
                continue;
            }
            let candidate = observed
                .iter()
                .find(|row| row.path == path && row.source_block == block);
            let variable_script = inline_launch::variable_script_target(&block);
            if candidate.is_some_and(|row| !row.executable && validated.contains(&row.id))
                && !contexts::selected_interpreter_call(&block)
                && !variable_script
            {
                continue;
            }
            if rooted_scripts::record(&mut builder, &path, &text, (line, &block), trust)? {
                continue;
            }
            for child in tokens(&block) {
                builder.push_edge(&path, line, &block, child, trust)?;
            }
        }
    }
    recipes::expand(root, &mut builder, &mut visited, &recipes)?;
    let mut boundaries = boundaries::Boundaries::load(root)?;
    boundaries.walk_gated(root, &mut builder, &mut visited)?;
    builder
        .edges
        .sort_by(|a, b| (&a.parent, a.line, &a.child).cmp(&(&b.parent, b.line, &b.child)));
    let classified = boundaries.classify(root, builder.edges, observed, validated)?;
    Ok(Graph {
        schema_version: 2,
        roots: roots.iter().map(|root| (*root).to_owned()).collect(),
        edges: classified.edges,
        disposition_counts: classified.counts,
        unresolved: classified.unresolved,
        complete_census: classified.unresolved == 0 && classified.unbound_records.is_empty(),
        unbound_boundary_records: classified.unbound_records,
    })
}

pub(super) fn check_census(
    root: &Path,
    paths: &[String],
    observed: &[Candidate],
    validated: &BTreeSet<String>,
    roots: &[&str],
) -> DynResult<()> {
    if roots.is_empty() {
        return Ok(());
    }
    require_census(&report(root, paths, observed, validated, roots)?)
}

pub(super) fn require_census(graph: &Graph) -> DynResult<()> {
    if graph.complete_census {
        return Ok(());
    }
    Err(format!(
        "required graph: {} unresolved edge(s) lack a typed disposition; stale or unbound boundary records {:?}",
        graph.unresolved, graph.unbound_boundary_records
    )
    .into())
}
