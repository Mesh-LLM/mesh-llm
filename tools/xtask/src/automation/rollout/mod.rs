mod bootstrap;
mod evidence;
mod git;
mod input;

#[cfg(test)]
#[path = "../../../tests/migration_rollout/mod.rs"]
mod tests;

use crate::command::DynResult;
use evidence::EvidenceFile;
use input::{CommitSha, Request};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::Path;

pub(crate) const USAGE: &str = "cargo xtool automation rollout --input <local-observations.json>";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Issue {
    Input,
    Evidence,
    Sequence,
    Catalog,
    Authority,
    Bootstrap,
    Predecessor,
    Policy,
}

#[derive(Debug)]
pub(crate) struct Rejected {
    pub(crate) issue: Issue,
    pub(crate) detail: String,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rollout {:?}: {}", self.issue, self.detail)
    }
}

impl std::error::Error for Rejected {}

type Checked<T> = Result<T, Rejected>;

fn reject(issue: Issue, detail: impl Into<String>) -> Rejected {
    Rejected {
        issue,
        detail: detail.into(),
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Scope {
    LocalFactConsistencyOnly,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Acceptance {
    NotEstablished,
}

#[derive(Debug, Serialize)]
pub(crate) struct LocalFacts {
    scope: Scope,
    task26_acceptance: Acceptance,
    protected_sha: CommitSha,
    source_sha: CommitSha,
    catalog_sha: CommitSha,
    support_sha: CommitSha,
    observations: EvidenceFile,
    catalogs: Vec<evidence::CatalogBinding>,
    predecessor_files: usize,
    externally_unverified: [&'static str; 4],
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let [flag, path] = args else {
        return Err(format!("usage: {USAGE}").into());
    };
    if flag != "--input" {
        return Err(format!("usage: {USAGE}").into());
    }
    crate::command::print_json(&validate_file(Path::new(path))?)
}

pub(crate) fn validate_file(path: &Path) -> Checked<LocalFacts> {
    let bytes = evidence::read(path)?;
    let request: Request =
        serde_json::from_slice(&bytes).map_err(|error| reject(Issue::Input, error.to_string()))?;
    let base = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let repository = git::Repository::open(&base.join(&request.repository))?;
    let catalogs = repository.validate_sequence(&request)?;
    validate_authority(&request)?;
    bootstrap::validate(base, &request)?;
    validate_predecessors(base, &request)?;
    Ok(LocalFacts {
        scope: Scope::LocalFactConsistencyOnly,
        task26_acceptance: Acceptance::NotEstablished,
        protected_sha: request.protected_sha,
        source_sha: request.source_sha,
        catalog_sha: request.catalog_sha,
        support_sha: request.support_sha,
        observations: EvidenceFile::from_bytes(path.to_path_buf(), &bytes),
        catalogs,
        predecessor_files: request.predecessors.len(),
        externally_unverified: [
            "protected_default_branch_and_merge_rebase_truth",
            "bootstrap_execution_and_binary_source_provenance",
            "tasks_11_through_25_acceptance_including_platform_and_model_evidence",
            "provider_administration_and_external_operation_authorization",
        ],
    })
}

fn validate_authority(request: &Request) -> Checked<()> {
    for (owner, revision) in [
        ("planner", &request.planner_sha),
        ("workspace discovery", &request.workspace_sha),
        ("runner policy", &request.runner_policy_sha),
        ("bootstrap", &request.bootstrap.source_sha),
    ] {
        if revision != &request.protected_sha {
            return Err(reject(
                Issue::Authority,
                format!("{owner} is not the protected revision"),
            ));
        }
    }
    Ok(())
}

fn validate_predecessors(base: &Path, request: &Request) -> Checked<()> {
    let mut tasks = BTreeSet::new();
    for predecessor in &request.predecessors {
        if !(11..=25).contains(&predecessor.task) || !tasks.insert(predecessor.task) {
            return Err(reject(
                Issue::Predecessor,
                "duplicate or out-of-range predecessor task",
            ));
        }
        let bytes = predecessor.evidence.read(base)?;
        if bytes.is_empty() {
            return Err(reject(
                Issue::Predecessor,
                format!("task {} has empty evidence", predecessor.task),
            ));
        }
    }
    if tasks != (11..=25).collect() {
        return Err(reject(
            Issue::Predecessor,
            "evidence references are required for every task 11 through 25",
        ));
    }
    Ok(())
}
