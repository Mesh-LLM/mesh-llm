//! Protected metadata events select only a PR number, never a verdict or source.
use super::*;
use serde::Deserialize;
use serde_json::Value;
use std::io::Read;

const PUBLISHER_REF: &str =
    "Mesh-LLM/mesh-llm/.github/workflows/ci-governance-publisher.yml@refs/heads/main";
const RELAY_PATH: &str = ".github/workflows/ci-governance-review-signal.yml";

pub(super) enum Trigger {
    Lifecycle { repository_id: u64, number: u64 },
    ReviewRun { repository_id: u64, run_id: u64 },
    Reconcile { repository_id: u64 },
}

impl Trigger {
    pub(super) fn resolve_expected(
        self,
        client: &mut impl GithubRead,
        anchors: &activation::Anchors,
        head: &str,
        expected: u64,
    ) -> DynResult<u64> {
        if let Self::Reconcile { repository_id } = self {
            if repository_id != anchors.repository_id
                || collector::read_identity(client, expected, anchors)?.main_sha != head
            {
                return Err("batch publisher target is not open on exact protected main".into());
            }
            return Ok(expected);
        }
        let number = self.resolve(client, anchors)?;
        if number != expected {
            return Err(
                "publisher event target changed after authenticated concurrency resolution".into(),
            );
        }
        Ok(number)
    }
    pub(super) fn load(root: &Path, head: &str) -> DynResult<Self> {
        for (name, expected) in [
            ("GITHUB_REPOSITORY", "Mesh-LLM/mesh-llm"),
            ("GITHUB_REF", "refs/heads/main"),
            ("GITHUB_SHA", head),
            ("GITHUB_WORKFLOW_SHA", head),
            ("GITHUB_WORKFLOW_REF", PUBLISHER_REF),
        ] {
            if std::env::var(name).ok().as_deref() != Some(expected) {
                return Err("publisher requires its exact protected-main workflow context".into());
            }
        }
        let path = std::env::var_os("GITHUB_EVENT_PATH")
            .ok_or("publisher GitHub event file is unavailable")?;
        let mut bytes = Vec::new();
        std::fs::File::open(root.join(path))?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1024 * 1024 {
            return Err("publisher event exceeds supported bound".into());
        }
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|_| "invalid publisher event JSON")?;
        let repository_id = positive(&value, "/repository/id")?;
        if value
            .pointer("/repository/full_name")
            .and_then(Value::as_str)
            != Some("Mesh-LLM/mesh-llm")
        {
            return Err("publisher event repository differs from the protected controller".into());
        }
        match std::env::var("GITHUB_EVENT_NAME").ok().as_deref() {
            Some("push") if value["ref"] == "refs/heads/main" => {
                Ok(Self::Reconcile { repository_id })
            }
            Some("schedule") => Ok(Self::Reconcile { repository_id }),
            Some("pull_request_target") => {
                if ![
                    "opened",
                    "reopened",
                    "synchronize",
                    "edited",
                    "ready_for_review",
                ]
                .contains(&value["action"].as_str().unwrap_or(""))
                {
                    return Err("unsupported metadata lifecycle event".into());
                }
                let number = positive(&value, "/number")?;
                if number != positive(&value, "/pull_request/number")?
                    || positive(&value, "/pull_request/base/repo/id")? != repository_id
                    || value
                        .pointer("/pull_request/base/ref")
                        .and_then(Value::as_str)
                        != Some("main")
                {
                    return Err(
                        "publisher lifecycle event must identify this repository's main-target PR"
                            .into(),
                    );
                }
                Ok(Self::Lifecycle {
                    repository_id,
                    number,
                })
            }
            Some("workflow_run") if value["action"] == "completed" => Ok(Self::ReviewRun {
                repository_id,
                run_id: positive(&value, "/workflow_run/id")?,
            }),
            _ => Err(
                "publisher supports only protected lifecycle and authenticated review-relay events"
                    .into(),
            ),
        }
    }

    pub(super) fn resolve_all(
        self,
        client: &mut impl GithubRead,
        anchors: &activation::Anchors,
        head: &str,
    ) -> DynResult<Vec<u64>> {
        if let Self::Reconcile { repository_id } = self {
            if repository_id != anchors.repository_id {
                return Err("reconciliation repository ID differs from reviewed anchors".into());
            }
            return collector::open_pr_numbers(client, anchors, head);
        }
        let number = self.resolve(client, anchors)?;
        if collector::read_identity(client, number, anchors)?.main_sha != head {
            return Err("publisher event resolver is not exact current protected main".into());
        }
        Ok(vec![number])
    }

    pub(super) fn resolve(
        self,
        client: &mut impl GithubRead,
        anchors: &activation::Anchors,
    ) -> DynResult<u64> {
        match self {
            Self::Reconcile { .. } => {
                Err("batch reconciliation requires complete bounded PR listing".into())
            }
            Self::Lifecycle {
                repository_id,
                number,
            } => {
                if repository_id != anchors.repository_id {
                    return Err("publisher event numeric repository identity differs".into());
                }
                Ok(number)
            }
            Self::ReviewRun {
                repository_id,
                run_id,
            } => {
                if repository_id != anchors.repository_id {
                    return Err("review relay numeric repository identity differs".into());
                }
                resolve_run(client, anchors, run_id)
            }
        }
    }
}

fn positive(value: &Value, pointer: &str) -> DynResult<u64> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .filter(|id| *id > 0)
        .ok_or_else(|| "publisher event numeric identity is missing or malformed".into())
}

#[derive(Deserialize)]
struct Workflow {
    id: u64,
    path: String,
    state: String,
}
#[derive(Deserialize)]
struct Repository {
    id: u64,
}
#[derive(Deserialize)]
struct PullRef {
    sha: String,
    repo: Repository,
}
#[derive(Deserialize)]
struct PullRequest {
    number: u64,
    head: PullRef,
    base: PullRef,
}
#[derive(Deserialize)]
struct RunOwner {
    login: String,
}
#[derive(Deserialize)]
struct HeadRepository {
    id: u64,
    owner: RunOwner,
}
#[derive(Deserialize)]
struct WorkflowRun {
    id: u64,
    workflow_id: u64,
    path: String,
    event: String,
    status: String,
    repository: Repository,
    pull_requests: Option<Vec<PullRequest>>,
    head_branch: Option<String>,
    head_repository: Option<HeadRepository>,
}

fn resolve_run(
    client: &mut impl GithubRead,
    anchors: &activation::Anchors,
    run_id: u64,
) -> DynResult<u64> {
    let workflow: Workflow = app_auth::response(client.send(&Request::ReviewSignalWorkflow)?, 200)?;
    if workflow.id == 0 || workflow.path != RELAY_PATH || workflow.state != "active" {
        return Err("canonical review-relay workflow identity is unavailable".into());
    }
    let run: WorkflowRun = app_auth::response(client.send(&Request::WorkflowRun { run_id })?, 200)?;
    if run.id != run_id
        || run.workflow_id != workflow.id
        || !relay_path(&run.path)
        || run.event != "pull_request_review"
        || run.status != "completed"
        || run.repository.id != anchors.repository_id
    {
        return Err(
            "review-relay run does not bind the fixed workflow, repository and event".into(),
        );
    }
    let associations = run.pull_requests.as_deref().unwrap_or_default();
    if associations.is_empty() {
        return resolve_by_head(client, anchors, &run);
    }
    let [pr] = associations else {
        return Err("review-relay PR association is ambiguous".into());
    };
    if pr.number == 0
        || pr.base.repo.id != anchors.repository_id
        || !super::super::policy::valid_sha(&pr.head.sha)
    {
        return Err("review-relay PR association is incomplete".into());
    }
    let current = collector::read_identity(client, pr.number, anchors)?;
    if current.head_sha != pr.head.sha {
        return Err("review-relay association is stale for the current PR head".into());
    }
    Ok(pr.number)
}

fn relay_path(path: &str) -> bool {
    let (base, reference) = path
        .split_once('@')
        .map_or((path, None), |(base, reference)| (base, Some(reference)));
    (base == RELAY_PATH || base == format!("Mesh-LLM/mesh-llm/{RELAY_PATH}"))
        && reference.is_none_or(|reference| {
            !reference.is_empty()
                && reference.len() <= 1024
                && !reference.chars().any(char::is_control)
        })
}

#[derive(Deserialize)]
struct CandidateHead {
    sha: String,
    r#ref: String,
    repo: Option<HeadRepository>,
}
#[derive(Deserialize)]
struct CandidateBase {
    r#ref: String,
    repo: Repository,
}
#[derive(Deserialize)]
struct Candidate {
    number: u64,
    state: String,
    head: CandidateHead,
    base: CandidateBase,
}

fn resolve_by_head(
    client: &mut impl GithubRead,
    anchors: &activation::Anchors,
    run: &WorkflowRun,
) -> DynResult<u64> {
    let repository = run
        .head_repository
        .as_ref()
        .ok_or("review relay has no authenticated source repository")?;
    let branch = run
        .head_branch
        .as_ref()
        .ok_or("review relay has no authenticated source branch")?;
    if repository.id == 0 {
        return Err("review relay source repository ID is missing".into());
    }
    let mut matches = Vec::new();
    for page in 1..=100 {
        let candidates: Vec<Candidate> = app_auth::response(
            client.send(&Request::OpenPullRequestsByHead {
                owner: repository.owner.login.clone(),
                branch: branch.clone(),
                page,
            })?,
            200,
        )?;
        if candidates.len() > 100 {
            return Err("review-relay PR page exceeds supported bound".into());
        }
        let complete = candidates.len() < 100;
        for candidate in candidates {
            let head = candidate
                .head
                .repo
                .as_ref()
                .ok_or("review-relay PR candidate repository is incomplete")?;
            if candidate.number == 0
                || candidate.state != "open"
                || candidate.base.r#ref != "main"
                || candidate.base.repo.id != anchors.repository_id
                || head.id != repository.id
                || head.owner.login != repository.owner.login
                || candidate.head.r#ref != *branch
                || !super::super::policy::valid_sha(&candidate.head.sha)
            {
                return Err(
                    "review-relay PR candidate differs from authenticated repository and branch"
                        .into(),
                );
            }
            matches.push((candidate.number, candidate.head.sha));
        }
        if complete {
            let [(number, head)] = matches.as_slice() else {
                return Err("review-relay current PR resolution is missing or ambiguous".into());
            };
            if collector::read_identity(client, *number, anchors)?.head_sha != *head {
                return Err("review-relay current PR changed during resolution".into());
            }
            return Ok(*number);
        }
    }
    Err("review-relay PR pagination is incomplete; no result selected".into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn documented_run_paths_accept_only_the_fixed_relay() {
        for path in [
            super::RELAY_PATH,
            ".github/workflows/ci-governance-review-signal.yml@feature/review",
            "Mesh-LLM/mesh-llm/.github/workflows/ci-governance-review-signal.yml@refs/pull/1/merge",
        ] {
            assert!(super::relay_path(path));
        }
        for path in [
            ".github/workflows/other.yml@main",
            "Other/repo/.github/workflows/ci-governance-review-signal.yml@main",
            ".github/workflows/ci-governance-review-signal.yml@",
        ] {
            assert!(!super::relay_path(path));
        }
    }
}
