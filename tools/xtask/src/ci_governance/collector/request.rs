//! Closed API surface; callers cannot supply arbitrary endpoints, queries or bodies.
use crate::ci_governance::policy::valid_sha;
use crate::command::DynResult;
use serde_json::json;

pub(crate) const REPOSITORY: &str = "Mesh-LLM/mesh-llm";
pub(crate) const CHECK_NAME: &str = "CI policy / independent approval";
pub(crate) const PUBLISHER_PERMISSIONS: [(&str, &str); 2] =
    [("checks", "write"), ("metadata", "read")];
pub(crate) const OBSERVER_PERMISSIONS: [(&str, &str); 8] = [
    ("actions", "read"),
    ("administration", "read"),
    ("checks", "read"),
    ("contents", "read"),
    ("environments", "read"),
    ("members", "read"),
    ("metadata", "read"),
    ("pull_requests", "read"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CheckConclusion {
    Success,
    Failure,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Request {
    Repository,
    CurrentMain,
    /// The legacy branch-protection endpoint; a denial is not absence.
    MainBranchProtection,
    /// One bounded page only. It is not a complete effective-ruleset census.
    RepositoryRulesetsFirstPage,
    Pull {
        number: u64,
    },
    Policy {
        revision: String,
    },
    Files {
        number: u64,
        page: u32,
    },
    Reviews {
        number: u64,
        cursor: Option<String>,
    },
    Team,
    TeamMembers {
        cursor: Option<String>,
    },
    TeamPermission,
    AdministrationEnvironment,
    AdministrationEnvironmentBranches {
        page: u32,
    },
    WorkflowRun {
        run_id: u64,
    },
    WorkflowRunApprovals {
        run_id: u64,
    },
    ReviewSignalWorkflow,
    OpenPullRequestsByHead {
        owner: String,
        branch: String,
        page: u32,
    },
    OpenPullRequests {
        page: u32,
    },
    App,
    AppInstallation {
        id: u64,
    },
    MintPublisherToken {
        installation_id: u64,
        repository_id: u64,
    },
    MintObserverToken {
        installation_id: u64,
        repository_id: u64,
    },
    CreateGovernanceCheck {
        head_sha: String,
        external_id: String,
    },
    GetGovernanceCheck {
        id: u64,
    },
    CompleteGovernanceCheck {
        id: u64,
        conclusion: CheckConclusion,
    },
}

pub(super) struct Encoded {
    pub method: &'static str,
    pub endpoint: String,
    pub body: Option<Vec<u8>>,
    pub accept: &'static str,
}

impl Request {
    pub(super) fn encode(&self) -> DynResult<Encoded> {
        let repo = format!("repos/{REPOSITORY}");
        let environment = format!("{repo}/environments/ci-governance-administration");
        let (method, endpoint, body) = match self {
            Self::Repository => ("GET", repo, None),
            Self::CurrentMain => ("GET", format!("{repo}/branches/main"), None),
            Self::MainBranchProtection => ("GET", format!("{repo}/branches/main/protection"), None),
            Self::RepositoryRulesetsFirstPage => {
                ("GET", format!("{repo}/rulesets?per_page=100&page=1"), None)
            }
            Self::Pull { number } => ("GET", format!("{repo}/pulls/{}", positive(*number)?), None),
            Self::Policy { revision } => {
                if !valid_sha(revision) {
                    return Err("invalid protected policy revision".into());
                }
                (
                    "GET",
                    format!("{repo}/contents/ci/governance/policy.json?ref={revision}"),
                    None,
                )
            }
            Self::Files { number, page } => (
                "GET",
                format!(
                    "{repo}/pulls/{}/files?per_page=100&page={}",
                    positive(*number)?,
                    page_number(*page)?
                ),
                None,
            ),
            Self::Reviews { number, cursor } => {
                positive(*number)?;
                validate_cursor(cursor)?;
                (
                    "POST",
                    "graphql".into(),
                    Some(
                        json!({"query":REVIEWS_QUERY,"variables":{"number":number,"cursor":cursor}}),
                    ),
                )
            }
            Self::Team => (
                "GET",
                "orgs/Mesh-LLM/teams/ci-policy-reviewers".into(),
                None,
            ),
            Self::TeamMembers { cursor } => {
                validate_cursor(cursor)?;
                (
                    "POST",
                    "graphql".into(),
                    Some(json!({"query":MEMBERS_QUERY,"variables":{"cursor":cursor}})),
                )
            }
            Self::TeamPermission => (
                "GET",
                format!("orgs/Mesh-LLM/teams/ci-policy-reviewers/repos/{REPOSITORY}"),
                None,
            ),
            Self::AdministrationEnvironment => ("GET", environment, None),
            Self::AdministrationEnvironmentBranches { page } => (
                "GET",
                format!(
                    "{environment}/deployment-branch-policies?per_page=100&page={}",
                    page_number(*page)?
                ),
                None,
            ),
            Self::WorkflowRun { run_id } => (
                "GET",
                format!("{repo}/actions/runs/{}", positive(*run_id)?),
                None,
            ),
            Self::WorkflowRunApprovals { run_id } => (
                "GET",
                format!("{repo}/actions/runs/{}/approvals", positive(*run_id)?),
                None,
            ),
            Self::ReviewSignalWorkflow => (
                "GET",
                format!("{repo}/actions/workflows/ci-governance-review-signal.yml"),
                None,
            ),
            Self::OpenPullRequestsByHead {
                owner,
                branch,
                page,
            } => ("GET", open_head_endpoint(owner, branch, *page)?, None),
            Self::OpenPullRequests { page } => (
                "GET",
                format!(
                    "{repo}/pulls?state=open&base=main&per_page=100&page={}",
                    page_number(*page)?
                ),
                None,
            ),
            Self::App => ("GET", "app".into(), None),
            Self::AppInstallation { id } => {
                ("GET", format!("app/installations/{}", positive(*id)?), None)
            }
            Self::MintPublisherToken {
                installation_id,
                repository_id,
            } => {
                positive(*repository_id)?;
                (
                    "POST",
                    format!(
                        "app/installations/{}/access_tokens",
                        positive(*installation_id)?
                    ),
                    Some(
                        json!({"repository_ids":[repository_id],"permissions":{"checks":"write","metadata":"read"}}),
                    ),
                )
            }
            Self::MintObserverToken {
                installation_id,
                repository_id,
            } => {
                positive(*repository_id)?;
                (
                    "POST",
                    format!(
                        "app/installations/{}/access_tokens",
                        positive(*installation_id)?
                    ),
                    Some(
                        json!({"repository_ids":[repository_id],"permissions":OBSERVER_PERMISSIONS.into_iter().collect::<std::collections::BTreeMap<_,_>>()}),
                    ),
                )
            }
            Self::CreateGovernanceCheck {
                head_sha,
                external_id,
            } => {
                if !valid_sha(head_sha)
                    || external_id.is_empty()
                    || external_id.len() > 255
                    || !external_id
                        .bytes()
                        .all(|ch| ch.is_ascii_alphanumeric() || b"-_:/.".contains(&ch))
                {
                    return Err("invalid fixed governance check identity".into());
                }
                (
                    "POST",
                    format!("{repo}/check-runs"),
                    Some(
                        json!({"name":CHECK_NAME,"head_sha":head_sha,"external_id":external_id,"status":"in_progress"}),
                    ),
                )
            }
            Self::GetGovernanceCheck { id } => {
                ("GET", format!("{repo}/check-runs/{}", positive(*id)?), None)
            }
            Self::CompleteGovernanceCheck { id, conclusion } => {
                let (result, title, summary) = match conclusion {
                    CheckConclusion::Success => (
                        "success",
                        "CI policy evaluation passed",
                        "Protected-source policy evaluation passed for this exact PR revision.",
                    ),
                    CheckConclusion::Failure => (
                        "failure",
                        "CI policy evaluation denied",
                        "Protected-source policy evaluation did not establish the required independent current-revision approval. Recollect current evidence before retrying.",
                    ),
                };
                (
                    "PATCH",
                    format!("{repo}/check-runs/{}", positive(*id)?),
                    Some(
                        json!({"status":"completed","conclusion":result,"output":{"title":title,"summary":summary}}),
                    ),
                )
            }
        };
        Ok(Encoded {
            method,
            endpoint,
            body: body.map(|body| serde_json::to_vec(&body)).transpose()?,
            accept: if matches!(self, Self::Policy { .. }) {
                "application/vnd.github.raw+json"
            } else {
                "application/vnd.github+json"
            },
        })
    }
}

fn open_head_endpoint(owner: &str, branch: &str, page: u32) -> DynResult<String> {
    if owner.is_empty()
        || owner.len() > 100
        || !owner
            .bytes()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == b'-')
        || branch.is_empty()
        || branch.len() > 1024
        || branch.chars().any(char::is_control)
    {
        return Err("invalid bounded PR head owner/branch selector".into());
    }
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("state", "open")
        .append_pair("base", "main")
        .append_pair("head", &format!("{owner}:{branch}"))
        .append_pair("per_page", "100")
        .append_pair("page", &page_number(page)?.to_string())
        .finish();
    Ok(format!("repos/{REPOSITORY}/pulls?{query}"))
}

fn positive(id: u64) -> DynResult<u64> {
    if id == 0 {
        Err("GitHub identity must be positive".into())
    } else {
        Ok(id)
    }
}
fn page_number(page: u32) -> DynResult<u32> {
    if !(1..=100).contains(&page) {
        Err("GitHub page exceeds supported bound".into())
    } else {
        Ok(page)
    }
}
fn validate_cursor(cursor: &Option<String>) -> DynResult<()> {
    if cursor.as_ref().is_some_and(|cursor| {
        cursor.is_empty() || cursor.len() > 1024 || cursor.chars().any(char::is_control)
    }) {
        return Err("invalid GitHub pagination cursor".into());
    }
    Ok(())
}

const REVIEWS_QUERY: &str = r#"query($number:Int!,$cursor:String){repository(owner:"Mesh-LLM",name:"mesh-llm"){pullRequest(number:$number){reviews(first:100,after:$cursor){totalCount nodes{fullDatabaseId state submittedAt commit{oid} author{__typename login ... on User{databaseId} ... on Bot{databaseId}}} pageInfo{hasNextPage endCursor}}}}}"#;
const MEMBERS_QUERY: &str = r#"query($cursor:String){organization(login:"Mesh-LLM"){team(slug:"ci-policy-reviewers"){databaseId members(first:100,after:$cursor,membership:ALL){totalCount nodes{__typename login databaseId} pageInfo{hasNextPage endCursor}}}}}"#;

#[cfg(test)]
mod inspection_request_tests {
    use super::*;

    #[test]
    fn inspection_endpoints_are_fixed_read_only_requests() {
        for (request, endpoint) in [
            (Request::Repository, "repos/Mesh-LLM/mesh-llm"),
            (
                Request::CurrentMain,
                "repos/Mesh-LLM/mesh-llm/branches/main",
            ),
            (Request::Team, "orgs/Mesh-LLM/teams/ci-policy-reviewers"),
            (
                Request::TeamPermission,
                "orgs/Mesh-LLM/teams/ci-policy-reviewers/repos/Mesh-LLM/mesh-llm",
            ),
            (
                Request::AdministrationEnvironment,
                "repos/Mesh-LLM/mesh-llm/environments/ci-governance-administration",
            ),
            (
                Request::MainBranchProtection,
                "repos/Mesh-LLM/mesh-llm/branches/main/protection",
            ),
            (
                Request::RepositoryRulesetsFirstPage,
                "repos/Mesh-LLM/mesh-llm/rulesets?per_page=100&page=1",
            ),
        ] {
            let encoded = request.encode().unwrap();
            assert_eq!(encoded.method, "GET");
            assert_eq!(encoded.endpoint, endpoint);
            assert!(encoded.body.is_none());
        }
    }
}
