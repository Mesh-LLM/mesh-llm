//! Fixed GitHub environment/configuration reads; configuration is not authorization.
use super::require;
use crate::ci_governance::{
    activation::Anchors,
    approval::{Account, AccountKind},
    collector::{GithubRead, Request},
    policy::Policy,
};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const ENVIRONMENT: &str = "ci-governance-administration";

#[derive(Clone, Debug, Serialize)]
pub(super) struct EnvironmentReadback {
    pub environment_id: u64,
    pub name: String,
    pub reviewers: Vec<Account>,
    pub prevent_self_review: bool,
    pub administrator_bypass: Option<bool>,
    pub allowed_branch: String,
    pub allowed_branch_kind: Option<String>,
    pub unverified_fields: Vec<&'static str>,
    pub configuration_verified: bool,
    pub approval_verified: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct ApprovalReadback {
    pub reviews_observed: usize,
    pub exact_attempt_binding_verified: bool,
    pub prepared_packet_digest_binding_verified: bool,
    pub authorization_verified: bool,
    pub blocking_reason: &'static str,
}

#[derive(Deserialize)]
struct EnvironmentWire {
    id: u64,
    name: String,
    can_admins_bypass: Option<bool>,
    protection_rules: Vec<ProtectionRule>,
    deployment_branch_policy: BranchSelection,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ProtectionRule {
    #[serde(rename = "required_reviewers")]
    RequiredReviewers {
        id: u64,
        prevent_self_review: bool,
        reviewers: Vec<Reviewer>,
    },
    #[serde(rename = "branch_policy")]
    BranchPolicy { id: u64 },
}

#[derive(Deserialize)]
struct Reviewer {
    #[serde(rename = "type")]
    kind: String,
    reviewer: AccountWire,
}

#[derive(Deserialize)]
struct BranchSelection {
    protected_branches: bool,
    custom_branch_policies: bool,
}

#[derive(Deserialize)]
struct BranchPage {
    total_count: usize,
    branch_policies: Vec<BranchPolicy>,
}

#[derive(Deserialize)]
struct BranchPolicy {
    id: u64,
    name: String,
    #[serde(rename = "type")]
    kind: Option<String>,
}

#[derive(Clone, Deserialize)]
pub(super) struct AccountWire {
    pub login: String,
    pub id: u64,
    #[serde(rename = "type")]
    pub kind: String,
}

impl AccountWire {
    pub fn human(self) -> DynResult<Account> {
        require(
            self.kind == "User"
                && self.id > 0
                && !self.login.is_empty()
                && self.login.len() <= 39
                && self
                    .login
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
            "GitHub metadata requires an authenticated human User identity",
        )?;
        Ok(Account {
            login: self.login,
            id: self.id,
            kind: AccountKind::Human,
        })
    }
}

pub(super) fn collect_environment(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    policy: &Policy,
) -> DynResult<EnvironmentReadback> {
    let wire: EnvironmentWire = get(client, Request::AdministrationEnvironment)?;
    require(
        wire.id == anchors.administration_environment_id
            && wire.name == ENVIRONMENT
            && wire.can_admins_bypass != Some(true)
            && !wire.deployment_branch_policy.protected_branches
            && wire.deployment_branch_policy.custom_branch_policies,
        "administration environment identity/bypass/main-only configuration mismatch",
    )?;
    let reviewers = reviewers(wire.protection_rules, policy, anchors)?;
    let allowed_branch_kind = branches(client)?;
    // The public environment GET contract does not expose administrator-bypass
    // state. An undocumented/synthetic false field cannot establish protection.
    let mut unverified_fields = vec!["administrator_bypass"];
    if allowed_branch_kind.is_none() {
        unverified_fields.push("allowed_branch_kind");
    }
    Ok(EnvironmentReadback {
        environment_id: wire.id,
        name: wire.name,
        reviewers,
        prevent_self_review: true,
        administrator_bypass: None,
        allowed_branch: "main".into(),
        allowed_branch_kind,
        unverified_fields,
        configuration_verified: false,
        approval_verified: false,
    })
}

fn reviewers(
    rules: Vec<ProtectionRule>,
    policy: &Policy,
    anchors: &Anchors,
) -> DynResult<Vec<Account>> {
    let mut reviewer_rule = None;
    let mut ids = BTreeSet::new();
    let mut branch_rule = false;
    for rule in rules {
        match rule {
            ProtectionRule::RequiredReviewers {
                id,
                prevent_self_review,
                reviewers,
            } => {
                require(
                    id > 0 && ids.insert(id) && prevent_self_review && reviewer_rule.is_none(),
                    "missing/duplicate reviewer rule or disabled environment self-review prevention",
                )?;
                reviewer_rule = Some(reviewers);
            }
            ProtectionRule::BranchPolicy { id } => {
                require(
                    id > 0 && ids.insert(id) && !branch_rule,
                    "duplicate environment branch protection rule",
                )?;
                branch_rule = true;
            }
        }
    }
    require(
        branch_rule,
        "environment branch-policy protection rule missing",
    )?;
    let mut accounts = Vec::new();
    for entry in reviewer_rule.ok_or("environment required-reviewer rule missing")? {
        require(
            entry.kind == "User",
            "team/custom environment reviewers are unsupported",
        )?;
        let account = entry.reviewer.human()?;
        require(
            anchors.human_ids.get(&account.login) == Some(&account.id),
            "environment reviewer numeric identity differs from reviewed human anchors",
        )?;
        accounts.push(account);
    }
    let logins = accounts
        .iter()
        .map(|account| account.login.as_str())
        .collect::<BTreeSet<_>>();
    let ids = accounts
        .iter()
        .map(|account| account.id)
        .collect::<BTreeSet<_>>();
    require(
        accounts.len() == policy.eligible_humans.len()
            && ids.len() == accounts.len()
            && logins == policy.eligible_humans.iter().map(String::as_str).collect(),
        "environment reviewer set must be exactly the eligible independent humans",
    )?;
    accounts.sort_by(|first, second| first.login.cmp(&second.login));
    Ok(accounts)
}

fn branches(client: &mut impl GithubRead) -> DynResult<Option<String>> {
    let mut expected_count = None;
    let mut policies = Vec::new();
    let mut ids = BTreeSet::new();
    for page in 1..=10 {
        let response: BranchPage =
            get(client, Request::AdministrationEnvironmentBranches { page })?;
        require(
            response.total_count <= 1000 && response.branch_policies.len() <= 100,
            "environment branch policy pagination exceeds bounds",
        )?;
        let total = *expected_count.get_or_insert(response.total_count);
        require(
            total == response.total_count,
            "environment branch policies changed during pagination",
        )?;
        for policy in response.branch_policies {
            require(
                policy.id > 0 && ids.insert(policy.id),
                "duplicate environment branch policy",
            )?;
            policies.push(policy);
        }
        require(
            policies.len() <= total,
            "environment branch policy count mismatch",
        )?;
        if policies.len() == total {
            require(
                policies.len() == 1
                    && policies[0].name == "main"
                    && policies[0]
                        .kind
                        .as_deref()
                        .is_none_or(|kind| kind == "branch"),
                "administration environment must allow exactly main, without tags or wildcard branches",
            )?;
            return Ok(policies.remove(0).kind);
        }
    }
    Err("incomplete environment branch-policy pagination".into())
}

#[derive(Deserialize)]
struct ApprovalWire {
    state: ReviewState,
    user: AccountWire,
    environments: Vec<EnvironmentReference>,
}
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ReviewState {
    Approved,
    Rejected,
    Pending,
}
#[derive(Deserialize)]
struct EnvironmentReference {
    id: u64,
    name: String,
}

pub(super) fn collect_approval_history(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    run_id: u64,
) -> DynResult<ApprovalReadback> {
    let history: Vec<ApprovalWire> = get(client, Request::WorkflowRunApprovals { run_id })?;
    require(
        history.len() <= 1000,
        "environment review history exceeds supported bounds",
    )?;
    for review in &history {
        let _state = &review.state;
        let user = review.user.clone().human()?;
        require(
            anchors.human_ids.get(&user.login) == Some(&user.id),
            "environment history reviewer is unlisted or differs from reviewed human identity anchors",
        )?;
        require(
            !review.environments.is_empty()
                && review.environments.iter().all(|environment| {
                    environment.id == anchors.administration_environment_id
                        && environment.name == ENVIRONMENT
                }),
            "environment approval history targets another/unverified environment",
        )?;
    }
    Ok(ApprovalReadback {
        reviews_observed: history.len(),
        exact_attempt_binding_verified: false,
        prepared_packet_digest_binding_verified: false,
        authorization_verified: false,
        blocking_reason: "GitHub review history alone does not bind this exact run attempt, prepared packet digest and approval time; protected authorization evidence remains required.",
    })
}

pub(super) fn get<T: serde::de::DeserializeOwned>(
    client: &mut impl GithubRead,
    request: Request,
) -> DynResult<T> {
    let response = client.send(&request)?;
    require(
        response.status == 200 && response.body.len() <= 1024 * 1024,
        "fixed GitHub metadata read failed, is unsupported, or exceeds bounds",
    )?;
    serde_json::from_slice(&response.body)
        .map_err(|_| "unsupported/incomplete GitHub governance metadata shape".into())
}

#[cfg(test)]
#[path = "environment_readback_tests.rs"]
mod tests;
