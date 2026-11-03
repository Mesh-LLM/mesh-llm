//! Versioned desired state and complete, non-secret operator normalization.
use super::{digest, require};
use crate::ci_governance::approval::{Account, AccountKind};
use crate::ci_governance::policy::{Policy, Scope, valid_sha};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExpectedState {
    pub schema_version: u32,
    pub repository: String,
    pub drift_target_seconds: u64,
    pub state: LiveState,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct LiveState {
    pub reviewer_team: ReviewerTeam,
    pub native_rule: NativeRule,
    pub administration_environment: AdministrationEnvironment,
    pub credentials: CredentialBoundaries,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ReviewerTeam {
    pub name: String,
    pub members: Vec<String>,
    pub visible: bool,
    pub repository_permission: RepositoryPermission,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum RepositoryPermission {
    None,
    Read,
    Triage,
    Write,
    Maintain,
    Admin,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeRule {
    pub branch: String,
    pub requires_pull_request: bool,
    pub reviewer_team: String,
    pub required_approvals: u32,
    pub dismiss_stale_reviews: bool,
    pub bypass_actors: Vec<String>,
    pub scope: Scope,
    pub product_only_requires_team_review: bool,
    pub required_check: String,
    pub required_check_source: CheckSource,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum CheckSource {
    SeparateGithubApp,
    GithubActions,
    Unrestricted,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AdministrationEnvironment {
    pub name: String,
    pub reviewers: Vec<String>,
    pub required_approvals: u32,
    pub prevent_self_review: bool,
    pub admin_bypass: bool,
    pub allowed_refs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CredentialBoundaries {
    pub observer_identity: String,
    pub observer_kind: CheckSource,
    pub observer_permissions: Vec<ObserverPermission>,
    pub observer_can_administer: bool,
    pub app_can_publish_checks: bool,
    pub administration_context: AdministrationContext,
    pub agents_have_administration: bool,
    pub pr_has_administration: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ObserverPermission {
    ActionsRead,
    AdministrationRead,
    ChecksRead,
    ChecksWrite,
    ContentsRead,
    EnvironmentsRead,
    MembersRead,
    MetadataRead,
    PullRequestsRead,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum AdministrationContext {
    ApprovedProtectedMainOnly,
    Unrestricted,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Observation {
    pub schema_version: u32,
    pub normalization: Normalization,
    pub repository: String,
    pub complete: bool,
    pub observed_at_unix: u64,
    pub protected_policy_revision: String,
    pub current_protected_main_revision: String,
    pub policy_sha256: String,
    pub readback_link: String,
    pub replay_ledger_complete: bool,
    pub replay_ledger_link: String,
    pub consumed_operation_ids: Vec<String>,
    pub observer: ObserverContext,
    pub execution: Option<super::record::ExecutionContext>,
    pub state: LiveState,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObserverContext {
    pub repository: String,
    pub run_id: u64,
    pub run_attempt: u64,
    pub ref_name: String,
    pub protected_revision: String,
    pub principal: Account,
}

impl ObserverContext {
    fn validate(&self, policy: &Policy, current_main: &str) -> DynResult<()> {
        let login = if self.principal.kind == AccountKind::Bot {
            self.principal
                .login
                .strip_suffix("[bot]")
                .unwrap_or(&self.principal.login)
        } else {
            &self.principal.login
        };
        require(
            self.repository == policy.repository
                && self.run_id > 0
                && self.run_attempt > 0
                && self.ref_name == "refs/heads/main"
                && self.protected_revision == current_main
                && matches!(self.principal.kind, AccountKind::Human | AccountKind::Bot)
                && self.principal.id > 0
                && !login.is_empty()
                && login.len() <= 39
                && login
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
            "read-only observer must bind a human/service principal and exact protected main run context",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Normalization {
    OperatorNormalizedV1,
}

impl ExpectedState {
    pub fn validate(&self, policy: &Policy) -> DynResult<()> {
        policy.validate()?;
        require(
            self.schema_version == 1
                && self.repository == policy.repository
                && self.drift_target_seconds == 900,
            "unsupported expected-state identity or proposed drift target",
        )?;
        self.state.normalized()?.validate_desired(policy)
    }
}

impl LiveState {
    fn validate_desired(&self, policy: &Policy) -> DynResult<()> {
        let team = &self.reviewer_team;
        require(
            team.name == policy.reviewer_team
                && team.visible
                && team.repository_permission == RepositoryPermission::Write
                && same_members(&team.members, &policy.eligible_humans),
            "desired reviewer team must be visible, exact roster, explicit repository write",
        )?;
        self.validate_rule(policy)?;
        let environment = &self.administration_environment;
        require(
            environment.name == "ci-governance-administration"
                && same_members(&environment.reviewers, &policy.eligible_humans)
                && environment.required_approvals == 1
                && environment.prevent_self_review
                && !environment.admin_bypass
                && environment.allowed_refs == ["refs/heads/main"],
            "administration environment must require independent review, no bypass, only main",
        )?;
        self.credentials.validate()
    }

    fn validate_rule(&self, policy: &Policy) -> DynResult<()> {
        let rule = &self.native_rule;
        require(
            rule.branch == policy.protected_branch
                && rule.requires_pull_request
                && rule.reviewer_team == policy.reviewer_team
                && rule.required_approvals == 1
                && rule.dismiss_stale_reviews
                && rule.bypass_actors.is_empty()
                && rule.scope == normalized_scope(&policy.scope)?
                && !rule.product_only_requires_team_review
                && rule.required_check == "CI policy / independent approval"
                && rule.required_check_source == CheckSource::SeparateGithubApp,
            "desired native rule must preserve CI-only scope, one review, stale dismissal and no bypass",
        )
    }

    pub fn normalized(&self) -> DynResult<Self> {
        let mut state = self.clone();
        sort_unique(&mut state.reviewer_team.members)?;
        sort_unique(&mut state.native_rule.bypass_actors)?;
        state.native_rule.scope = normalized_scope(&state.native_rule.scope)?;
        sort_unique(&mut state.administration_environment.reviewers)?;
        sort_unique(&mut state.administration_environment.allowed_refs)?;
        sort_unique(&mut state.credentials.observer_permissions)?;
        Ok(state)
    }

    pub fn equivalent(&self, other: &Self) -> DynResult<bool> {
        Ok(self.normalized()? == other.normalized()?)
    }

    pub fn canonical_digest(&self) -> DynResult<String> {
        digest(&self.normalized()?)
    }

    pub fn differences(&self, other: &Self) -> DynResult<Vec<&'static str>> {
        let expected = self.normalized()?;
        let actual = other.normalized()?;
        let mut result = Vec::new();
        if expected.reviewer_team != actual.reviewer_team {
            result.push("reviewer-team");
        }
        if expected.native_rule != actual.native_rule {
            result.push("ci-ruleset");
        }
        if expected.administration_environment != actual.administration_environment {
            result.push("administration-environment");
        }
        if expected.credentials != actual.credentials {
            result.push("credential-boundaries");
        }
        Ok(result)
    }
}

impl CredentialBoundaries {
    fn validate(&self) -> DynResult<()> {
        let permissions = [
            ObserverPermission::ActionsRead,
            ObserverPermission::MetadataRead,
            ObserverPermission::MembersRead,
            ObserverPermission::AdministrationRead,
            ObserverPermission::EnvironmentsRead,
            ObserverPermission::ChecksRead,
            ObserverPermission::ContentsRead,
            ObserverPermission::PullRequestsRead,
        ];
        require(
            self.observer_identity == "dedicated-governance-github-app"
                && self.observer_kind == CheckSource::SeparateGithubApp
                && self.observer_permissions.len() == permissions.len()
                && self
                    .observer_permissions
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    == permissions.into_iter().collect()
                && !self.observer_can_administer
                && self.app_can_publish_checks
                && self.administration_context == AdministrationContext::ApprovedProtectedMainOnly
                && !self.agents_have_administration
                && !self.pr_has_administration,
            "credentials must separate check-only App and approved main-only administration",
        )
    }
}

impl Observation {
    pub fn validate(&self, policy: &Policy, now: u64, max_age: u64) -> DynResult<()> {
        require(
            self.schema_version == 1
                && self.repository == policy.repository
                && self.complete
                && self.replay_ledger_complete
                && valid_sha(&self.protected_policy_revision)
                && self.current_protected_main_revision == self.protected_policy_revision
                && self.policy_sha256 == digest(policy)?,
            "incomplete observation or mismatched protected policy identity/digest",
        )?;
        require(
            self.observed_at_unix <= now && now - self.observed_at_unix <= max_age,
            "observation is stale or from the future",
        )?;
        audit_link(&self.readback_link)?;
        audit_link(&self.replay_ledger_link)?;
        unique_operation_ids(&self.consumed_operation_ids)?;
        self.state.normalized()?;
        self.observer
            .validate(policy, &self.current_protected_main_revision)?;
        if let Some(execution) = &self.execution {
            execution.validate(policy, &self.protected_policy_revision)?;
            require(
                execution.repository == self.observer.repository
                    && execution.run_id == self.observer.run_id
                    && execution.run_attempt == self.observer.run_attempt,
                "mutation and observer contexts must bind the same repository/run/attempt",
            )?;
        }
        Ok(())
    }
}

fn normalized_scope(scope: &Scope) -> DynResult<Scope> {
    let mut normalized = scope.clone();
    sort_unique(&mut normalized.directory_prefixes)?;
    sort_unique(&mut normalized.exact_paths)?;
    sort_unique(&mut normalized.root_filename_prefixes)?;
    Ok(normalized)
}

fn sort_unique<T: Ord>(values: &mut [T]) -> DynResult<()> {
    values.sort_unstable();
    require(
        !values.windows(2).any(|pair| pair[0] == pair[1]),
        "duplicate identity or setting in an unordered governance list",
    )
}

fn same_members(actual: &[String], expected: &[String]) -> bool {
    actual.len() == expected.len()
        && actual.iter().collect::<BTreeSet<_>>() == expected.iter().collect::<BTreeSet<_>>()
}

pub(super) fn audit_link(link: &str) -> DynResult<()> {
    let parsed = url::Url::parse(link)?;
    require(
        parsed.scheme() == "https"
            && parsed.host_str() == Some("github.com")
            && parsed.username().is_empty()
            && parsed.password().is_none()
            && parsed.path().starts_with("/Mesh-LLM/mesh-llm/"),
        "audit evidence must link to this repository on GitHub over HTTPS",
    )
}

pub(super) fn unique_operation_ids(ids: &[String]) -> DynResult<()> {
    for id in ids {
        let parsed = uuid::Uuid::parse_str(id)?;
        require(
            parsed.hyphenated().to_string() == *id,
            "operation IDs must use canonical lowercase hyphenated UUIDs",
        )?;
    }
    require(
        ids.iter().collect::<BTreeSet<_>>().len() == ids.len(),
        "duplicate operation ID in replay ledger",
    )
}
