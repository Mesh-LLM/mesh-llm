//! Bounded reviewed requests and separately recorded manual completion evidence.
use super::state::{ExpectedState, LiveState, Observation, audit_link, unique_operation_ids};
use super::{digest, require};
use crate::ci_governance::approval::{Account, AccountKind};
use crate::ci_governance::policy::{Policy, valid_sha};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ChangeRecord {
    pub schema_version: u32,
    pub request: ChangeRequest,
    pub approval: Approval,
    pub manual_readback: Option<ManualReadback>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ChangeRequest {
    pub operation_id: String,
    pub repository: String,
    pub policy_revision: String,
    pub policy_sha256: String,
    pub expected_before_sha256: String,
    pub before: LiveState,
    pub desired_after: LiveState,
    pub controls: Vec<Control>,
    pub execution_method: ExecutionMethod,
    pub execution: ExecutionContext,
    pub owner: Account,
    pub author: Account,
    pub sponsor: Account,
    pub rationale: String,
    pub affected_jobs: Vec<String>,
    pub tests: Vec<String>,
    pub rollout: String,
    pub rollback: String,
    pub requested_at_unix: u64,
    pub expires_at_unix: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ExecutionContext {
    pub repository: String,
    pub run_id: u64,
    pub run_attempt: u64,
    pub ref_name: String,
    pub protected_revision: String,
    pub operator: Account,
    pub triggerer: Account,
    pub rerun_actor: Account,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ExecutionMethod {
    ManualProviderOperation,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Control {
    ReviewerTeam,
    CiRuleset,
    AdministrationEnvironment,
    CredentialBoundaries,
}

impl Control {
    fn name(self) -> &'static str {
        match self {
            Self::ReviewerTeam => "reviewer-team",
            Self::CiRuleset => "ci-ruleset",
            Self::AdministrationEnvironment => "administration-environment",
            Self::CredentialBoundaries => "credential-boundaries",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Approval {
    pub approver: Account,
    pub state: ApprovalState,
    pub request_sha256: String,
    pub policy_revision: String,
    pub execution: ExecutionContext,
    pub approved_at_unix: u64,
    pub expires_at_unix: u64,
    pub link: String,
    pub environment: String,
    pub environment_prevented_self_review: bool,
    pub environment_admin_bypass: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ApprovalState {
    Approved,
    Dismissed,
    Superseded,
    Pending,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ManualReadback {
    pub operation_id: String,
    pub request_sha256: String,
    pub execution: ExecutionContext,
    pub operated_at_unix: u64,
    pub observed_at_unix: u64,
    pub operator: Account,
    pub actual_state: LiveState,
    pub operation_link: String,
    pub readback_link: String,
    pub replay_ledger_link: String,
    pub consumed_operation_ids: Vec<String>,
    pub replay_ledger_complete: bool,
}

impl ChangeRecord {
    pub fn validate(
        &self,
        policy: &Policy,
        expected: &ExpectedState,
        observed: &Observation,
        now: u64,
    ) -> DynResult<()> {
        require(
            self.schema_version == 1,
            "unsupported change-record schema version",
        )?;
        self.request.validate(policy, expected, observed, now)?;
        self.approval.validate(policy, &self.request, now)?;
        require(
            observed.observed_at_unix > self.approval.approved_at_unix,
            "before-state observation must be collected after independent approval",
        )?;
        if let Some(receipt) = &self.manual_readback {
            receipt.validate(&self.request, &self.approval, observed, now)?;
        }
        Ok(())
    }
}

impl ChangeRequest {
    fn validate(
        &self,
        policy: &Policy,
        expected: &ExpectedState,
        observed: &Observation,
        now: u64,
    ) -> DynResult<()> {
        unique_operation_ids(std::slice::from_ref(&self.operation_id))?;
        let execution = observed
            .execution
            .as_ref()
            .ok_or("validate-change requires a human mutation execution context")?;
        require(
            self.repository == policy.repository
                && self.policy_revision == observed.protected_policy_revision
                && self.policy_sha256 == digest(policy)?
                && self.execution == *execution,
            "request must bind current protected policy revision/digest and exact observed run/attempt/ref/actors",
        )?;
        self.execution.validate(policy, &self.policy_revision)?;
        self.validate_before_after(expected, observed)?;
        self.validate_people()?;
        bounded_window(self.requested_at_unix, self.expires_at_unix, now)?;
        self.validate_explanation()
    }

    fn validate_before_after(
        &self,
        expected: &ExpectedState,
        observed: &Observation,
    ) -> DynResult<()> {
        require(
            self.before.equivalent(&observed.state)?
                && self.expected_before_sha256 == observed.state.canonical_digest()?
                && self.desired_after.equivalent(&expected.state)?,
            "stale before-state precondition or mismatched desired-after state",
        )?;
        require(
            !observed.consumed_operation_ids.contains(&self.operation_id),
            "operation ID already consumed: replay rejected",
        )?;
        let changed = self
            .before
            .differences(&self.desired_after)?
            .into_iter()
            .collect::<BTreeSet<_>>();
        let declared = self
            .controls
            .iter()
            .map(|control| control.name())
            .collect::<BTreeSet<_>>();
        require(
            !changed.is_empty() && changed == declared && declared.len() == self.controls.len(),
            "controls must enumerate exactly the changed supported governance settings",
        )
    }

    pub fn canonical_digest(&self) -> DynResult<String> {
        let mut normalized = self.clone();
        normalized.before = self.before.normalized()?;
        normalized.desired_after = self.desired_after.normalized()?;
        digest(&normalized)
    }

    fn validate_people(&self) -> DynResult<()> {
        for account in [&self.owner, &self.author, &self.sponsor] {
            human(account)?;
        }
        require(
            self.owner == self.sponsor && self.author == self.sponsor,
            "supported submission requires authenticated human author, owner and sponsor to be the same person",
        )
    }

    fn validate_explanation(&self) -> DynResult<()> {
        for text in [&self.rationale, &self.rollout, &self.rollback] {
            nonempty_text(text)?;
        }
        for entries in [&self.affected_jobs, &self.tests] {
            require(
                !entries.is_empty() && entries.len() <= 100,
                "affected jobs and tests must be explicitly recorded",
            )?;
            for entry in entries {
                nonempty_text(entry)?;
            }
        }
        Ok(())
    }
}

impl ExecutionContext {
    pub fn validate(&self, policy: &Policy, revision: &str) -> DynResult<()> {
        require(
            self.repository == policy.repository
                && self.run_id > 0
                && self.run_attempt > 0
                && self.ref_name == "refs/heads/main"
                && valid_sha(revision)
                && self.protected_revision == revision,
            "execution must bind this repository, exact current protected main revision, run and attempt",
        )?;
        for account in [&self.operator, &self.triggerer, &self.rerun_actor] {
            human(account)?;
        }
        Ok(())
    }
}

impl Approval {
    fn validate(&self, policy: &Policy, request: &ChangeRequest, now: u64) -> DynResult<()> {
        human(&self.approver)?;
        require(
            policy.eligible_humans.contains(&self.approver.login)
                && self.state == ApprovalState::Approved
                && self.request_sha256 == request.canonical_digest()?
                && self.policy_revision == request.policy_revision
                && self.execution == request.execution,
            "missing eligible current request-digest/policy/run-bound approval",
        )?;
        self.validate_independence(request)?;
        bounded_window(self.approved_at_unix, self.expires_at_unix, now)?;
        require(
            self.approved_at_unix > request.requested_at_unix
                && self.expires_at_unix <= request.expires_at_unix,
            "approval must follow request and expire within its bounded window",
        )?;
        require(
            self.environment == "ci-governance-administration"
                && self.environment_prevented_self_review
                && !self.environment_admin_bypass,
            "approval requires protected administration environment with self-review prevention and no admin bypass",
        )?;
        audit_link(&self.link)
    }

    fn validate_independence(&self, request: &ChangeRequest) -> DynResult<()> {
        for actor in [
            &request.author,
            &request.sponsor,
            &request.owner,
            &request.execution.operator,
            &request.execution.triggerer,
            &request.execution.rerun_actor,
        ] {
            require(
                !same_person(&self.approver, actor),
                "approver must differ from author, sponsor, owner, operator, triggerer and rerun actor",
            )?;
        }
        Ok(())
    }
}

impl ManualReadback {
    fn validate(
        &self,
        request: &ChangeRequest,
        approval: &Approval,
        before: &Observation,
        now: u64,
    ) -> DynResult<()> {
        require(
            self.operation_id == request.operation_id
                && self.request_sha256 == request.canonical_digest()?
                && self.execution == request.execution
                && self.operator == request.execution.operator,
            "manual receipt must bind the exact approved operation, request, run and operator",
        )?;
        require(
            self.operated_at_unix > approval.approved_at_unix
                && self.operated_at_unix > before.observed_at_unix
                && self.operated_at_unix <= approval.expires_at_unix
                && self.observed_at_unix > self.operated_at_unix
                && self.observed_at_unix <= now
                && now - self.observed_at_unix <= 900
                && self.actual_state.equivalent(&request.desired_after)?,
            "manual operation requires timely actual desired-after readback",
        )?;
        unique_operation_ids(&self.consumed_operation_ids)?;
        let mut expected_consumed = before.consumed_operation_ids.clone();
        expected_consumed.push(request.operation_id.clone());
        require(
            self.replay_ledger_complete
                && expected_consumed.iter().collect::<BTreeSet<_>>()
                    == self.consumed_operation_ids.iter().collect::<BTreeSet<_>>(),
            "manual receipt must record exactly one newly consumed operation ID",
        )?;
        for link in [
            &self.operation_link,
            &self.readback_link,
            &self.replay_ledger_link,
        ] {
            audit_link(link)?;
        }
        Ok(())
    }
}

fn bounded_window(start: u64, end: u64, now: u64) -> DynResult<()> {
    require(
        start > 0 && start <= now && end > now && end > start && end - start <= 3600,
        "request or approval expired, future-dated or outside the one-hour maximum window",
    )
}

fn human(account: &Account) -> DynResult<()> {
    require(
        account.kind == AccountKind::Human
            && account.id > 0
            && !account.login.is_empty()
            && account.login.len() <= 39
            && account
                .login
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "an authenticated human account identity is required",
    )
}

fn same_person(first: &Account, second: &Account) -> bool {
    first.id == second.id || first.login.eq_ignore_ascii_case(&second.login)
}

fn nonempty_text(value: &str) -> DynResult<()> {
    require(
        !value.trim().is_empty() && value.len() <= 4096 && !value.chars().any(|ch| ch.is_control()),
        "explanation text must be nonempty, bounded, and contain no control characters",
    )
}
