//! Precommittable bounded intent; runtime revision and run identities are added later.
use super::record::{Control, ExecutionMethod};
use super::state::{ExpectedState, LiveState, unique_operation_ids};
use super::{digest, require};
use crate::ci_governance::approval::{Account, AccountKind};
use crate::ci_governance::policy::Policy;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReviewedIntent {
    pub schema_version: u32,
    pub operation_id: String,
    pub repository: String,
    pub policy_sha256: String,
    pub desired_state_sha256: String,
    pub expected_before_sha256: String,
    pub before: LiveState,
    pub desired_after: LiveState,
    pub controls: Vec<Control>,
    pub execution_method: ExecutionMethod,
    pub owner: Account,
    pub author: Account,
    pub sponsor: Account,
    pub rationale: String,
    pub affected_jobs: Vec<String>,
    pub tests: Vec<String>,
    pub rollout: String,
    pub rollback: String,
    pub created_at_unix: u64,
    pub expires_at_unix: u64,
}

impl ReviewedIntent {
    pub fn validate(&self, policy: &Policy, expected: &ExpectedState, now: u64) -> DynResult<()> {
        expected.validate(policy)?;
        unique_operation_ids(std::slice::from_ref(&self.operation_id))?;
        require(
            self.schema_version == 1
                && self.repository == policy.repository
                && self.policy_sha256 == digest(policy)?
                && self.desired_state_sha256 == expected.state.canonical_digest()?
                && self.expected_before_sha256 == self.before.canonical_digest()?
                && self.desired_after.equivalent(&expected.state)?,
            "intent must bind current protected policy, exact before and final desired state",
        )?;
        require(
            self.created_at_unix > 0
                && self.created_at_unix <= now
                && self.expires_at_unix > now
                && self.expires_at_unix > self.created_at_unix
                && self.expires_at_unix - self.created_at_unix <= 7 * 24 * 3600,
            "static intent expired, future-dated or exceeds seven-day review window",
        )?;
        self.validate_people()?;
        self.validate_controls()?;
        self.validate_explanation()
    }

    fn validate_people(&self) -> DynResult<()> {
        require(
            self.author == self.owner
                && self.author == self.sponsor
                && self.author.kind == AccountKind::Human
                && self.author.id > 0
                && !self.author.login.is_empty()
                && self.author.login.len() <= 39
                && self
                    .author
                    .login
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
            "intent requires one responsible human author, owner and sponsor",
        )
    }

    fn validate_controls(&self) -> DynResult<()> {
        let changed = self
            .before
            .differences(&self.desired_after)?
            .into_iter()
            .map(|name| serde_json::from_value::<Control>(name.into()))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let declared = self.controls.iter().copied().collect::<BTreeSet<_>>();
        require(
            !changed.is_empty() && changed == declared && declared.len() == self.controls.len(),
            "intent controls must name exactly the changed bounded governance settings",
        )
    }

    fn validate_explanation(&self) -> DynResult<()> {
        for value in [&self.rationale, &self.rollout, &self.rollback] {
            text(value)?;
        }
        for list in [&self.affected_jobs, &self.tests] {
            require(
                !list.is_empty() && list.len() <= 100,
                "intent jobs/tests must be explicit and bounded",
            )?;
            for value in list {
                text(value)?;
            }
        }
        Ok(())
    }

    pub fn canonical_digest(&self) -> DynResult<String> {
        let mut normalized = self.clone();
        normalized.before = self.before.normalized()?;
        normalized.desired_after = self.desired_after.normalized()?;
        digest(&normalized)
    }
}

fn text(value: &str) -> DynResult<()> {
    require(
        !value.trim().is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control),
        "intent explanation must be nonempty bounded text",
    )
}

#[cfg(test)]
#[path = "intent_tests.rs"]
mod tests;
