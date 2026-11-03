//! Protected-base policy data. Loading a file does not authenticate its source.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub(crate) const REVIEWERS: [&str; 3] = ["micspiral", "ndizazzo", "i386"];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Policy {
    pub schema_version: u32,
    pub repository: String,
    pub protected_branch: String,
    pub reviewer_team: String,
    pub eligible_humans: Vec<String>,
    pub required_approvals: u32,
    pub submission: String,
    pub scope: Scope,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub directory_prefixes: Vec<String>,
    pub exact_paths: Vec<String>,
    pub root_filename_prefixes: Vec<String>,
}

impl Policy {
    pub(crate) fn from_bytes(bytes: &[u8]) -> DynResult<Self> {
        let policy: Self = serde_json::from_slice(bytes)?;
        policy.validate()?;
        Ok(policy)
    }

    pub(crate) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.repository != "Mesh-LLM/mesh-llm"
            || self.protected_branch != "main"
            || self.reviewer_team != "Mesh-LLM/ci-policy-reviewers"
            || self.required_approvals != 1
            || self.submission != "human-author-is-sponsor"
        {
            return Err("unsupported CI-governance policy identity or approval contract".into());
        }
        let actual: BTreeSet<_> = self.eligible_humans.iter().map(String::as_str).collect();
        if actual != REVIEWERS.into_iter().collect() || self.eligible_humans.len() != 3 {
            return Err("eligible humans must be exactly micspiral, ndizazzo, and i386".into());
        }
        self.scope.validate()
    }
}

impl Scope {
    fn validate(&self) -> DynResult<()> {
        validate_unique(&self.exact_paths)?;
        validate_unique(&self.directory_prefixes)?;
        validate_unique(&self.root_filename_prefixes)?;
        for path in &self.exact_paths {
            validate_literal(path)?;
        }
        for prefix in &self.directory_prefixes {
            let Some(path) = prefix.strip_suffix('/') else {
                return Err("scope directory prefixes must end with /".into());
            };
            validate_literal(path)?;
        }
        for prefix in &self.root_filename_prefixes {
            validate_literal(prefix)?;
            if prefix.contains('/') {
                return Err("root filename prefix must contain no /".into());
            }
        }
        // The validator, policy and ownership controls cannot exempt themselves.
        for control in [".github/", "ci/", "tools/", ".agents/skills/manage-ci/"] {
            if !self.directory_prefixes.iter().any(|path| path == control) {
                return Err(format!("policy omits mandatory control prefix {control}").into());
            }
        }
        Ok(())
    }
}

fn validate_unique(values: &[String]) -> DynResult<()> {
    if values.iter().collect::<BTreeSet<_>>().len() != values.len() {
        return Err("duplicate scope entry".into());
    }
    Ok(())
}

fn validate_literal(path: &str) -> DynResult<()> {
    super::scope::validate_path(path)?;
    if path
        .chars()
        .any(|ch| ch.is_whitespace() || "*?![]\\#".contains(ch))
    {
        return Err("scope entries must be literal paths supported by CODEOWNERS".into());
    }
    Ok(())
}

pub(crate) fn valid_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
}

#[cfg(test)]
pub(crate) fn fixture() -> Policy {
    serde_json::from_str(include_str!("../../../../ci/governance/policy.json")).unwrap()
}
