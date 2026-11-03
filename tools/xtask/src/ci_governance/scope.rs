//! Scope uses literal paths and bounded prefixes, never fnmatch.
use super::policy::Policy;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ChangeKind {
    Added,
    Modified,
    Deleted,
    Renamed,
    Copied,
    TypeChanged,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChangedPath {
    pub kind: ChangeKind,
    pub path: String,
    pub previous_path: Option<String>,
}

pub(crate) fn validate_path(path: &str) -> DynResult<()> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains(['\0', '\\'])
        || path.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err("invalid repository-relative path".into());
    }
    Ok(())
}

pub(crate) fn is_policy_path(policy: &Policy, path: &str) -> bool {
    policy.scope.covers_path(path)
}

pub(crate) fn policy_changes(policy: &Policy, changes: &[ChangedPath]) -> DynResult<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut controlled = BTreeSet::new();
    for change in changes {
        validate_path(&change.path)?;
        if !seen.insert(&change.path) {
            return Err("duplicate changed path in complete snapshot".into());
        }
        if matches!(change.kind, ChangeKind::Renamed | ChangeKind::Copied)
            != change.previous_path.is_some()
        {
            return Err("rename/copy requires previous_path; other changes forbid it".into());
        }
        for path in std::iter::once(&change.path).chain(change.previous_path.iter()) {
            validate_path(path)?;
            if is_policy_path(policy, path) {
                controlled.insert(path.clone());
            }
        }
    }
    Ok(controlled.into_iter().collect())
}
