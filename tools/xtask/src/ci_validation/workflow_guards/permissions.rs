use super::{Node, field};
use crate::command::DynResult;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    None,
    Read,
    Write,
}

impl Level {
    fn parse(text: &str) -> Result<Self, String> {
        match text.trim().trim_matches(['\'', '"']) {
            "none" => Ok(Self::None),
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            other => Err(format!("invalid permission level {other}")),
        }
    }
}

enum Grant {
    All(Level),
    Named(BTreeMap<String, Level>),
}

impl Grant {
    fn level(&self, name: &str) -> Level {
        match self {
            Self::All(level) => *level,
            Self::Named(levels) => levels.get(name).copied().unwrap_or(Level::None),
        }
    }
}

fn parse(node: &Node) -> Result<Grant, String> {
    match node {
        Node::Scalar(text) if text == "read-all" => Ok(Grant::All(Level::Read)),
        Node::Scalar(text) if text == "write-all" => Ok(Grant::All(Level::Write)),
        Node::Scalar(text) if text.starts_with('{') && text.ends_with('}') => {
            let mut levels = BTreeMap::new();
            for entry in text[1..text.len() - 1]
                .split(',')
                .filter(|entry| !entry.trim().is_empty())
            {
                let (name, level) = entry.split_once(':').ok_or("invalid permission mapping")?;
                levels.insert(name.trim().to_owned(), Level::parse(level)?);
            }
            Ok(Grant::Named(levels))
        }
        Node::Map(entries) => {
            let mut levels = BTreeMap::new();
            for (name, value) in entries {
                levels.insert(
                    name.clone(),
                    Level::parse(value.text().ok_or("permission level must be scalar")?)?,
                );
            }
            Ok(Grant::Named(levels))
        }
        Node::Scalar(_) | Node::Seq(_) => Err("invalid permissions declaration".into()),
    }
}

fn requested(document: &Node) -> Result<Option<BTreeMap<String, Level>>, String> {
    let mut requested: Option<BTreeMap<String, Level>> = None;
    let blocks = document.get("permissions").into_iter().chain(
        document
            .get("jobs")
            .into_iter()
            .flat_map(Node::entries)
            .filter_map(|(_, job)| job.get("permissions")),
    );
    for block in blocks {
        match parse(block)? {
            Grant::All(_) => return Ok(None),
            Grant::Named(levels) => {
                let output = requested.get_or_insert_with(BTreeMap::new);
                for (name, level) in levels {
                    output
                        .entry(name)
                        .and_modify(|old| *old = (*old).max(level))
                        .or_insert(level);
                }
            }
        }
    }
    Ok(requested)
}

fn satisfies(requested: &BTreeMap<String, Level>, grant: &Grant) -> bool {
    requested
        .iter()
        .all(|(name, level)| grant.level(name) >= *level)
}

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    for (name, document) in workflows {
        for (job_name, job) in document.get("jobs").into_iter().flat_map(Node::entries) {
            let Some(callee_name) =
                field(job, "uses").and_then(|uses| uses.strip_prefix("./.github/workflows/"))
            else {
                continue;
            };
            let callee = workflows
                .get(callee_name)
                .ok_or_else(|| format!("{name}:{job_name}: missing callee {callee_name}"))?;
            if callee
                .get("on")
                .and_then(|events| events.get("workflow_call"))
                .is_none()
            {
                continue;
            }
            let Some(needs) = requested(callee)? else {
                continue;
            };
            let Some(grant) = job
                .get("permissions")
                .or_else(|| document.get("permissions"))
            else {
                continue;
            };
            if !satisfies(&needs, &parse(grant)?) {
                return Err(format!(
                    "{name}:{job_name}: permission downgrade calling {callee_name}"
                )
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_validation::lane_results::workflow_yaml;

    fn document(text: &str) -> Node {
        workflow_yaml::parse(text).unwrap()
    }

    #[test]
    fn rejects_read_grant_when_write_requested() {
        let needs = requested(&document("permissions:\n  contents: write\n"))
            .unwrap()
            .unwrap();
        assert!(!satisfies(&needs, &Grant::All(Level::Read)));
        assert!(satisfies(&needs, &Grant::All(Level::Write)));
    }
    #[test]
    fn merges_strictest_job_request() {
        let needs = requested(&document("permissions:\n  contents: read\njobs:\n  build:\n    permissions:\n      contents: write\n      packages: read\n")).unwrap().unwrap();
        assert_eq!(needs["contents"], Level::Write);
        assert_eq!(needs["packages"], Level::Read);
    }
    #[test]
    fn all_scope_grants_cover_only_their_level() {
        let needs = BTreeMap::from([("packages".into(), Level::Read)]);
        assert!(satisfies(&needs, &Grant::All(Level::Read)));
        assert!(satisfies(&needs, &Grant::All(Level::Write)));
        assert!(!satisfies(&needs, &Grant::Named(BTreeMap::new())));
    }
    #[test]
    fn all_scope_request_is_not_enumerated() {
        assert!(
            requested(&document("permissions: read-all\n"))
                .unwrap()
                .is_none()
        );
        assert!(
            requested(&document("jobs:\n  build:\n    permissions: write-all\n"))
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn actual_edges_reject_downgrade_and_accept_equal_grant() {
        let callee = document("on:\n  workflow_call:\npermissions:\n  contents: write\n");
        let caller = document(
            "permissions:\n  contents: read\njobs:\n  invoke:\n    uses: ./.github/workflows/callee.yaml\n",
        );
        let mut workflows = BTreeMap::from([
            ("callee.yaml".into(), callee),
            ("caller.yml".into(), caller),
        ]);
        assert!(check(&workflows).is_err());
        workflows.insert("caller.yml".into(), document("permissions: write-all\njobs:\n  invoke:\n    uses: ./.github/workflows/callee.yaml\n"));
        assert!(check(&workflows).is_ok());
    }
}
