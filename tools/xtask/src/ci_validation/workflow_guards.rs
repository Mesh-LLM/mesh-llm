mod permissions;
mod shell;

use super::lane_results::workflow_yaml::{self, Node};
use crate::command::DynResult;
use std::collections::BTreeMap;
use std::path::Path;

pub(super) fn check(root: &Path) -> DynResult<()> {
    let mut workflows = BTreeMap::new();
    for entry in std::fs::read_dir(root.join(".github/workflows"))? {
        let path = entry?.path();
        if !matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("yml" | "yaml")
        ) {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("invalid workflow name")?
            .to_owned();
        let source = std::fs::read_to_string(&path)?;
        shell::check_expressions(&source).map_err(|error| format!("{name}: {error}"))?;
        let document = workflow_yaml::parse(&source).map_err(|error| format!("{name}: {error}"))?;
        shell::check_containers(&document).map_err(|error| format!("{name}: {error}"))?;
        workflows.insert(name, document);
    }
    permissions::check(&workflows)
}

fn field<'a>(node: &'a Node, name: &str) -> Option<&'a str> {
    node.get(name).and_then(Node::text)
}
