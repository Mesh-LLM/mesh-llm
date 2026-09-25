use super::scan::Candidate;
use super::shard_rows::{PythonDisposition, ScriptShard};
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub(super) fn validate(
    root: Option<&Path>,
    shard: &ScriptShard,
    observed: &[Candidate],
) -> DynResult<()> {
    if shard.schema_version != 1 {
        return Err("script-edges.json: unsupported shard version".into());
    }
    let candidates = observed
        .iter()
        .filter(|row| {
            row.path.starts_with("scripts/")
                && row.path.ends_with(".py")
                && !row.path.starts_with("scripts/tests/")
        })
        .map(|row| (row.id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut recorded = BTreeSet::new();
    let mut files = BTreeSet::new();
    for group in &shard.python_implementation_groups {
        if !group.file.starts_with("scripts/")
            || !group.file.ends_with(".py")
            || group.file.starts_with("scripts/tests/")
            || group.file.split('/').any(|part| part == "..")
            || group.root.trim().is_empty()
            || group.boundary.trim().is_empty()
            || !files.insert(group.file.as_str())
        {
            return Err(format!(
                "script-edges.json: invalid Python source group {}",
                group.file
            )
            .into());
        }
        let source = root
            .map(|root| fs::read_to_string(root.join(&group.file)))
            .transpose()?;
        for (line, hash, occurrence, kind, disposition, target, contract) in &group.members {
            if *line == 0
                || *occurrence == 0
                || hash.len() != 16
                || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                || target.trim().is_empty()
                || contract.trim().is_empty()
            {
                return Err(format!(
                    "script-edges.json: incomplete Python member in {}",
                    group.file
                )
                .into());
            }
            let id = format!("{}#{}:{hash}:{occurrence}", group.file, kind.label());
            if !recorded.insert(id.clone()) {
                return Err(format!("script-edges.json: duplicate Python member {id}").into());
            }
            let actual = candidates
                .get(id.as_str())
                .ok_or_else(|| format!("script-edges.json: stale source identity {id}"))?;
            let source_line = source.as_ref().map(|text| text.lines().nth(line - 1));
            if source_line
                .is_some_and(|line| line.is_none_or(|text| text.trim() != actual.source_block))
            {
                return Err(format!("script-edges.json: stale source line {id}:{line}").into());
            }
            if matches!(disposition, PythonDisposition::Execution)
                && (target == "unknown" || target.starts_with("unresolved:"))
            {
                return Err(format!("script-edges.json: unresolved Python target {id}").into());
            }
        }
    }
    let mut edge_ids = BTreeSet::new();
    for edge in &shard.python_implementation_edges {
        if !edge_ids.insert(edge.id.as_str()) {
            return Err(format!("script-edges.json: duplicate Python edge {}", edge.id).into());
        }
        let actual = candidates
            .get(edge.id.as_str())
            .ok_or_else(|| format!("script-edges.json: stale source identity {}", edge.id))?;
        if recorded.contains(&edge.id) {
            return Err(format!("script-edges.json: duplicate Python member {}", edge.id).into());
        }
        let source_location = edge
            .source
            .rsplit_once(':')
            .and_then(|(path, line)| line.parse::<usize>().ok().map(|number| (path, number)));
        let source_line = root
            .map(|root| fs::read_to_string(root.join(&actual.path)))
            .transpose()?;
        if source_location.is_none_or(|(path, line)| {
            path != actual.path
                || line == 0
                || source_line.as_ref().is_some_and(|source| {
                    source
                        .lines()
                        .nth(line - 1)
                        .is_none_or(|text| text.trim() != actual.source_block)
                })
        }) || edge.source_block != actual.source_block
            || edge.caller_contract.trim().is_empty()
            || edge.target.trim().is_empty()
            || edge.root.trim().is_empty()
            || edge.boundary.trim().is_empty()
            || matches!(edge.disposition, PythonDisposition::Execution)
                && (edge.target == "unknown" || edge.target.starts_with("unresolved:"))
        {
            return Err(format!("script-edges.json: changed Python edge {}", edge.id).into());
        }
        recorded.insert(edge.id.clone());
    }
    let missing = candidates
        .keys()
        .filter(|id| !recorded.contains(**id))
        .take(5)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("script-edges.json: missing source members {missing:?}").into());
    }
    if let Some(root) = root {
        let mut roster = BTreeSet::new();
        for (path, digest, count) in &shard.python_implementation_sources {
            if !path.starts_with("scripts/")
                || !path.ends_with(".py")
                || path.starts_with("scripts/tests/")
                || path.split('/').any(|part| part == "..")
                || !roster.insert(path.as_str())
            {
                return Err(
                    format!("script-edges.json: invalid Python source roster {path}").into(),
                );
            }
            let bytes = fs::read(root.join(path))?;
            let text = std::str::from_utf8(&bytes)?;
            if hex::encode(Sha256::digest(&bytes)) != *digest || text.split('\n').count() != *count
            {
                return Err(format!("script-edges.json: changed Python source {path}").into());
            }
        }
        let observed_files = candidates
            .values()
            .map(|row| row.path.as_str())
            .collect::<BTreeSet<_>>();
        if roster != observed_files || !files.is_subset(&roster) {
            return Err("script-edges.json: missing Python source roster entries".into());
        }
    }
    Ok(())
}
