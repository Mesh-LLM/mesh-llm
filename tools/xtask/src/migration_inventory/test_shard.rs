use super::scan::{self, Candidate};
use super::shard_rows::TestShard;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

fn source_lines(path: &str, text: &str) -> BTreeMap<usize, Candidate> {
    let mut scanned = scan::scan_source(path, text).into_iter();
    text.lines()
        .enumerate()
        .filter_map(|(index, line)| {
            (!scan::scan_source(path, line).is_empty())
                .then(|| scanned.next().map(|candidate| (index + 1, candidate)))
                .flatten()
        })
        .collect()
}

fn record(
    id: &str,
    line: usize,
    target: &str,
    candidates: &BTreeMap<&str, &Candidate>,
    located: Option<&BTreeMap<usize, Candidate>>,
    recorded: &mut BTreeSet<String>,
) -> DynResult<()> {
    if line == 0 || target.trim().is_empty() {
        return Err(format!("test-edges.json: incomplete test edge {id}").into());
    }
    if target == "unknown" || target.starts_with("unresolved:") {
        return Err(format!("test-edges.json: unresolved test target {id}").into());
    }
    if !recorded.insert(id.to_owned()) {
        return Err(format!("test-edges.json: duplicate test edge {id}").into());
    }
    if !candidates.contains_key(id) {
        return Err(format!("test-edges.json: stale source identity {id}").into());
    }
    if located.is_some_and(|lines| lines.get(&line).is_none_or(|row| row.id != id)) {
        return Err(format!("test-edges.json: stale source line {id}:{line}").into());
    }
    Ok(())
}

pub(super) fn validate(root: Option<&Path>, text: &str, observed: &[Candidate]) -> DynResult<()> {
    let shard: TestShard = serde_json::from_str(text)?;
    if shard.schema_version != 1 {
        return Err("test-edges.json: unsupported shard version".into());
    }
    let candidates = observed
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let paths = observed
        .iter()
        .map(|row| row.path.as_str())
        .collect::<BTreeSet<_>>();
    let mut located = BTreeMap::new();
    let mut digest = Sha256::new();
    if let Some(root) = root {
        for path in paths {
            let source = fs::read_to_string(root.join(path))?;
            digest.update(path.as_bytes());
            digest.update([0]);
            digest.update(source.as_bytes());
            digest.update([0]);
            located.insert(path, source_lines(path, &source));
        }
        if hex::encode(digest.finalize()) != shard.observed_source_sha256 {
            return Err("test-edges.json: changed test source tree".into());
        }
    }
    let mut recorded = BTreeSet::new();
    for group in &shard.groups {
        if !valid_path(&group.file) {
            return Err(format!("test-edges.json: invalid test group {}", group.file).into());
        }
        for (line, hash, occurrence, kind, _disposition, target, reason) in &group.members {
            let id = format!("{}#{}:{hash}:{occurrence}", group.file, kind.label());
            if *occurrence == 0
                || hash.len() != 16
                || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                || reason.trim().is_empty()
            {
                return Err(format!("test-edges.json: incomplete test edge {id}").into());
            }
            record(
                &id,
                *line,
                target,
                &candidates,
                located.get(group.file.as_str()),
                &mut recorded,
            )?;
        }
    }
    for (path, cases) in &shard.additional_groups {
        if !valid_path(path) {
            return Err(format!("test-edges.json: invalid test group {path}").into());
        }
        for (_disposition, target, reason, lines) in cases {
            if reason.trim().is_empty() || lines.is_empty() {
                return Err(
                    format!("test-edges.json: incomplete test caller contract {path}").into(),
                );
            }
            for line in lines {
                let id = located
                    .get(path.as_str())
                    .and_then(|lines| lines.get(line))
                    .map(|row| row.id.as_str())
                    .ok_or_else(|| format!("test-edges.json: stale source line {path}:{line}"))?;
                record(
                    id,
                    *line,
                    target,
                    &candidates,
                    located.get(path.as_str()),
                    &mut recorded,
                )?;
            }
        }
    }
    let missing = candidates
        .keys()
        .filter(|id| !recorded.contains(**id))
        .take(5)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("test-edges.json: missing source members {missing:?}").into());
    }
    Ok(())
}

fn valid_path(path: &str) -> bool {
    path.starts_with("scripts/tests/")
        && path.ends_with(".py")
        && !path.split('/').any(|part| part == "..")
}
