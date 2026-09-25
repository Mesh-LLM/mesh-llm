use super::other_shard;
use super::python_shard;
use super::scan::Candidate;
use super::script_source_calls;
use super::shard_rows::{GithubMember, Group, Member, ScriptShard, Shard};
use super::test_shard;
use crate::command::DynResult;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const SHARDS: [&str; 4] = [
    "github-edges.json",
    "script-edges.json",
    "test-edges.json",
    "other-edges.json",
];

#[cfg(test)]
pub(super) fn check_other_shard(root: &Path, text: &str, observed: &[Candidate]) -> DynResult<()> {
    other_shard::check_other_shard(root, text, observed)
}

fn validate_groups<M>(
    name: &str,
    shard: Shard<Group<M>>,
    observed: &[Candidate],
    complete: bool,
    root: Option<&Path>,
) -> DynResult<()>
where
    M: Into<Member>,
{
    if shard.schema_version != 1 {
        return Err(format!("{name}: unsupported shard version").into());
    }
    let source = observed
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut recorded = BTreeSet::new();
    for group in shard.groups {
        let lines = root
            .map(|root| fs::read_to_string(root.join(&group.file)))
            .transpose()?
            .map(|text| {
                text.lines()
                    .map(str::trim)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            });
        for raw in group.members {
            let row: Member = raw.into();
            if row.line == 0
                || row.occurrence == 0
                || row.hash.len() != 16
                || !row.hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                || row.details.iter().any(|detail| detail.trim().is_empty())
            {
                return Err(format!("{name}: incomplete member in {}", group.file).into());
            }
            let id = format!(
                "{}#{}:{}:{}",
                group.file, row.kind, row.hash, row.occurrence
            );
            if !recorded.insert(id.clone()) {
                return Err(format!("{name}: duplicate member {id}").into());
            }
            let Some(actual) = source.get(id.as_str()) else {
                return Err(format!("{name}: stale source identity {id}").into());
            };
            if lines.as_ref().is_some_and(|lines| {
                lines
                    .get(row.line - 1)
                    .is_none_or(|text| text != &actual.source_block)
            }) {
                return Err(format!("{name}: stale source line {id}:{}", row.line).into());
            }
            if name == "github-edges.json" && actual.executable != row.executable {
                return Err(format!("{name}: incorrect execution disposition {id}").into());
            }
        }
    }
    let missing = source
        .values()
        .filter(|row| (complete || row.executable) && !recorded.contains(row.id.as_str()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        let examples = missing
            .iter()
            .take(5)
            .map(|row| row.id.as_str())
            .collect::<Vec<_>>();
        return Err(format!(
            "{name}: {} missing source members {examples:?}",
            missing.len()
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn check_shard(name: &str, text: &str, observed: &[Candidate]) -> DynResult<()> {
    validate_shard(name, text, observed, None)
}

#[cfg(test)]
pub(super) fn check_script_source_shard(
    root: &Path,
    text: &str,
    observed: &[Candidate],
) -> DynResult<()> {
    validate_shard("script-edges.json", text, observed, Some(root))
}

#[cfg(test)]
pub(super) fn check_test_source_shard(
    root: &Path,
    text: &str,
    observed: &[Candidate],
) -> DynResult<()> {
    validate_shard("test-edges.json", text, observed, Some(root))
}

fn validate_shard(
    name: &str,
    text: &str,
    observed: &[Candidate],
    root: Option<&Path>,
) -> DynResult<()> {
    match name {
        "github-edges.json" => validate_groups(
            name,
            serde_json::from_str::<Shard<Group<GithubMember>>>(text)?,
            observed,
            true,
            root,
        ),
        "script-edges.json" => {
            let shard: ScriptShard = serde_json::from_str(text)?;
            python_shard::validate(root, &shard, observed)?;
            if let Some(root) = root {
                script_source_calls::check_script_source_calls(root, &shard.source_calls)?;
            }
            validate_groups(
                name,
                Shard {
                    schema_version: shard.schema_version,
                    groups: shard.groups,
                },
                &observed
                    .iter()
                    .filter(|row| !row.path.ends_with(".py"))
                    .cloned()
                    .collect::<Vec<_>>(),
                true,
                root,
            )
        }
        "test-edges.json" => test_shard::validate(root, text, observed),
        _ => Err(format!("unknown automation shard {name}").into()),
    }
}

pub(super) fn check_shards(root: &Path, observed: &[Candidate]) -> DynResult<BTreeSet<String>> {
    for name in SHARDS {
        let path = root.join("ci/automation-migration").join(name);
        let text = fs::read_to_string(&path)?;
        let subset = observed
            .iter()
            .filter(|row| match name {
                "github-edges.json" => {
                    row.path.starts_with(".github/")
                        && (row.path.ends_with(".yml") || row.path.ends_with(".yaml"))
                }
                "script-edges.json" => {
                    row.path == "Justfile"
                        || row.path.starts_with("just/")
                        || row.path.starts_with("scripts/") && !row.path.ends_with(".py")
                        || row.path == "tools/skippy-stage-rewriter/CMakeLists.txt"
                        || row.path.starts_with("scripts/")
                            && row.path.ends_with(".py")
                            && !row.path.starts_with("scripts/tests/")
                }
                "test-edges.json" => {
                    row.path.starts_with("scripts/tests/") && row.path.ends_with(".py")
                }
                "other-edges.json" => {
                    !row.path.starts_with(".github/")
                        && row.path != "Justfile"
                        && !row.path.starts_with("just/")
                        && !row.path.starts_with("scripts/")
                        && row.path != "tools/skippy-stage-rewriter/CMakeLists.txt"
                }
                _ => unreachable!("closed shard list"),
            })
            .cloned()
            .collect::<Vec<_>>();
        if name == "other-edges.json" {
            other_shard::check_other_shard(root, &text, &subset)?;
        } else {
            validate_shard(name, &text, &subset, Some(root))?;
        }
    }
    Ok(observed.iter().map(|row| row.id.clone()).collect())
}
