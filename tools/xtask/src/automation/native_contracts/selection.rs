use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

#[derive(Deserialize)]
struct Series {
    schema_version: u8,
    generator_version: String,
    shards: Vec<Shard>,
}

#[derive(Deserialize)]
struct Shard {
    sources: Vec<String>,
    families: Vec<String>,
    sha256: String,
}

#[derive(Serialize)]
struct Selection {
    families: BTreeSet<String>,
    mode: &'static str,
    reason: &'static str,
}

impl Selection {
    fn empty(mode: &'static str, reason: &'static str) -> Self {
        Self {
            families: BTreeSet::new(),
            mode,
            reason,
        }
    }
}

fn keyed(series: &Series) -> DynResult<BTreeMap<&[String], &Shard>> {
    if series.schema_version != 1 {
        return Err("invalid generated family series schema".into());
    }
    let mut result = BTreeMap::new();
    for shard in &series.shards {
        if shard.sources.is_empty() || result.insert(shard.sources.as_slice(), shard).is_some() {
            return Err("duplicate or empty source sets".into());
        }
    }
    Ok(result)
}

fn select(base: &Series, current: &Series) -> DynResult<Selection> {
    let old = keyed(base)?;
    let new = keyed(current)?;
    if base.generator_version != current.generator_version {
        return Ok(Selection::empty("full", "generator-version-changed"));
    }
    let keys: BTreeSet<_> = old.keys().chain(new.keys()).copied().collect();
    let mut selection = Selection::empty("none", "no-shard-changes");
    for key in keys {
        if old.get(key).map(|shard| &shard.sha256) == new.get(key).map(|shard| &shard.sha256) {
            continue;
        }
        selection.mode = "targeted";
        selection.reason = "mapped-shards-changed";
        for shard in [old.get(key), new.get(key)].into_iter().flatten() {
            if shard.families.is_empty() {
                return Ok(Selection::empty("full", "unmapped-shard-changed"));
            }
            selection.families.extend(shard.families.iter().cloned());
        }
    }
    Ok(selection)
}

#[derive(Deserialize)]
struct FamilyMap {
    schema_version: u8,
    families: BTreeMap<String, Vec<String>>,
}

fn select_paths(paths: &str, map: &FamilyMap) -> DynResult<Selection> {
    if map.schema_version != 1
        || map.families.iter().any(|(family, sources)| {
            family.is_empty() || sources.is_empty() || sources.iter().any(String::is_empty)
        })
    {
        return Err("invalid generated family source map".into());
    }
    let mut result = Selection::empty("none", "no-upstream-changes");
    for path in paths.lines().map(str::trim).filter(|path| !path.is_empty()) {
        let owners: Vec<_> = map
            .families
            .iter()
            .filter(|(_, sources)| sources.iter().any(|source| source == path))
            .map(|(family, _)| family.clone())
            .collect();
        if owners.is_empty() {
            return Ok(Selection::empty(
                "full",
                if path.starts_with("src/models/") && path.ends_with(".cpp") {
                    "unmapped-model-source-changed"
                } else {
                    "shared-upstream-source-changed"
                },
            ));
        }
        result.mode = "targeted";
        result.reason = "mapped-upstream-model-sources-changed";
        result.families.extend(owners);
    }
    Ok(result)
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let values = super::options(
        args,
        &[
            "--base",
            "--current",
            "--changed-paths",
            "--family-map",
            "--output",
            "--include-sentinels",
        ],
    )?;
    let shard_mode = values.contains_key("--base") || values.contains_key("--current");
    let path_mode = values.contains_key("--changed-paths") || values.contains_key("--family-map");
    if shard_mode == path_mode {
        return Err("provide either --base/--current or --changed-paths/--family-map".into());
    }
    let mut result = if shard_mode {
        let base = serde_json::from_slice(&std::fs::read(super::required(&values, "--base")?)?)?;
        let current =
            serde_json::from_slice(&std::fs::read(super::required(&values, "--current")?)?)?;
        select(&base, &current)?
    } else {
        let paths = std::fs::read_to_string(super::required(&values, "--changed-paths")?)?;
        let map =
            serde_json::from_slice(&std::fs::read(super::required(&values, "--family-map")?)?)?;
        select_paths(&paths, &map)?
    };
    if result.mode == "targeted" && values.contains_key("--include-sentinels") {
        result
            .families
            .extend(["qwen3-dense", "qwen3-moe", "mamba", "lfm2-vl"].map(str::to_owned));
    }
    let mut bytes = serde_json::to_vec_pretty(&result)?;
    bytes.push(b'\n');
    if values.contains_key("--output") {
        super::publish(&super::required(&values, "--output")?, &bytes, false)?;
    }
    crate::cli_output::stdout().write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removed_shard_retains_previous_owners() {
        let base = Series {
            schema_version: 1,
            generator_version: "v1".into(),
            shards: vec![Shard {
                sources: vec!["src/models/a.cpp".into()],
                families: vec!["a".into()],
                sha256: "old".into(),
            }],
        };
        let current = Series {
            schema_version: 1,
            generator_version: "v1".into(),
            shards: vec![],
        };
        let result = select(&base, &current).unwrap();
        assert_eq!(result.families, BTreeSet::from(["a".into()]));
    }

    #[test]
    fn shared_source_requires_full_roster() {
        let map = FamilyMap {
            schema_version: 1,
            families: BTreeMap::from([("a".into(), vec!["src/models/a.cpp".into()])]),
        };
        let result = select_paths("src/llama-graph.cpp", &map).unwrap();
        assert_eq!(result.mode, "full");
    }
}
