use super::{Error, GitDiff};
use crate::process::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
pub(super) struct ShardOutput {
    pub(super) output: PathBuf,
    pub(super) map: PathBuf,
    pub(super) manifest: PathBuf,
}
pub(super) struct Options {
    pub(super) git: GitDiff,
    pub(super) output: PathBuf,
    pub(super) shards: Option<ShardOutput>,
}
pub(super) fn parse(arguments: &[String]) -> Result<Options, Error> {
    let mut values = BTreeMap::new();
    let (pairs, remainder) = arguments.as_chunks::<2>();
    for [key, value] in pairs {
        if !matches!(
            key.as_str(),
            "--source-root"
                | "--git"
                | "--output"
                | "--max-diff-bytes"
                | "--diff-base"
                | "--timeout"
                | "--shard-output-dir"
                | "--family-source-map"
                | "--family-manifest"
        ) {
            return Err(Error::Arguments("unknown option"));
        }
        if values.insert(key.as_str(), value.as_str()).is_some() {
            return Err(Error::Arguments("duplicate option"));
        }
    }
    if !remainder.is_empty() {
        return Err(Error::Arguments("missing value"));
    }
    let required = |key: &str| {
        values
            .get(key)
            .copied()
            .ok_or(Error::Arguments("missing required option"))
    };
    let cwd = std::env::current_dir()?;
    let absolute = |value: &str| cwd.join(value);
    let executable = PathBuf::from(required("--git")?);
    if !executable.is_absolute() {
        return Err(Error::Arguments("--git must be absolute"));
    }
    let max_bytes = required("--max-diff-bytes")?
        .parse()
        .map_err(|_| Error::Arguments("invalid byte cap"))?;
    let timeout: u64 = values
        .get("--timeout")
        .copied()
        .unwrap_or("60")
        .parse()
        .map_err(|_| Error::Arguments("invalid timeout"))?;
    if !(1..=86400).contains(&timeout) {
        return Err(Error::Arguments("timeout must be 1..=86400 seconds"));
    }
    let shards = match (
        values.get("--shard-output-dir"),
        values.get("--family-source-map"),
        values.get("--family-manifest"),
    ) {
        (None, None, None) => None,
        (Some(output), Some(map), Some(manifest)) => Some(ShardOutput {
            output: absolute(output),
            map: absolute(map),
            manifest: absolute(manifest),
        }),
        _ => return Err(Error::Arguments("shard options must be used together")),
    };
    let environment = [
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
    .collect();
    Ok(Options {
        git: GitDiff {
            executable,
            source_root: std::fs::canonicalize(required("--source-root")?)?,
            base: values
                .get("--diff-base")
                .copied()
                .unwrap_or("HEAD")
                .to_owned(),
            environment,
            max_bytes,
            timeout: std::time::Duration::from_secs(timeout),
        },
        output: absolute(required("--output")?),
        shards,
    })
}
