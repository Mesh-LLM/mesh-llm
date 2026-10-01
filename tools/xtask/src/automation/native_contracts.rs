mod api_doc;
mod coverage;
mod inventory;
mod selection;
mod spec_manifest;

use crate::command::DynResult;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [flag] if flag == "--help" => {
            use std::io::Write;
            writeln!(crate::cli_output::stdout(), "cargo xtool automation native-generator contracts {{coverage|select-shards|spec-manifest|inventory|api-doc}} [typed options] [--check for generated documents]")?;
            Ok(())
        }
        [verb, rest @ ..] if verb == "coverage" => coverage::run(rest),
        [verb, rest @ ..] if verb == "select-shards" => selection::run(rest),
        [verb, rest @ ..] if verb == "spec-manifest" => spec_manifest::run(rest),
        [verb, rest @ ..] if verb == "inventory" => inventory::run(rest),
        [verb, rest @ ..] if verb == "api-doc" => api_doc::run(rest),
        _ => Err("usage: cargo xtool automation native-generator contracts {coverage|select-shards|spec-manifest|inventory|api-doc} ...".into()),
    }
}

fn options(args: &[String], allowed: &[&str]) -> DynResult<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    let mut remaining = args;
    while let Some((key, rest)) = remaining.split_first() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown option: {key}").into());
        }
        let (value, rest) = match key.as_str() {
            "--check" | "--include-sentinels" => ("true".to_owned(), rest),
            _ => {
                let (value, rest) = rest.split_first().ok_or("missing option value")?;
                (value.clone(), rest)
            }
        };
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option: {key}").into());
        }
        remaining = rest;
    }
    Ok(values)
}

fn required(values: &BTreeMap<String, String>, key: &str) -> DynResult<PathBuf> {
    Ok(PathBuf::from(
        values.get(key).ok_or_else(|| format!("missing {key}"))?,
    ))
}

fn publish(path: &Path, bytes: &[u8], check: bool) -> DynResult<()> {
    if check {
        if std::fs::read(path)? != bytes {
            return Err(format!("stale generated output: {}", path.display()).into());
        }
    } else {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_rejects_stale_output_without_overwriting_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("output");
        std::fs::write(&path, b"stale").unwrap();
        let result = publish(&path, b"fresh", true);
        assert!(result.is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"stale");
    }
}
