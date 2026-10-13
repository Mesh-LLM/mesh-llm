use super::super::{Error, args};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

pub(super) struct Options {
    pub(super) publication: args::Options,
    pub(super) build: PathBuf,
    pub(super) rewriter: PathBuf,
    pub(super) report: PathBuf,
    pub(super) extra: Vec<String>,
}

pub(super) fn parse(arguments: &[String]) -> Result<Options, Error> {
    let (pairs, remainder) = arguments.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(Error::Arguments("missing value"));
    }
    let mut generation = BTreeMap::new();
    let mut publication = Vec::new();
    let mut extra = Vec::new();
    for [key, value] in pairs {
        match key.as_str() {
            "--extra-arg" => extra.push(value.clone()),
            "--build-dir" | "--rewriter" | "--report" => {
                if generation.insert(key.as_str(), value.as_str()).is_some() {
                    return Err(Error::Arguments("duplicate option"));
                }
            }
            _ => publication.extend([key.clone(), value.clone()]),
        }
    }
    let required = |key| {
        generation
            .get(key)
            .copied()
            .ok_or(Error::Arguments("missing required generation option"))
    };
    let rewriter = PathBuf::from(required("--rewriter")?);
    if !rewriter.is_absolute() {
        return Err(Error::Arguments("--rewriter must be absolute"));
    }
    let mut publication = args::parse(&publication)?;
    publication.output = resolve(&publication.output)?;
    if publication.output.to_str().is_none() {
        return Err(Error::Arguments("output path must be UTF-8"));
    }
    if let Some(shards) = &mut publication.shards {
        shards.output = resolve(&shards.output)?;
        shards.map = resolve(&shards.map)?;
        shards.manifest = resolve(&shards.manifest)?;
    }
    Ok(Options {
        publication,
        build: resolve(Path::new(required("--build-dir")?))?,
        rewriter: resolve(&rewriter)?,
        report: resolve(Path::new(required("--report")?))?,
        extra,
    })
}

fn resolve(path: &Path) -> Result<PathBuf, Error> {
    let absolute = std::env::current_dir()?.join(path);
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                resolved.pop();
            }
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                resolved.push(component.as_os_str());
                match std::fs::canonicalize(&resolved) {
                    Ok(path) => resolved = path,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
    Ok(resolved)
}
