//! Native release inventory argument admission; no source/remote mutation surface.
use super::provenance::{Error, Result};
use std::path::PathBuf;
pub(crate) const USAGE: &str =
    "release inventory [--repo OWNER/REPO] [--head REF] [--release-tag TAG] [--output PATH]";
pub(crate) struct Args {
    pub(crate) repository: String,
    pub(crate) head: String,
    pub(crate) release_tag: Option<String>,
    pub(crate) output: Option<PathBuf>,
}
pub(crate) fn parse(args: &[String]) -> Result<Option<Args>> {
    if matches!(args,[help]if matches!(help.as_str(),"--help"|"-h")) {
        return Ok(None);
    }
    let mut result = Args {
        repository: "Mesh-LLM/mesh-llm".into(),
        head: "HEAD".into(),
        release_tag: None,
        output: None,
    };
    let mut cursor = 0;
    while cursor < args.len() {
        let argument = &args[cursor];
        cursor += 1;
        let (flag, value) = if let Some(pair) = argument.split_once('=') {
            pair
        } else {
            let value = args.get(cursor).ok_or_else(|| Error(USAGE.into()))?;
            cursor += 1;
            (argument.as_str(), value.as_str())
        };
        if value.is_empty() || value.contains('\0') {
            return Err(Error(
                "release inventory options require nonempty values without NUL".into(),
            ));
        }
        match flag {
            "--repo" => result.repository = value.into(),
            "--head" => result.head = value.into(),
            "--release-tag" => result.release_tag = Some(value.into()),
            "--output" => result.output = Some(value.into()),
            _ => return Err(Error(USAGE.into())),
        }
    }
    Ok(Some(result))
}
