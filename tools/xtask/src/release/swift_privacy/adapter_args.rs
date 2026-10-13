use super::adapter_error::Error;
use std::{num::NonZeroUsize, path::Path, time::Duration};

pub(super) const USAGE: &str = "cargo xtool release swift-privacy --template <path> [--xcframework <path>] --plutil <absolute-executable> [--timeout-ms <1..=86400000>] [--grace-ms <1..=86400000>] [--cleanup-ms <1..=86400000>] [--max-output-bytes <1..=16777216>]\nExperimental. Paths are borrowed identities; concurrent filesystem mutation is not supported.";

pub(super) struct Options<'a> {
    pub(super) template: &'a Path,
    pub(super) xcframework: Option<&'a Path>,
    pub(super) plutil: &'a Path,
    pub(super) budget: NativeBudget,
}

pub(super) struct NativeBudget {
    pub(super) execution: Duration,
    pub(super) grace: Duration,
    pub(super) cleanup: Duration,
    pub(super) output: NonZeroUsize,
}

pub(super) fn parse(arguments: &[String]) -> Result<Options<'_>, Error> {
    let mut template = None;
    let mut xcframework = None;
    let mut plutil = None;
    let mut execution = 30000;
    let mut grace = 1000;
    let mut cleanup = 2000;
    let mut output = 65536;
    let mut seen = std::collections::BTreeSet::new();
    let (pairs, remainder) = arguments.as_chunks::<2>();
    for [name, value] in pairs {
        if !seen.insert(name.as_str()) || value.is_empty() {
            return Err(Error::Arguments("duplicate option or empty value"));
        }
        match name.as_str() {
            "--template" => template = Some(Path::new(value)),
            "--xcframework" => xcframework = Some(Path::new(value)),
            "--plutil" => plutil = Some(Path::new(value)),
            "--timeout-ms" => execution = bounded(value, 86400000)?,
            "--grace-ms" => grace = bounded(value, 86400000)?,
            "--cleanup-ms" => cleanup = bounded(value, 86400000)?,
            "--max-output-bytes" => output = bounded(value, 16777216)?,
            _ => return Err(Error::Arguments("unknown option")),
        }
    }
    if !remainder.is_empty() {
        return Err(Error::Arguments("missing option value"));
    }
    let plutil = plutil.ok_or(Error::Arguments("--plutil is required"))?;
    if !plutil.is_absolute() {
        return Err(Error::Arguments("--plutil must be absolute"));
    }
    Ok(Options {
        template: template.ok_or(Error::Arguments("--template is required"))?,
        xcframework,
        plutil,
        budget: NativeBudget {
            execution: Duration::from_millis(execution),
            grace: Duration::from_millis(grace),
            cleanup: Duration::from_millis(cleanup),
            output: NonZeroUsize::new(
                usize::try_from(output)
                    .map_err(|_| Error::Arguments("output bound does not fit this platform"))?,
            )
            .ok_or(Error::Arguments("output bound must be positive"))?,
        },
    })
}

fn bounded(value: &str, maximum: u64) -> Result<u64, Error> {
    let value = value
        .parse::<u64>()
        .map_err(|_| Error::Arguments("limit must be an integer"))?;
    if value == 0 || value > maximum {
        return Err(Error::Arguments("limit is outside its documented bounds"));
    }
    Ok(value)
}
