//! Local configuration smoke execution and evidence coverage; no model acquisition.
mod coverage;
mod http;
mod runtime;
use crate::command::DynResult;
use std::{io::Read, path::Path};
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [verb, rest @ ..] if verb == "coverage" => coverage::run(rest),
        [verb, rest @ ..] if verb == "run" => runtime::run(rest),
        _ => Err("usage: cargo xtool automation manual-smoke {run|coverage} ...".into()),
    }
}
fn text(path: &Path) -> DynResult<String> {
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("manual smoke input/evidence must be a regular file".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened manual smoke input is not regular".into());
    }
    let mut bytes = Vec::new();
    file.take(4 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err("manual smoke text exceeds 4 MiB".into());
    }
    Ok(String::from_utf8(bytes)?)
}
