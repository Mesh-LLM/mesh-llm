//! Retain failed repair source as bounded, explicitly unverified diagnostics.
#[cfg(unix)]
use super::process;
use super::source;
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::path::Path;

#[cfg(unix)]
#[path = "source_recovery/archive.rs"]
mod archive;
#[cfg(unix)]
#[path = "source_recovery/git.rs"]
mod git;
#[cfg(unix)]
#[path = "source_recovery/snapshot.rs"]
mod snapshot;
#[cfg(all(test, unix))]
#[path = "source_recovery/tests.rs"]
mod tests;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation canary-receipts recover-source --root ABS --output ABS --base SHA",
    values: &["--root", "--output", "--base"],
    flags: &["--help"],
};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let base = parsed.last("--base").ok_or("missing --base")?;
    source::revision(base)?;
    let root = Path::new(parsed.last("--root").ok_or("missing --root")?);
    let output = Path::new(parsed.last("--output").ok_or("missing --output")?);
    if !root.is_absolute() || !output.is_absolute() {
        return Err("recovery root and output must be absolute".into());
    }
    #[cfg(unix)]
    return process::operation(|| snapshot::save(root, output, base));
    #[cfg(not(unix))]
    Err("diagnostic source recovery currently requires a Unix runner".into())
}
