//! Read-only prepared-source SHA adapter for local workload oracle callers.
use super::{process, source};
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation canary-receipts prepared-source --root ABSOLUTE_PATH",
    values: &["--root"],
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
    let root = Path::new(parsed.last("--root").ok_or("missing --root")?);
    if !root.is_absolute() {
        return Err("prepared source requires an explicit absolute repository root".into());
    }
    let root = root.canonicalize()?;
    let provenance = process::operation(|| source::prepared(&root))?;
    CheckReport::success(format!("{}\n", provenance.head)).emit()
}
