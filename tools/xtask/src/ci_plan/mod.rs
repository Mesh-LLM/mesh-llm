//! `ci plan`: the Rust owner of `scripts/plan-ci.py`. Reads the planner input
//! on stdin and prints the versioned, provider-neutral CI plan.
//!
//! `--manifest-root` is the audit path. Cargo discovery and the
//! reverse-dependency closure always run in the protected checkout; only the
//! `ci/{ownership,slices}.yml` routing catalogs are read from the manifest
//! root, which a PR caller points at the source revision's copies after
//! proving them byte-identical to the protected ones. The two roots are
//! deliberately different trust domains.

pub(crate) mod catalog;
mod diagnostics;
pub(crate) mod document;
mod glob_pattern;
mod matrices;
mod plan;
pub(crate) mod plan_bytes;
mod profile_catalog;
mod request;
mod row_catalog;
mod selection;
mod signals;
mod slice_catalog;
mod slice_graph;
mod workspace;

use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::io::Read;
use std::path::{Path, PathBuf};

const GRAMMAR: Grammar = Grammar {
    usage: "plan-ci.py [-h] [--manifest-root MANIFEST_ROOT]",
    values: &["--manifest-root"],
    flags: &[],
};

/// `root` is the protected checkout; catalogs default to it.
pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) if parsed.positionals.is_empty() => parsed,
        Ok(parsed) => {
            let extra = parsed.positionals.join(" ");
            return usage_error(&format!("unrecognized arguments: {extra}"));
        }
        Err(report) => return argparse_report(report).emit(),
    };
    let manifests = parsed.last("--manifest-root").map(PathBuf::from);
    let mut stdin = Vec::new();
    std::io::stdin().read_to_end(&mut stdin)?;
    let roots = plan::Roots {
        workspace: root,
        manifests: manifests.as_deref().unwrap_or(root),
    };
    report(&stdin, &roots).emit()
}

fn report(stdin: &[u8], roots: &plan::Roots<'_>) -> CheckReport {
    // Invalid JSON keeps the legacy prefix and status; the parser detail is
    // serde_json's wording rather than Python's `JSONDecodeError` text.
    let outcome = document::Json::parse(stdin)
        .map_err(|error| diagnostics::PlanError(error.to_string()))
        .and_then(|payload| plan::build(&payload, roots));
    match outcome {
        Ok(plan) => CheckReport::success(plan_bytes::render(&plan) + "\n"),
        Err(error) => CheckReport {
            stdout: String::new(),
            stderr: format!("ERROR: unable to build CI plan: {error}\n"),
            code: 2,
        },
    }
}

/// argparse prints `usage: ...` then `<prog>: error: ...` with status 2.
fn argparse_report(report: CheckReport) -> CheckReport {
    CheckReport {
        stderr: report
            .stderr
            .replacen("\nerror: ", "\nplan-ci.py: error: ", 1),
        ..report
    }
}

fn usage_error(message: &str) -> DynResult<()> {
    argparse_report(GRAMMAR.error(message)).emit()
}
