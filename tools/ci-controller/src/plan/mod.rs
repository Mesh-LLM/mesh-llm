//! `ci plan`: the Rust owner of `scripts/plan-ci.py`. Reads the planner input
//! on stdin and prints the versioned, provider-neutral CI plan.
//!
//! `--manifest-root` is the audit path. Cargo discovery and the
//! reverse-dependency closure always run in the protected checkout; only the
//! `ci/{ownership,slices}.yml` routing catalogs are read from the manifest
//! root, which a PR caller points at the source revision's copies after
//! proving them byte-identical to the protected ones. The two roots are
//! deliberately different trust domains.

pub mod catalog;
mod construction;
pub mod diagnostics;
pub mod document;
mod glob_pattern;
mod matrices;
pub use construction::CONTRACT_TESTED_CRATES;
pub mod plan_bytes;
mod profile_catalog;
mod request;
mod row_catalog;
pub mod selection;
mod signals;
mod slice_catalog;
mod slice_graph;
mod workspace;

use crate::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::io::Read;
use std::path::{Path, PathBuf};

const GRAMMAR: Grammar = Grammar {
    usage: "ci plan [--manifest-root MANIFEST_ROOT]",
    values: &["--manifest-root"],
    flags: &[],
};

/// `root` is the protected checkout; catalogs default to it.
pub fn run(root: &Path, args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) if parsed.positionals.is_empty() => parsed,
        Ok(parsed) => {
            let extra = parsed.positionals.join(" ");
            return usage_error(&format!("unrecognized arguments: {extra}"));
        }
        Err(report) => return report.emit(),
    };
    let manifests = parsed.last("--manifest-root").map(PathBuf::from);
    let mut stdin = Vec::new();
    std::io::stdin().read_to_end(&mut stdin)?;
    let roots = construction::Roots {
        workspace: root,
        manifests: manifests.as_deref().unwrap_or(root),
    };
    report(&stdin, &roots).emit()
}

pub fn build_for_validation(root: &Path, input: &[u8]) -> DynResult<serde_json::Value> {
    let payload = document::Json::parse(input)?;
    construction::build(
        &payload,
        &construction::Roots {
            workspace: root,
            manifests: root,
        },
    )
    .map_err(|error| error.0.into())
}

fn report(stdin: &[u8], roots: &construction::Roots<'_>) -> CheckReport {
    // Invalid JSON keeps the legacy prefix and status; the parser detail is
    // serde_json's wording rather than Python's `JSONDecodeError` text.
    let outcome = document::Json::parse(stdin)
        .map_err(|error| diagnostics::PlanError(error.to_string()))
        .and_then(|payload| construction::build(&payload, roots));
    match outcome {
        Ok(plan) => CheckReport::success(plan_bytes::render(&plan) + "\n"),
        Err(error) => CheckReport {
            stdout: String::new(),
            stderr: format!("ERROR: unable to build CI plan: {error}\n"),
            code: 2,
        },
    }
}

fn usage_error(message: &str) -> DynResult<()> {
    GRAMMAR.error(message).emit()
}

/// `_select_rows(slices, profile="main", domains=[], selected=set(),
/// force_all_rows=True)["runtime_products"]` after `_validate_manifests`,
/// for callers that audit planner rows against another catalog. Errors are
/// the `PlanError` text.
pub fn exhaustive_runtime_rows(
    ownership: &document::Json,
    slices: &document::Json,
) -> Result<Vec<serde_json::Value>, String> {
    let owners = catalog::validate_ownership(ownership).map_err(|error| error.0)?;
    let catalog = slice_catalog::validate(slices, &owners).map_err(|error| error.0)?;
    let request = matrices::RowRequest {
        profile: request::Profile::Main,
        domains: &[],
        required: &[],
        force_all_rows: true,
    };
    let mut rows = matrices::select_rows(&catalog.rows, &request).map_err(|error| error.0)?;
    match rows.remove("runtime_products") {
        Some(serde_json::Value::Array(items)) => Ok(items),
        _ => Ok(Vec::new()),
    }
}
