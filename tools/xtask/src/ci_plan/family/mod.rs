mod arguments;
mod artifact;
mod document;
mod equality;
mod execution;
mod failure;
mod fields;
mod integer;
mod model;
mod output;
mod plan;
mod policy;
mod projection;
mod resources;
mod shard_count;
mod shards;
mod text;

use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use arguments::{Arguments, Parsed};
use failure::Failure;
use std::path::Path;

pub(crate) fn controller_plan(root: &Path, manifest: &Path) -> DynResult<Vec<u8>> {
    let families = text::FamilyString::from("");
    let count = shard_count::ShardCount::from(256);
    let built = plan::build(
        root,
        manifest,
        plan::Selection {
            families: &families,
            shard_count: &count,
        },
    )
    .map_err(controller_error)?;
    Ok(output::write(&built, (None, None))
        .map_err(controller_error)?
        .into_bytes())
}

pub(crate) fn verify_controller_plan(
    root: &Path,
    manifest: &Path,
    supplied: &Path,
) -> DynResult<()> {
    plan::verify(root, manifest, supplied).map_err(controller_error)?;
    Ok(())
}

fn controller_error(error: Failure) -> crate::command::DynError {
    error.report().stderr.into()
}

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    let report = match arguments::parse(args) {
        Err(report) => report,
        Ok(Parsed::Help) => arguments::help(),
        Ok(Parsed::Options(parsed)) => dispatch(root, &parsed),
    };
    report.emit()
}

fn dispatch(root: &Path, parsed: &Arguments<'_>) -> CheckReport {
    let outcome = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("family-plan".into())
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(scope, || execute(root, parsed))
            .map_err(Failure::io)?;
        worker
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    });
    match outcome {
        Ok(stdout) => CheckReport::success(stdout),
        Err(error) => error.report(),
    }
}

fn execute(root: &Path, parsed: &Arguments<'_>) -> Result<String, Failure> {
    let manifest = parsed
        .manifest
        .clone()
        .unwrap_or_else(|| root.join("ci/llama-canary/family-certified.json"));
    if let Some(path) = &parsed.verify_plan {
        return plan::verify(root, &manifest, path).map(|()| String::new());
    }
    let families = text::FamilyString::from(parsed.families);
    let selection = plan::Selection {
        families: &families,
        shard_count: &parsed.count,
    };
    let built = plan::build(root, &manifest, selection)?;
    output::write(
        &built,
        (parsed.output.as_deref(), parsed.github_output.as_deref()),
    )
}
