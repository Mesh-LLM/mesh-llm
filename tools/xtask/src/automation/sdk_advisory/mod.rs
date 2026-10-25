mod admission;
mod command;
mod coverage;
mod error;
mod identity;
mod input;
mod rows;
mod selection;

use crate::command::DynResult;
use std::path::Path;

pub(crate) const USAGE: &str = "cargo xtool automation sdk-advisory --event-name <workflow_run|workflow_dispatch> --controller-repository <owner/repo> --controller-ref <ref> --event <json> --producer-run <json> --artifacts <json>";

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    command::report(root, args).emit()
}

#[cfg(test)]
#[path = "../../../tests/migration_sdk_advisory/mod.rs"]
mod migration_sdk_advisory;
