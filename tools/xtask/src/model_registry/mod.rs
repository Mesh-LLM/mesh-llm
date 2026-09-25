//! `models`: the test-model artifact registry. `generate` projects
//! `ci/model-artifacts/registry.json` into suite manifests and the family
//! roster; `resolve` and `restore-inputs` hand consumers exactly one
//! cadence-authorized, integrity-pinned artifact. Outputs are byte-compatible
//! with the legacy Python generator, resolver and restore-action step.

mod argv;
mod certification;
mod family_roster;
mod fields;
mod generate;
mod manifest;
mod projection;
mod python_json;
mod registry;
mod resolve;
mod restore_inputs;

use crate::command::DynResult;
use std::path::Path;

/// A `models` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum ModelsCommand {
    Generate,
    Resolve,
    RestoreInputs,
}

impl ModelsCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "generate" => Some(Self::Generate),
            "resolve" => Some(Self::Resolve),
            "restore-inputs" => Some(Self::RestoreInputs),
            _ => None,
        }
    }
}

/// `root` is only resolved for `generate`, whose outputs are checkout paths;
/// the resolver takes paths relative to the working directory.
pub(crate) fn run(
    command: ModelsCommand,
    args: &[String],
    root: impl FnOnce() -> DynResult<std::path::PathBuf>,
) -> DynResult<()> {
    let report = match command {
        ModelsCommand::Generate => generate::run(Path::new(&root()?), args),
        ModelsCommand::Resolve => resolve::run(args),
        ModelsCommand::RestoreInputs => restore_inputs::run(args),
    };
    report.emit()
}
