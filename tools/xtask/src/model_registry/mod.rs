//! `models`: the test-model artifact registry. `generate` projects
//! `ci/model-artifacts/registry.json` into suite manifests and the family
//! roster; `resolve` and `restore-inputs` hand consumers exactly one
//! cadence-authorized, integrity-pinned artifact. Their established consumer
//! formats are retained. `parity-download` owns manual discovery and verified
//! candidate materialization without inline Python.

mod argv;
mod certification;
mod family_roster;
mod fields;
mod generate;
pub(crate) mod json_bytes;
mod manifest;
pub(crate) mod parity_download;
mod projection;
mod projector_download;
mod registry;
mod resolve;
mod restore_inputs;
pub(crate) mod serving_entry;

use crate::command::DynResult;
use std::path::Path;

/// A `models` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum ModelsCommand {
    Generate,
    Resolve,
    RestoreInputs,
    ParityDownload,
    ProjectorDownload,
}

impl ModelsCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "generate" => Some(Self::Generate),
            "resolve" => Some(Self::Resolve),
            "restore-inputs" => Some(Self::RestoreInputs),
            "projector-download" => Some(Self::ProjectorDownload),
            "parity-download" => Some(Self::ParityDownload),
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
    if matches!(command, ModelsCommand::ProjectorDownload) {
        return projector_download::run(args);
    }
    let report = match command {
        ModelsCommand::Generate => generate::run(Path::new(&root()?), args),
        ModelsCommand::Resolve => resolve::run(args),
        ModelsCommand::RestoreInputs => restore_inputs::run(args),
        ModelsCommand::ParityDownload => parity_download::run(args),
        ModelsCommand::ProjectorDownload => unreachable!("projector dispatch returned"),
    };
    report.emit()
}

pub(crate) fn acquire_pinned_projector(
    input: &str,
    output: &std::path::Path,
    expected: &str,
    maximum: u64,
    deadline: std::time::Instant,
    cancellation: &crate::process::Cancellation,
) -> DynResult<String> {
    projector_download::acquire_pinned(input, output, expected, maximum, deadline, cancellation)
}

pub(crate) fn validate_projector_origin(input: &str) -> DynResult<()> {
    projector_download::validate_origin(input)
}
