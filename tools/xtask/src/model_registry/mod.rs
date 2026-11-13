//! `models`: the test-model artifact registry. `generate` projects
//! `ci/model-artifacts/registry.json` into suite manifests and the family
//! roster; `resolve` and `restore-inputs` hand consumers exactly one
//! cadence-authorized, integrity-pinned artifact. Their established consumer
//! formats are retained. `parity-download` owns manual discovery and verified
//! candidate materialization without inline Python.

mod certification;
mod dflash_draft;
mod family_roster;
mod generate;
pub(crate) mod parity_download;
mod projection;
mod projector_download;
mod registry;

// Transitional: `resolve` and `restore-inputs` and the modules they share live in
// authority-automation; the remaining `models` commands import them from here.
pub(crate) use authority_automation::model_registry::{
    argv, fields, json_bytes, manifest, resolve, restore_inputs, serving_entry,
};

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
    DflashDraftConvert,
}

impl ModelsCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "generate" => Some(Self::Generate),
            "resolve" => Some(Self::Resolve),
            "restore-inputs" => Some(Self::RestoreInputs),
            "projector-download" => Some(Self::ProjectorDownload),
            "parity-download" => Some(Self::ParityDownload),
            "dflash-draft-convert" => Some(Self::DflashDraftConvert),
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
    if matches!(command, ModelsCommand::DflashDraftConvert) {
        return dflash_draft::run(args);
    }
    let report = match command {
        ModelsCommand::Generate => generate::run(Path::new(&root()?), args),
        ModelsCommand::Resolve => resolve::run(args),
        ModelsCommand::RestoreInputs => restore_inputs::run(args),
        ModelsCommand::ParityDownload => parity_download::run(args),
        ModelsCommand::ProjectorDownload | ModelsCommand::DflashDraftConvert => {
            unreachable!("direct dispatch returned")
        }
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
