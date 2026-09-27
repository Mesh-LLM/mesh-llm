//! `artifact`: release-artifact integrity commands. `verify-checksum` is the
//! Rust owner of `scripts/verify-checksum-sidecar.py`; archive extraction
//! is `extract-tar` and `extract-zip`, the owners of
//! `scripts/safe-extract-tar.py` and `scripts/safe-extract-zip.py`. Every command takes explicit
//! paths relative to the working directory and needs no repository checkout.

mod argv;
mod checksum;
pub(crate) mod tar_extract;
mod tar_header;
mod tar_read;
mod zip_directory;
pub(crate) mod zip_extract;
mod zip_read;
mod zip_text;

pub(crate) use zip_extract::os_error_line;

use crate::command::DynResult;

/// An `artifact` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum ArtifactCommand {
    VerifyChecksum,
    ExtractTar,
    ExtractZip,
}

impl ArtifactCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "verify-checksum" => Some(Self::VerifyChecksum),
            "extract-tar" => Some(Self::ExtractTar),
            "extract-zip" => Some(Self::ExtractZip),
            _ => None,
        }
    }
}

pub(crate) fn run(command: ArtifactCommand, args: &[String]) -> DynResult<()> {
    check(command, args).emit()
}

pub(crate) fn check(
    command: ArtifactCommand,
    args: &[String],
) -> crate::repository::check_report::CheckReport {
    match command {
        ArtifactCommand::VerifyChecksum => checksum::run(args),
        ArtifactCommand::ExtractTar => tar_extract::run(args),
        ArtifactCommand::ExtractZip => zip_extract::run(args),
    }
}
