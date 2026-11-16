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

/// Decode the single small file expected from a packaging-readiness artifact.
pub(crate) fn packaging_receipt_zip(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    const LIMIT: u64 = 1024 * 1024;
    if bytes.len() > LIMIT as usize {
        return Err("packaging receipt ZIP exceeds size limit".into());
    }
    let archive =
        zip_directory::open(bytes).map_err(|error| format!("invalid receipt ZIP: {error:?}"))?;
    if archive.infos.len() != 1 {
        return Err("packaging receipt ZIP must contain exactly one file".into());
    }
    let info = &archive.infos[0];
    let kind = (info.external >> 16) & 0o170000;
    if info.filename != "packaging-readiness.json"
        || info.orig_filename != "packaging-readiness.json"
        || info.file_size > LIMIT
        || info.is_dir()
        || !matches!(kind, 0 | 0o100000)
    {
        return Err("invalid packaging receipt ZIP member".into());
    }
    let range = archive
        .open_member(info)
        .map_err(|error| format!("invalid receipt ZIP: {error:?}"))?;
    archive
        .decode(info, range)
        .map_err(|error| format!("invalid receipt ZIP: {error:?}"))
}

#[cfg(test)]
mod packaging_receipt_zip_tests {
    use super::packaging_receipt_zip;

    #[test]
    fn admits_real_compressed_receipt_and_rejects_corrupt_or_oversized_zip() {
        let archive = include_bytes!("../../tests/fixtures/packaging-readiness.zip");
        let receipt: serde_json::Value =
            serde_json::from_slice(&packaging_receipt_zip(archive.to_vec()).unwrap()).unwrap();
        assert_eq!(receipt["schema"], "mesh-packaging-readiness-v1");
        assert!(packaging_receipt_zip(vec![0; 1024 * 1024 + 1]).is_err());
        assert!(packaging_receipt_zip(archive[..archive.len() / 2].to_vec()).is_err());
    }
}

pub(crate) use zip_extract::os_error_line;

use crate::command::DynResult;

/// An `artifact` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum ArtifactCommand {
    VerifyChecksum,
    ExtractTar,
    ExtractZip,
    FileProjection,
}

impl ArtifactCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "verify-checksum" => Some(Self::VerifyChecksum),
            "extract-tar" => Some(Self::ExtractTar),
            "extract-zip" => Some(Self::ExtractZip),
            "file-projection" => Some(Self::FileProjection),
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
        ArtifactCommand::FileProjection => authority_automation::file_projection::run(args),
    }
}
