//! `product`: immutable product composition. `compose` is the Rust owner of
//! `scripts/compose-product-bundle.py`: it verifies one backend-neutral host
//! and one native runtime against the requested version and backend, then
//! writes (or with `--check`, verifies) `product-manifest.json` with the
//! exact host and runtime digests. It never builds, copies or substitutes an
//! input; a missing or mismatched input fails.
//!
//! The packaging callers' inline Python is owned here too:
//! `runtime-release-manifest` (from
//! `scripts/generate-native-runtime-release-manifest.sh`), and
//! `canonical-inputs` and `runtime-version` (from
//! `scripts/ci-compose-product-input.sh`).

mod archive;
mod archive_tar;
mod archive_zip;
mod attestation_status;
mod canonical_inputs;
mod compose;
mod host_contract;
mod compose_argv;
pub(crate) mod digest;
mod manifest_load;
mod posix_path;
mod pure_path;
mod python_object;
mod rc_ok;
mod release_manifest;
mod release_manifest_order;
mod runtime_version;

use crate::command::DynResult;

/// A `product` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum ProductCommand {
    Compose,
    CanonicalInputs,
    RuntimeReleaseManifest,
    RuntimeVersion,
    ArchivePlan,
    ArchiveWrite,
    AttestationStatus,
    RcOk,
}

impl ProductCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "compose" => Some(Self::Compose),
            "canonical-inputs" => Some(Self::CanonicalInputs),
            "runtime-release-manifest" => Some(Self::RuntimeReleaseManifest),
            "runtime-version" => Some(Self::RuntimeVersion),
            "archive-plan" => Some(Self::ArchivePlan),
            "archive-write" => Some(Self::ArchiveWrite),
            "attestation-status" => Some(Self::AttestationStatus),
            "rc-ok" => Some(Self::RcOk),
            _ => None,
        }
    }
}

pub(crate) fn run(command: ProductCommand, args: &[String]) -> DynResult<()> {
    let report = match command {
        ProductCommand::Compose => compose::run(args),
        ProductCommand::CanonicalInputs => canonical_inputs::run(args),
        ProductCommand::RuntimeReleaseManifest => release_manifest::run(args),
        ProductCommand::RuntimeVersion => runtime_version::run(args),
        ProductCommand::ArchivePlan => archive::run(args, false),
        ProductCommand::ArchiveWrite => archive::run(args, true),
        ProductCommand::AttestationStatus => attestation_status::run(args),
        ProductCommand::RcOk => rc_ok::run(args),
    };
    report.emit()
}
