//! `qualify`: standalone Skippy product qualification. These verbs are the
//! Rust owners of the legacy `skippy/scripts/ci-qualification-availability.py`,
//! `verify-cli-input.py`, `compose-ci-product.py`, `qualify-ci-packaging.py`,
//! `ci-standalone-model-qualification.py`, `ci-standalone-laya-qualification.py`,
//! `assemble-ci-qualification.py`, `validate-ci-qualification.py`, and
//! `scripts/ci-host-qualification-receipts.py`. Each composes, qualifies, or
//! validates the exact bytes CI produced for one core runtime row; every check
//! fails closed.

mod assemble;
mod availability;
mod cli_input;
mod compose;
pub(crate) mod contract;
pub(crate) mod host_receipts;
mod http;
mod laya;
mod models;
mod packaging;
mod serve;
mod validate;

use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};

/// A `qualify` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum QualifyCommand {
    Availability,
    VerifyCliInput,
    Compose,
    Packaging,
    Models,
    Laya,
    Assemble,
    Validate,
    HostReceipts,
}

impl QualifyCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "availability" => Some(Self::Availability),
            "verify-cli-input" => Some(Self::VerifyCliInput),
            "compose" => Some(Self::Compose),
            "packaging" => Some(Self::Packaging),
            "models" => Some(Self::Models),
            "laya" => Some(Self::Laya),
            "assemble" => Some(Self::Assemble),
            "validate" => Some(Self::Validate),
            "host-receipts" => Some(Self::HostReceipts),
            _ => None,
        }
    }
}

pub(crate) const USAGE: &str = "usage:\n  cargo xtool qualify availability --source-sha <sha> --plan-digest <digest> --row-id <row> --output <file>\n  cargo xtool qualify verify-cli-input <directory> --source-sha <sha>\n  cargo xtool qualify compose --cli-input <dir> --runtime-input <dir> --output <dir> --source-sha <sha> --target <target> --backend <backend>\n  cargo xtool qualify packaging --product-dir <dir> --product-archive <file> --row-id <row> --source-sha <sha> --evidence <file>\n  cargo xtool qualify models --product-dir <dir> --row-id <row> --suite <dense|recurrent|moe|kv-cache> [--device <dev>] [--ctx-size 2048] --model <artifact-id> <path> <sha256> <model-id> --evidence <file> [--availability <file> --hardware-evidence <file>]\n  cargo xtool qualify laya --product-dir <dir> --model <path> --model-sha256 <sha256> --model-id <id> --row-id <row> [--device <dev>] --evidence-dir <dir>\n  cargo xtool qualify assemble --product-dir <dir> --availability <file> --evidence-dir <dir> [--hardware <file>] --source-sha <sha> --plan-digest <digest> --row-id <row> --output <file>\n  cargo xtool qualify validate --receipt <file> --product-manifest <file> --product-dir <dir> --availability <file> --evidence-dir <dir> --source-sha <sha> --plan-digest <digest> --row-id <row>\n  cargo xtool qualify host-receipts --receipts <dir> --source-sha <sha> --plan-digest <digest> --platform <linux|macos|windows> --output <file>";

/// `root` resolves lazily: only verbs that consult the checked-in row catalog
/// or model registry need the checkout.
pub(crate) fn run(
    command: QualifyCommand,
    args: &[String],
    root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let needs_root = matches!(
        command,
        QualifyCommand::Availability
            | QualifyCommand::Compose
            | QualifyCommand::Packaging
            | QualifyCommand::Models
            | QualifyCommand::Laya
            | QualifyCommand::Assemble
            | QualifyCommand::Validate
    );
    let report = if needs_root {
        let root = root()?;
        dispatch_with_root(command, args, &root)
    } else {
        match command {
            QualifyCommand::VerifyCliInput => cli_input::run(args),
            QualifyCommand::HostReceipts => host_receipts::run(args),
            _ => unreachable!("rootless dispatch"),
        }
    };
    report.emit()
}

fn dispatch_with_root(command: QualifyCommand, args: &[String], root: &Path) -> CheckReport {
    match command {
        QualifyCommand::Availability => availability::run(root, args),
        QualifyCommand::Compose => compose::run(root, args),
        QualifyCommand::Packaging => packaging::run(root, args),
        QualifyCommand::Models => models::run(root, args),
        QualifyCommand::Laya => laya::run(root, args),
        QualifyCommand::Assemble => assemble::run(root, args),
        QualifyCommand::Validate => validate::run(root, args),
        _ => unreachable!("rootless dispatch"),
    }
}
