//! `prepared-input`: consumers that bind prepared CI inputs to their exact
//! identity and bytes, replacing the checks in `scripts/ui-distribution.py`,
//! `scripts/verify-static-abi-build-stamp.py`, the static ABI input manifest
//! programs, the native SDK manifest consumers and the SDK runtime-list JSON
//! reader in `scripts/ci-prepare-native-runtime.sh`.
//!
//! Every command only reads and compares. None builds, repairs or replaces
//! an input: UI builds and SDK compilation stay with their owning tools.
//! Paths resolve against the working directory, like the scripts, so the
//! commands work in fixture directories outside a checkout.

mod abi_manifest;
mod abi_stamp;
mod html_modules;
mod python_io;
pub(crate) mod python_json;
pub(crate) mod python_value;
mod runtime_select;
mod sdk_artifact_file;
mod sdk_identity;
mod sdk_identity_contract;
mod sdk_manifest;
mod ui_distribution;

use crate::command::DynResult;
use crate::repository::check_report::CheckReport;

const USAGE: &str = "xtask prepared-input {ui-distribution {stamp|verify} --dist DIST --source-sha SHA --release-tag TAG | static-abi-stamp STAMP --backend B --link-mode M --stamp-version V --toolchain-epoch E [--patched-sha S] | static-abi-manifest describe MANIFEST TARGET BACKEND BUILD_DIR EPOCH PATCHED_SHA_FILE STAMP | static-abi-manifest verify MANIFEST STAMP TARGET BACKEND BUILD_DIR EPOCH | native-sdk-manifest ARTIFACT_DIR MANIFEST | native-sdk-identity MANIFEST TARGET BACKEND PROFILE | native-sdk-library-dir MANIFEST | sdk-runtime-select RUNTIME_ROOT BACKEND REPORT SKIPPY_ABI}";

/// A rejected input: the legacy `SystemExit(message)` shape, status 1.
pub(crate) struct Rejected(pub(crate) String);

pub(crate) type Checked<T> = Result<T, Rejected>;

impl From<String> for Rejected {
    fn from(message: String) -> Self {
        Self(message)
    }
}

impl From<&str> for Rejected {
    fn from(message: &str) -> Self {
        Self(message.to_owned())
    }
}

/// `Ok(stdout)` or a rejection printed on stderr with status 1.
fn report(outcome: Checked<String>) -> CheckReport {
    match outcome {
        Ok(stdout) => CheckReport::success(stdout),
        Err(Rejected(message)) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

/// Positional-only commands keep the legacy unpacking contract: a wrong
/// argument count fails with status 1, like the inline programs.
fn positional<'a, const N: usize>(args: &'a [String], names: &str) -> Checked<[&'a str; N]> {
    let values: Vec<&str> = args.iter().map(String::as_str).collect();
    values
        .try_into()
        .map_err(|_| Rejected(format!("expected {N} arguments: {names}")))
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let outcome = match args {
        [command, rest @ ..] if command == "ui-distribution" => ui_distribution::run(rest),
        [command, rest @ ..] if command == "static-abi-stamp" => abi_stamp::run(rest),
        [command, mode, rest @ ..] if command == "static-abi-manifest" && mode == "describe" => {
            report(abi_manifest::describe(rest))
        }
        [command, mode, rest @ ..] if command == "static-abi-manifest" && mode == "verify" => {
            report(abi_manifest::verify(rest))
        }
        [command, rest @ ..] if command == "native-sdk-manifest" => report(sdk_manifest::run(rest)),
        [command, rest @ ..] if command == "native-sdk-identity" => {
            report(sdk_identity::identity(rest))
        }
        [command, rest @ ..] if command == "native-sdk-library-dir" => {
            report(sdk_identity::library_dir(rest))
        }
        [command, rest @ ..] if command == "sdk-runtime-select" => {
            report(runtime_select::run(rest))
        }
        _ => CheckReport::usage(USAGE, "unknown prepared-input command"),
    };
    outcome.emit()
}
