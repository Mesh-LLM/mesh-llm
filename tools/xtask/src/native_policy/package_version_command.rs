use super::package_version::{abi_version, read_source, workspace_version};
use crate::repository::check_report::CheckReport;
use std::path::Path;

const USAGE: &str = "xtask native package-source-version {workspace|abi} SOURCE (experimental)";

pub(crate) fn run(args: &[String]) -> CheckReport {
    let [kind, source] = args else {
        return CheckReport::usage(USAGE, "expected a version kind and source path");
    };
    let extract: fn(&[u8]) -> Result<String, super::package_version::VersionError> =
        match kind.as_str() {
            "workspace" => workspace_output,
            "abi" => abi_output,
            _ => return CheckReport::usage(USAGE, "expected workspace or abi"),
        };
    let outcome = read_source(Path::new(source)).and_then(|bytes| extract(&bytes));
    match outcome {
        Ok(version) => CheckReport::success(format!("{version}\n")),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}

fn workspace_output(bytes: &[u8]) -> Result<String, super::package_version::VersionError> {
    workspace_version(bytes).map(|version| version.to_string())
}

fn abi_output(bytes: &[u8]) -> Result<String, super::package_version::VersionError> {
    abi_version(bytes).map(|version| version.to_string())
}
