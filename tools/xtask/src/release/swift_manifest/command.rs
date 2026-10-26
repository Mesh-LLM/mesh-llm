use super::{ManifestError, ReleaseTag, ReleaseValues, SwiftChecksum, update_file, verify_file};
use crate::repository::check_report::CheckReport;
use std::path::Path;

const USAGE: &str = "release swift-manifest-text {update|verify} <tag> <artifact> <Package.swift> <previously-obtained-checksum>";

enum Operation {
    Update,
    Verify,
}

pub(crate) fn run(args: &[String]) -> CheckReport {
    let [operation, tag, artifact, manifest, checksum] = args else {
        return CheckReport::usage(USAGE, "expected five arguments");
    };
    let operation = match operation.as_str() {
        "update" => Operation::Update,
        "verify" => Operation::Verify,
        _ => return CheckReport::usage(USAGE, "expected update or verify"),
    };
    if !Path::new(artifact).is_file() {
        return CheckReport::failure(
            String::new(),
            format!("Swift release artifact does not exist: {artifact}\n"),
        );
    }
    if !Path::new(manifest).is_file() {
        return CheckReport::failure(
            String::new(),
            format!("Package.swift does not exist: {manifest}\n"),
        );
    }
    let values = ReleaseValues {
        tag: ReleaseTag(tag),
        checksum: SwiftChecksum(checksum),
    };
    let result = match operation {
        Operation::Update => update_file(Path::new(manifest), &values),
        Operation::Verify => verify_file(Path::new(manifest), &values),
    };
    match result {
        Ok(()) => CheckReport::success(match operation {
            Operation::Update => format!(
                "updated Swift package manifest for {tag}\n  url: {}\n  checksum: {checksum}\n",
                values.url()
            ),
            Operation::Verify => format!("verified Swift package manifest for {tag}\n"),
        }),
        Err(ManifestError::Missing(field)) => {
            CheckReport::failure(String::new(), format!("missing {field} in {manifest}\n"))
        }
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}
