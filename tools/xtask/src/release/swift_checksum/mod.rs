mod arguments;
mod capture;
#[cfg(test)]
mod command_tests;
mod error;
#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod support;
#[cfg(test)]
mod tests;

use crate::automation::command_interrupt::Interrupt;
use crate::repository::check_report::CheckReport;
use error::Error;

const USAGE: &str = "release swift-manifest {update|verify} <tag> <artifact> <Package.swift> <absolute-swift> <timeout-seconds:1..3600> <max-output-bytes:1..1048576>";

pub(crate) fn run(args: &[String]) -> CheckReport {
    let options = match arguments::parse(args) {
        Ok(options) => options,
        Err(message) => return CheckReport::usage(USAGE, message),
    };
    if !cfg!(target_os = "macos") {
        let operation = match options.operation {
            arguments::Operation::Update => "updates",
            arguments::Operation::Verify => "verification",
        };
        return CheckReport::failure(
            String::new(),
            format!("error: Swift package manifest {operation} must run on macOS\n"),
        );
    }
    match execute(&options) {
        Ok(report) => report,
        Err(error) => error.report(),
    }
}

fn execute(options: &arguments::Options<'_>) -> Result<CheckReport, Error> {
    if !std::path::Path::new(options.artifact).is_file() {
        return Err(Error::Artifact(options.artifact.to_owned()));
    }
    if !std::path::Path::new(options.manifest).is_file() {
        return Err(Error::Manifest(options.manifest.to_owned()));
    }
    let interrupt = Interrupt::install()?;
    let result = (|| {
        interrupt.check()?;
        let checksum = capture::compute(&options.native()?, &interrupt.cancellation())?;
        interrupt.check()?;
        Ok(checksum)
    })();
    let checksum = finalize(result, interrupt.finish())?;
    Ok(super::swift_manifest::run(&[
        options.operation.name().to_owned(),
        options.tag.to_owned(),
        options.artifact.to_owned(),
        options.manifest.to_owned(),
        checksum,
    ]))
}

fn finalize(
    result: Result<String, Error>,
    finish: Result<(), crate::automation::command_interrupt::Reason>,
) -> Result<String, Error> {
    match (result, finish) {
        (result, Ok(())) => result,
        (Ok(_), Err(reason)) => Err(Error::Interrupt(reason)),
        (Err(primary), Err(reason)) => Err(Error::Finalization {
            primary: Box::new(primary),
            reason,
        }),
    }
}
