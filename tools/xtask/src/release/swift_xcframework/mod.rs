mod arguments;
mod contract;
mod error;
mod input;
mod layout;
mod lipo;

#[cfg(all(test, unix))]
use crate::process::Cancellation;
use crate::{command_interrupt::Interrupt, repository::check_report::CheckReport};
pub(crate) use contract::Mode;
pub(crate) use error::Error;
pub(crate) use lipo::NativeLipo;
use std::{collections::BTreeSet, path::Path};

#[cfg(all(test, unix))]
mod command_tests;
#[cfg(test)]
mod contract_tests;
#[cfg(test)]
mod fixtures;
#[cfg(all(test, unix))]
mod layout_tests;
#[cfg(all(test, unix))]
mod native_tests;

pub(crate) fn run(args: &[String]) -> CheckReport {
    let options = match arguments::parse(args) {
        Ok(options) => options,
        Err(message) => return CheckReport::usage(arguments::USAGE, &message),
    };
    let result = execute(Path::new(&options.root), options.mode, || options.native());
    match result {
        Ok(count) => CheckReport::success(format!(
            "verified {count} XCFramework slice(s){}\n",
            options
                .mode
                .map_or(String::new(), |mode| format!(" for {} mode", mode.name()))
        )),
        Err(error) => CheckReport::failure(String::new(), format!("error: {error}\n")),
    }
}

pub(crate) fn emit(report: CheckReport) -> crate::command::DynResult<()> {
    use std::io::Write;
    let mut stdout = crate::cli_output::stdout();
    stdout.write_all(report.stdout.as_bytes())?;
    stdout.flush()?;
    let mut stderr = crate::cli_output::stderr();
    stderr.write_all(report.stderr.as_bytes())?;
    stderr.flush()?;
    if report.code != 0 {
        std::process::exit(report.code);
    }
    Ok(())
}

fn execute(
    root: &Path,
    mode: Option<Mode>,
    native: impl FnOnce() -> Result<NativeLipo, Error>,
) -> Result<usize, Error> {
    let document = input::Document::read(root)?;
    let entries = document.entries(mode)?;
    let mut scope: Option<Interrupt> = None;
    let mut adapter = None;
    let mut create = Some(native);
    let result = verify(
        Verification {
            entries: &entries,
            root,
            mode,
        },
        |binary| {
            if scope.is_none() {
                scope = Some(Interrupt::install()?);
                let make = create
                    .take()
                    .ok_or_else(|| Error::Contract("native adapter already initialized".into()))?;
                adapter = Some(make()?);
            }
            let interrupt = scope
                .as_ref()
                .ok_or_else(|| Error::Contract("missing interruption scope".into()))?;
            interrupt.check()?;
            let native = adapter
                .as_ref()
                .ok_or_else(|| Error::Contract("missing native adapter".into()))?;
            let result = native.inspect(binary, &interrupt.cancellation())?;
            interrupt.check()?;
            Ok(result)
        },
    );
    match scope {
        Some(interrupt) => error::finalize(result, interrupt.finish()),
        None => result,
    }
}

#[cfg(all(test, unix))]
pub(crate) fn verify_with_native(
    root: &Path,
    mode: Option<Mode>,
    native: &NativeLipo,
    cancellation: &Cancellation,
) -> Result<usize, Error> {
    let document = input::Document::read(root)?;
    let entries = document.entries(mode)?;
    verify(
        Verification {
            entries: &entries,
            root,
            mode,
        },
        |binary| native.inspect(binary, cancellation),
    )
}

struct Verification<'a> {
    entries: &'a [input::Entry<'a>],
    root: &'a Path,
    mode: Option<Mode>,
}

fn verify(
    request: Verification<'_>,
    mut inspect: impl FnMut(&Path) -> Result<BTreeSet<String>, Error>,
) -> Result<usize, Error> {
    for entry in request.entries {
        let declared = entry.architectures(request.mode)?;
        let framework = layout::framework(request.root, &entry.location()?)?;
        if entry.key.is_macos() {
            layout::macos(&framework)?;
        }
        let actual = inspect(&layout::binary(&framework)?)?;
        if actual != declared {
            return Err(Error::Contract(format!(
                "XCFramework slice {:?} lipo architectures {actual:?} do not match SupportedArchitectures {declared:?}",
                entry.key
            )));
        }
    }
    Ok(request.entries.len())
}
