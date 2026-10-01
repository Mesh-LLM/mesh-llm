pub(super) mod digest;
pub(super) mod export;
mod input;
mod integer;
pub(super) mod invocation;
mod parameters;
pub(super) mod pins;
pub(super) mod publication;
pub(super) mod run_family;
mod serialization;
mod value;

use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix validate --matrix <path>",
    values: &["--matrix"],
    flags: &["--help"],
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let report = match GRAMMAR.parse(args) {
        Err(report) => report,
        Ok(parsed) if parsed.flag("--help") => CheckReport::success(format!(
            "usage: {}\n\nValidate replay parameters and print one tab-delimited shell line.\nReads only --matrix, relative to the invocation directory. No replay is executed.\n\nOptions:\n  --matrix <path>  Required matrix input\n  --help           Show this help\n",
            GRAMMAR.usage
        )),
        Ok(parsed) if !parsed.positionals.is_empty() => GRAMMAR.error(&format!(
            "unrecognized arguments: {}",
            parsed.positionals.join(" ")
        )),
        Ok(parsed) => match parsed.last("--matrix") {
            None => GRAMMAR.error("the following arguments are required: --matrix"),
            Some(path) => validate(&PathBuf::from(path))?,
        },
    };
    report.emit()
}

fn validate(path: &Path) -> DynResult<CheckReport> {
    with_worker(|| match input::load(path) {
        Ok(loaded) => CheckReport::success(loaded.parameters.shell_line()),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    })
}

fn with_worker<T: Send>(operation: impl FnOnce() -> T + Send) -> DynResult<T> {
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("replay-matrix".into())
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(scope, operation)?;
        match worker.join() {
            Ok(report) => Ok(report),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}
