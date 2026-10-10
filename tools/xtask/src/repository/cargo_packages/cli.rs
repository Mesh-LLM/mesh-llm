use super::{Generation, TranslationRequest, metadata};
use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(crate) const USAGE: &str = "repository cargo-packages --generation {legacy|current} --crates <JSON> [--batches <JSON>] {--cargo <absolute-executable> [--timeout <seconds>] | --metadata <fixture-path>}";

pub(crate) fn run(cwd: &Path, args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        return CheckReport::success(format!("usage: {USAGE}\nReal mode runs metadata --locked --no-deps --format-version=1 in the selected --repo-root or invocation cwd. --cargo must be absolute; timeout defaults to 120 and must be 1..=86400 seconds. --metadata is explicit fixture mode and cannot be combined with --cargo or --timeout.\n")).emit();
    }
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(error) => return CheckReport::usage(USAGE, &error.to_string()).emit(),
    };
    match execute(cwd, options) {
        Ok(output) => CheckReport::success(output).emit(),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("CI package resolution failed: {error}\n"),
        )
        .emit(),
    }
}

struct Options {
    request: TranslationRequest,
    source: metadata::Source,
}

impl Options {
    fn parse(args: &[String]) -> DynResult<Self> {
        let mut generation = None;
        let mut crates = None;
        let mut batches = None;
        let mut metadata = None;
        let mut cargo = None;
        let mut timeout = None;
        let mut arguments = args.iter();
        while let Some(flag) = arguments.next() {
            let value = arguments.next().ok_or("missing option value")?;
            let slot = match flag.as_str() {
                "--generation" => &mut generation,
                "--crates" => &mut crates,
                "--batches" => &mut batches,
                "--metadata" => &mut metadata,
                "--cargo" => &mut cargo,
                "--timeout" => &mut timeout,
                _ => return Err("unknown cargo-packages option".into()),
            };
            if slot.replace(value.as_str()).is_some() {
                return Err("duplicate cargo-packages option".into());
            }
        }
        let generation = Generation::parse(generation.ok_or("missing --generation")?)?;
        let request =
            TranslationRequest::parse(crates.ok_or("missing --crates")?, batches, generation)?;
        let source = match (metadata, cargo, timeout) {
            (Some(path), None, None) => metadata::Source::File(path.into()),
            (None, Some(executable), timeout) => {
                let executable = PathBuf::from(executable);
                if !executable.is_absolute() {
                    return Err("--cargo must be an absolute executable path".into());
                }
                let seconds = timeout.unwrap_or("120").parse::<u64>()?;
                if !(1..=86400).contains(&seconds) {
                    return Err("--timeout must be 1..=86400 seconds".into());
                }
                metadata::Source::Cargo {
                    executable,
                    timeout: Duration::from_secs(seconds),
                }
            }
            (None, None, _) => {
                return Err("real mode requires --cargo; fixture mode requires --metadata".into());
            }
            (Some(_), _, _) => {
                return Err("--metadata cannot be combined with --cargo or --timeout".into());
            }
        };
        Ok(Self { request, source })
    }
}

fn execute(cwd: &Path, options: Options) -> DynResult<String> {
    let available = metadata::discover(cwd, options.source)?;
    Ok(options.request.resolve(&available)?.python_json_line())
}
