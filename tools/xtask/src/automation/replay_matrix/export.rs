use super::{input, serialization, with_worker};
use crate::command::DynResult;
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: crate::automation::REPLAY_EXPORT_USAGE,
    values: &["--matrix", "--json-output", "--github-env"],
    flags: &["--help", "--print-shell"],
};

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let report = match GRAMMAR.parse(args) {
        Err(report) => report,
        Ok(parsed) if parsed.flag("--help") => CheckReport::success(format!(
            "usage: {}\n\nValidate the complete matrix before exporting replay parameters.\nPaths are relative to the invocation directory; '-' is a literal filename.\nNo parent directories are created. No replay is executed.\n\nOptions:\n  --matrix <path>       Required matrix input\n  --json-output <path>  Replace with sorted, ASCII-escaped replay JSON plus LF\n  --github-env <path>   Append twelve ordered environment fields after JSON\n  --print-shell        Print the shell line after requested files succeed\n  --help               Show this help\n\nGITHUB_ENV is not read. With no export options, validate silently.\n",
            GRAMMAR.usage
        )),
        Ok(parsed) if !parsed.positionals.is_empty() => GRAMMAR.error(&format!(
            "unrecognized arguments: {}",
            parsed.positionals.join(" ")
        )),
        Ok(parsed) => match parsed.last("--matrix") {
            None => GRAMMAR.error("the following arguments are required: --matrix"),
            Some(path) => with_worker(|| export(Path::new(path), &parsed))?,
        },
    };
    report.emit()
}

fn export(path: &Path, options: &ParsedArgs) -> CheckReport {
    let loaded = match input::load(path) {
        Ok(loaded) => loaded,
        Err(error) => return CheckReport::failure(String::new(), format!("{error}\n")),
    };
    export_loaded(&loaded, options, true)
}

pub(super) fn export_loaded(
    loaded: &input::LoadedReplay,
    options: &ParsedArgs,
    include_shell: bool,
) -> CheckReport {
    if let Some(path) = options.last("--json-output") {
        let json = serialization::render(&loaded.replay);
        if let Err(error) = std::fs::write(path, json) {
            return io_failure("JSON", path, error);
        }
    }
    if let Some(path) = options.last("--github-env") {
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut file| file.write_all(loaded.parameters.github_env().as_bytes()));
        if let Err(error) = result {
            return io_failure("GitHub env", path, error);
        }
    }
    CheckReport::success(if include_shell && options.flag("--print-shell") {
        loaded.parameters.shell_line()
    } else {
        String::new()
    })
}

fn io_failure(kind: &str, path: &str, error: std::io::Error) -> CheckReport {
    CheckReport::failure(
        String::new(),
        format!("replay matrix export: {kind} {path}: {error}\n"),
    )
}
