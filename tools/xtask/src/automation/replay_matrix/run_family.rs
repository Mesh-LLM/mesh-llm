use super::export;
use super::input;
use super::with_worker;
use crate::command::DynResult;
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use self::run_family_process::InvocationContext;
use self::run_family_report::RunFamilyReport;

#[path = "run_family_invocation.rs"]
mod run_family_invocation;
#[path = "run_family_process.rs"]
mod run_family_process;
#[path = "run_family_process_failure.rs"]
mod run_family_process_failure;
#[path = "run_family_report.rs"]
mod run_family_report;
#[path = "run_family_spec.rs"]
mod run_family_spec;
const GRAMMAR: Grammar = Grammar {
    usage: crate::automation::REPLAY_RUN_FAMILY_USAGE,
    values: &[
        "--matrix",
        "--run-family",
        "--ref",
        "--dataset-file",
        "--output",
        "--python",
        "--json-output",
        "--github-env",
        "--timeout",
    ],
    flags: &["--print-shell", "--help"],
};

struct Options {
    parsed: ParsedArgs,
    matrix: PathBuf,
    family: Option<String>,
    refs: Vec<OsString>,
    dataset: Option<PathBuf>,
    output: Option<PathBuf>,
    python: Option<PathBuf>,
    timeout: Option<Duration>,
}

pub(in crate::automation) fn run(args: &[String], explicit_root: Option<&Path>) -> DynResult<()> {
    let result = match GRAMMAR.parse(args) {
        Err(report) => RunFamilyReport::from_check(report),
        Ok(parsed) if parsed.flag("--help") => {
            RunFamilyReport::from_check(CheckReport::success(format!(
                "usage: {}\n\nExport replay parameters, select one pinned family, then supervise one replay child. The replay child is never launched without --python; fixture tests use only an isolated fixture executable. Paths are relative to the invocation directory.\n\nOptions:\n  --matrix <path>       Required complete replay matrix\n  --run-family <name>   Required unique model family\n  --ref <label=ref>     Ordered repeatable replay ref\n  --dataset-file <path> Required replay dataset path\n  --output <path>       Required replay output directory\n  --python <path>       Absolute child interpreter executable\n  --json-output <path> Export sorted replay JSON before selection\n  --github-env <path>  Append replay environment before selection\n  --print-shell         Print shell line only after child success\n  --timeout <seconds>  Child deadline, 1..=86400 (default 3600)\n  --help                Show this help\n",
                GRAMMAR.usage
            )))
        }
        Ok(parsed) if !parsed.positionals.is_empty() => RunFamilyReport::from_check(GRAMMAR.error(
            &format!("unrecognized arguments: {}", parsed.positionals.join(" ")),
        )),
        Ok(parsed) => match options(parsed) {
            Ok(options) => with_worker(|| execute(&options, explicit_root))?,
            Err(error) => RunFamilyReport::from_check(error),
        },
    };
    result.emit()
}

fn options(parsed: ParsedArgs) -> Result<Options, CheckReport> {
    let matrix = match parsed.last("--matrix") {
        Some(path) => PathBuf::from(path),
        None => return Err(GRAMMAR.error("the following arguments are required: --matrix")),
    };
    let timeout = parsed
        .last("--timeout")
        .map(|text| match text.parse::<u64>() {
            Ok(seconds @ 1..=86400) => Ok(Duration::from_secs(seconds)),
            Ok(_) | Err(_) => Err(GRAMMAR.error("timeout must be an integer in 1..=86400")),
        })
        .transpose()?;
    let family = parsed.last("--run-family").map(str::to_owned);
    let refs = parsed
        .all("--ref")
        .into_iter()
        .map(OsString::from)
        .collect();
    let dataset = parsed.last("--dataset-file").map(PathBuf::from);
    let output = parsed.last("--output").map(PathBuf::from);
    let python = parsed.last("--python").map(PathBuf::from);
    Ok(Options {
        parsed,
        matrix,
        family,
        refs,
        dataset,
        output,
        python,
        timeout,
    })
}

fn execute(options: &Options, explicit_root: Option<&Path>) -> RunFamilyReport {
    let first = match input::load(&options.matrix) {
        Ok(loaded) => loaded,
        Err(error) => return RunFamilyReport::failure(format!("{error}\n")),
    };
    let exported = export::export_loaded(&first, &options.parsed, false);
    if exported.code != 0 {
        return RunFamilyReport::from_check(exported);
    }
    let family = match options.family.as_deref() {
        Some("") => {
            let stdout = if options.parsed.flag("--print-shell") {
                first.parameters.shell_line().into_bytes()
            } else {
                Vec::new()
            };
            return RunFamilyReport::success(stdout, Vec::new());
        }
        Some(family) => family,
        None => return usage_failure("--run-family is required"),
    };
    if options.refs.is_empty() {
        return usage_failure("--run-family needs --ref");
    }
    let dataset = match options.dataset.as_deref() {
        Some(path) => path,
        None => return usage_failure("--run-family needs --dataset-file"),
    };
    let output = match options.output.as_deref() {
        Some(path) => path,
        None => return usage_failure("--run-family needs --output"),
    };
    let python = match options.python.as_deref() {
        Some(path) => path,
        None => return usage_failure("--python is required for bounded child execution"),
    };
    let timeout = options.timeout.unwrap_or(Duration::from_secs(3600));
    let cwd = match std::env::current_dir() {
        Ok(cwd) => cwd,
        Err(error) => return RunFamilyReport::failure(format!("run-family cwd: {error}\n")),
    };
    let root = match explicit_root {
        Some(root) => root.to_path_buf(),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
    };
    let second = match input::load(&options.matrix) {
        Ok(loaded) => loaded,
        Err(error) => return RunFamilyReport::failure(format!("{error}\n")),
    };
    let matrix = match input::load_matrix(&options.matrix) {
        Ok(matrix) => matrix,
        Err(error) => return RunFamilyReport::failure(format!("{error}\n")),
    };
    run_family_process::run_selected(
        options,
        &first,
        &second.parameters,
        &matrix,
        InvocationContext {
            family,
            python,
            dataset,
            output,
            cwd: &cwd,
            repo_root: &root,
            timeout,
        },
    )
}

fn usage_failure(message: &str) -> RunFamilyReport {
    RunFamilyReport::from_check(GRAMMAR.error(message))
}
