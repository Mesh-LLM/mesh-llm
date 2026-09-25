//! argparse emulation for `scripts/collect-ci-metrics.py`: every option of
//! `parse_args` (string, `int` and `append` options), unique-prefix
//! abbreviations with ambiguity errors, `=value`, `-h`, `--` handling,
//! Python 3.13's sequential error order, then the script's own
//! `parser.error` rules in source order.

use crate::ci_operations::build_cache_options::{
    Kind, ambiguous, classify, help_flag, is_option_like,
};
use crate::ci_operations::ci_metrics_int::python_int;
use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;

const PROG: &str = "collect-ci-metrics.py";
const USAGE: &str = "\
usage: collect-ci-metrics.py [-h] [--repo REPO] [--workflow WORKFLOW]
                             [--run-id RUN_ID] [--input INPUT]
                             [--compare-input COMPARE_INPUT] [--limit LIMIT]
                             [--status STATUS] [--branch BRANCH]
                             [--event EVENT] [--created CREATED] [--top TOP]
                             [--label KEY=VALUE] [--json-out JSON_OUT]
                             [--markdown-out MARKDOWN_OUT] [--raw-out RAW_OUT]
";
const HELP: &str = "
Collect read-only GitHub Actions timing and runner metrics.

options:
  -h, --help            show this help message and exit
  --repo REPO
  --workflow WORKFLOW
  --run-id RUN_ID
  --input INPUT         Detailed run JSON, --raw-out JSON, or -
  --compare-input COMPARE_INPUT
                        Detailed/raw run JSON for a historical baseline cohort
  --limit LIMIT
  --status STATUS
  --branch BRANCH
  --event EVENT
  --created CREATED     GitHub date filter, e.g. >=2026-07-01
  --top TOP
  --label KEY=VALUE
  --json-out JSON_OUT
  --markdown-out MARKDOWN_OUT
  --raw-out RAW_OUT     Save detailed inputs for offline analysis
";
const OPTIONS: [&str; 17] = [
    "-h",
    "--help",
    "--repo",
    "--workflow",
    "--run-id",
    "--input",
    "--compare-input",
    "--limit",
    "--status",
    "--branch",
    "--event",
    "--created",
    "--top",
    "--label",
    "--json-out",
    "--markdown-out",
    "--raw-out",
];

/// The parsed namespace; unset string options are `None` like argparse.
#[derive(Debug)]
pub(crate) struct Args {
    pub(crate) repo: String,
    pub(crate) workflow: Option<String>,
    pub(crate) run_id: Vec<i64>,
    pub(crate) input: Option<String>,
    pub(crate) compare_input: Option<String>,
    pub(crate) limit: i64,
    pub(crate) status: String,
    pub(crate) branch: Option<String>,
    pub(crate) event: Option<String>,
    pub(crate) created: Option<String>,
    pub(crate) top: i64,
    pub(crate) label: Vec<String>,
    pub(crate) json_out: Option<String>,
    pub(crate) markdown_out: Option<String>,
    pub(crate) raw_out: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            repo: "Mesh-LLM/mesh-llm".to_owned(),
            workflow: None,
            run_id: Vec::new(),
            input: None,
            compare_input: None,
            limit: 20,
            status: "success".to_owned(),
            branch: None,
            event: None,
            created: None,
            top: 10,
            label: Vec::new(),
            json_out: None,
            markdown_out: None,
            raw_out: None,
        }
    }
}

fn fail(message: &str) -> CheckReport {
    error(USAGE, PROG, message)
}

/// Parses `args`, or returns the help/usage report argparse would produce.
pub(crate) fn parse(args: &[String]) -> Result<Args, CheckReport> {
    let mut parsed = Args::default();
    let mut extras: Vec<String> = Vec::new();
    let mut positional_only = false;
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        index += 1;
        if positional_only || arg == "--" {
            positional_only = true;
            extras.push(arg.clone());
            continue;
        }
        match classify(arg, &OPTIONS) {
            Kind::Positional | Kind::Unknown => extras.push(arg.clone()),
            Kind::Ambiguous(names) => return Err(ambiguous(arg, &names, &fail)),
            Kind::Known("-h" | "--help", explicit, sep) => {
                help_flag(explicit, sep, "-h/--help", &fail)?;
                return Err(CheckReport::success(format!("{USAGE}{HELP}")));
            }
            Kind::Known(option, explicit, _) => {
                let value = take_value(explicit, args, &mut index, option)?;
                apply(&mut parsed, option, value)?;
            }
        }
    }
    if !extras.is_empty() {
        return Err(fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    validate(&parsed)?;
    Ok(parsed)
}

fn take_value(
    explicit: Option<String>,
    args: &[String],
    index: &mut usize,
    option: &str,
) -> Result<String, CheckReport> {
    if let Some(value) = explicit {
        return Ok(value);
    }
    match args.get(*index) {
        Some(next) if !is_option_like(next, &OPTIONS) => {
            *index += 1;
            Ok(next.clone())
        }
        _ => Err(fail(&format!("argument {option}: expected one argument"))),
    }
}

fn integer(option: &str, value: &str) -> Result<i64, CheckReport> {
    python_int(value).ok_or_else(|| {
        fail(&format!(
            "argument {option}: invalid int value: {}",
            repr(value)
        ))
    })
}

fn apply(parsed: &mut Args, option: &str, value: String) -> Result<(), CheckReport> {
    match option {
        "--repo" => parsed.repo = value,
        "--workflow" => parsed.workflow = Some(value),
        "--run-id" => parsed.run_id.push(integer(option, &value)?),
        "--input" => parsed.input = Some(value),
        "--compare-input" => parsed.compare_input = Some(value),
        "--limit" => parsed.limit = integer(option, &value)?,
        "--status" => parsed.status = value,
        "--branch" => parsed.branch = Some(value),
        "--event" => parsed.event = Some(value),
        "--created" => parsed.created = Some(value),
        "--top" => parsed.top = integer(option, &value)?,
        "--label" => parsed.label.push(value),
        "--json-out" => parsed.json_out = Some(value),
        "--markdown-out" => parsed.markdown_out = Some(value),
        _ => parsed.raw_out = Some(value),
    }
    Ok(())
}

/// The script's `parser.error` rules; empty strings are falsy in Python.
fn validate(args: &Args) -> Result<(), CheckReport> {
    let truthy = |value: &Option<String>| value.as_deref().is_some_and(|text| !text.is_empty());
    let input = truthy(&args.input);
    let workflow = truthy(&args.workflow);
    let run_id = !args.run_id.is_empty();
    if input && (workflow || run_id) {
        return Err(fail(
            "--input cannot be combined with --workflow or --run-id",
        ));
    }
    if workflow && run_id {
        return Err(fail("--workflow cannot be combined with --run-id"));
    }
    if !input && !workflow && !run_id {
        return Err(fail("one of --input, --workflow, or --run-id is required"));
    }
    if args.limit < 1 || args.top < 1 {
        return Err(fail("--limit and --top must be at least 1"));
    }
    let stdout_outputs = [&args.json_out, &args.markdown_out, &args.raw_out]
        .into_iter()
        .filter(|path| path.as_deref() == Some("-"))
        .count();
    if stdout_outputs > 1 {
        return Err(fail("only one output may use stdout (-)"));
    }
    Ok(())
}
