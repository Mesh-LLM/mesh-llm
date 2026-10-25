//! Native option admission for read-only CI timing analysis.

use crate::repository::check_report::CheckReport;
use std::collections::BTreeSet;

const USAGE: &str = "cargo xtool ci-ops collect-metrics (--input <file|-> | --workflow <workflow> | --run-id <positive-id>...) [--repo <repo>] [--compare-input <file>] [--limit <positive-integer>] [--top <positive-integer>] [--status <status>] [--branch <branch>] [--event <event>] [--created <date-filter>] [--label KEY=VALUE]... [--json-out <file|->] [--markdown-out <file|->] [--raw-out <file|->]";

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
    CheckReport::usage(USAGE, message)
}

pub(crate) fn parse(arguments: &[String]) -> Result<Args, CheckReport> {
    if matches!(arguments, [flag] if flag == "--help" || flag == "-h") {
        return Err(CheckReport::success(format!("usage: {USAGE}\n")));
    }
    let mut parsed = Args::default();
    let mut seen = BTreeSet::new();
    let mut arguments = arguments.iter();
    while let Some(argument) = arguments.next() {
        let (option, inline) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(option, value)| {
                (option, Some(value))
            });
        if !matches!(
            option,
            "--repo"
                | "--workflow"
                | "--run-id"
                | "--input"
                | "--compare-input"
                | "--limit"
                | "--status"
                | "--branch"
                | "--event"
                | "--created"
                | "--top"
                | "--label"
                | "--json-out"
                | "--markdown-out"
                | "--raw-out"
        ) {
            return Err(fail(&format!(
                "unknown option or positional argument: {option}"
            )));
        }
        if !matches!(option, "--run-id" | "--label") && !seen.insert(option.to_owned()) {
            return Err(fail(&format!("{option} may be supplied only once")));
        }
        let value = inline
            .or_else(|| arguments.next().map(String::as_str))
            .filter(|value| !value.is_empty() && !value.starts_with("--"))
            .ok_or_else(|| fail(&format!("{option} requires a nonempty value")))?;
        apply(&mut parsed, option, value)?;
    }
    validate(&parsed)?;
    Ok(parsed)
}

fn integer(option: &str, value: &str) -> Result<i64, CheckReport> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(fail(&format!(
            "{option} requires a positive decimal integer"
        )));
    }
    value
        .parse::<i64>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| fail(&format!("{option} requires a positive decimal integer")))
}

fn apply(parsed: &mut Args, option: &str, value: &str) -> Result<(), CheckReport> {
    match option {
        "--repo" => parsed.repo = value.into(),
        "--workflow" => parsed.workflow = Some(value.into()),
        "--run-id" => parsed.run_id.push(integer(option, value)?),
        "--input" => parsed.input = Some(value.into()),
        "--compare-input" => parsed.compare_input = Some(value.into()),
        "--limit" => parsed.limit = integer(option, value)?,
        "--status" => parsed.status = value.into(),
        "--branch" => parsed.branch = Some(value.into()),
        "--event" => parsed.event = Some(value.into()),
        "--created" => parsed.created = Some(value.into()),
        "--top" => parsed.top = integer(option, value)?,
        "--label" => parsed.label.push(value.into()),
        "--json-out" => parsed.json_out = Some(value.into()),
        "--markdown-out" => parsed.markdown_out = Some(value.into()),
        "--raw-out" => parsed.raw_out = Some(value.into()),
        _ => unreachable!("options admitted before values"),
    }
    Ok(())
}

fn validate(args: &Args) -> Result<(), CheckReport> {
    let sources = usize::from(args.input.is_some())
        + usize::from(args.workflow.is_some())
        + usize::from(!args.run_id.is_empty());
    if sources != 1 {
        return Err(fail(
            "exactly one of --input, --workflow, or --run-id is required",
        ));
    }
    let stdout = [&args.json_out, &args.markdown_out, &args.raw_out]
        .into_iter()
        .filter(|value| value.as_deref() == Some("-"))
        .count();
    if stdout > 1 {
        return Err(fail("only one output may use stdout (-)"));
    }
    Ok(())
}
