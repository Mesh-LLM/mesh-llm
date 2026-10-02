//! argparse emulation for `scripts/release-notes-classify.py` under Python
//! 3.13: string options, the `--has-entries` flag, unique-prefix
//! abbreviations, `=value`, `-h`, the required `--body`, unrecognized
//! arguments, and the script's own `parser.error` for the plan options.

use crate::ci_operations::build_cache_options::{Kind, classify, help_flag, is_option_like};
use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;

const PROG: &str = "release-notes-classify.py";
/// The legacy `-h` output with `COLUMNS` unset (80 columns).
const HELP: &str = include_str!("classify_help.txt");
const USAGE: &str = "\
usage: release-notes-classify.py [-h] --body BODY [--range RANGE]
                                 [--version VERSION] [--date DATE] [--out OUT]
                                 [--repo-root REPO_ROOT] [--has-entries]
                                 [--links LINKS]
";
const OPTIONS: [&str; 10] = [
    "-h",
    "--help",
    "--body",
    "--range",
    "--version",
    "--date",
    "--out",
    "--repo-root",
    "--has-entries",
    "--links",
];

/// The plan-building invocation: every option the script requires itself.
#[derive(Debug)]
pub(crate) struct PlanArgs {
    pub(crate) body: String,
    pub(crate) range: String,
    pub(crate) version: String,
    pub(crate) date: String,
    pub(crate) out: String,
    pub(crate) repo_root: Option<String>,
    pub(crate) links: Option<String>,
}

/// What the command line asks for.
#[derive(Debug)]
pub(crate) enum Invocation {
    /// `--has-entries`: only the body is read.
    HasEntries(String),
    Plan(PlanArgs),
}

#[derive(Default)]
struct Raw {
    body: Option<String>,
    range: Option<String>,
    version: Option<String>,
    date: Option<String>,
    out: Option<String>,
    repo_root: Option<String>,
    links: Option<String>,
    has_entries: bool,
}

fn fail(message: &str) -> CheckReport {
    error(USAGE, PROG, message)
}

/// Parses `args`, or returns the help/usage report argparse would produce.
pub(crate) fn parse(args: &[String]) -> Result<Invocation, CheckReport> {
    let raw = parse_raw(args)?;
    let Some(body) = raw.body else {
        return Err(fail("the following arguments are required: --body"));
    };
    if raw.has_entries {
        return Ok(Invocation::HasEntries(body));
    }
    let missing: Vec<&str> = [
        ("--range", raw.range.is_none()),
        ("--version", raw.version.is_none()),
        ("--date", raw.date.is_none()),
        ("--out", raw.out.is_none()),
    ]
    .into_iter()
    .filter_map(|(flag, absent)| absent.then_some(flag))
    .collect();
    match (raw.range, raw.version, raw.date, raw.out) {
        (Some(range), Some(version), Some(date), Some(out)) => Ok(Invocation::Plan(PlanArgs {
            body,
            range,
            version,
            date,
            out,
            repo_root: raw.repo_root,
            links: raw.links,
        })),
        _ => Err(fail(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ))),
    }
}

fn parse_raw(args: &[String]) -> Result<Raw, CheckReport> {
    let mut raw = Raw::default();
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
            Kind::Known("-h" | "--help", explicit, sep) => {
                help_flag(explicit, sep, "-h/--help", &fail)?;
                return Err(CheckReport::success(HELP.to_owned()));
            }
            Kind::Known("--has-entries", explicit, sep) => {
                help_flag(explicit, sep, "--has-entries", &fail)?;
                raw.has_entries = true;
            }
            Kind::Known(option, explicit, _) => {
                let value = take_value(explicit, args, &mut index, option)?;
                apply(&mut raw, option, value);
            }
        }
    }
    if raw.body.is_none() {
        return Err(fail("the following arguments are required: --body"));
    }
    if !extras.is_empty() {
        return Err(fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    Ok(raw)
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

fn apply(raw: &mut Raw, option: &str, value: String) {
    let slot = match option {
        "--body" => &mut raw.body,
        "--range" => &mut raw.range,
        "--version" => &mut raw.version,
        "--date" => &mut raw.date,
        "--out" => &mut raw.out,
        "--repo-root" => &mut raw.repo_root,
        _ => &mut raw.links,
    };
    *slot = Some(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(text: &str) -> Vec<String> {
        text.split(' ')
            .filter(|part| !part.is_empty())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn has_entries_mode_accepts_empty_body_without_render_metadata() {
        let parsed = parse(&argv("--body= --has-entries --range r"));
        assert!(matches!(parsed, Ok(Invocation::HasEntries(body)) if body.is_empty()));
    }
}
