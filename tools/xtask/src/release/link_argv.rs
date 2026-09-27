//! argparse emulation for `scripts/release-notes-link.py` under Python 3.13:
//! required string options, the `int` budget, unique-prefix abbreviations,
//! `=value`, `-h`, then required-option and unrecognized-argument errors.

use crate::ci_operations::build_cache_options::{
    Kind, ambiguous, classify, help_flag, is_option_like,
};
use crate::ci_operations::ci_metrics_int::python_int;
use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;

const PROG: &str = "release-notes-link.py";
/// The legacy `-h` output with `COLUMNS` unset (80 columns).
const HELP: &str = include_str!("link_help.txt");
const USAGE: &str = "\
usage: release-notes-link.py [-h] --body BODY --range RANGE --repo REPO
                             [--repo-root REPO_ROOT] --out-body OUT_BODY
                             --out-links OUT_LINKS [--api-budget API_BUDGET]
";
const OPTIONS: [&str; 9] = [
    "-h",
    "--help",
    "--body",
    "--range",
    "--repo",
    "--repo-root",
    "--out-body",
    "--out-links",
    "--api-budget",
];
const REQUIRED: [&str; 5] = ["--body", "--range", "--repo", "--out-body", "--out-links"];

/// `DEFAULT_API_BUDGET`.
const DEFAULT_API_BUDGET: i64 = 200;

#[derive(Debug, Default)]
pub(crate) struct Args {
    pub(crate) body: String,
    pub(crate) range: String,
    pub(crate) repo: String,
    pub(crate) repo_root: Option<String>,
    pub(crate) out_body: String,
    pub(crate) out_links: String,
    pub(crate) api_budget: i64,
}

fn fail(message: &str) -> CheckReport {
    error(USAGE, PROG, message)
}

/// Parses `args`, or returns the help/usage report argparse would produce.
pub(crate) fn parse(args: &[String]) -> Result<Args, CheckReport> {
    let mut parsed = Args {
        api_budget: DEFAULT_API_BUDGET,
        ..Args::default()
    };
    let mut seen: Vec<&str> = Vec::new();
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
                return Err(CheckReport::success(HELP.to_owned()));
            }
            Kind::Known(option, explicit, _) => {
                let value = take_value(explicit, args, &mut index, option)?;
                apply(&mut parsed, option, value)?;
                seen.push(option);
            }
        }
    }
    let missing: Vec<&str> = REQUIRED
        .into_iter()
        .filter(|option| !seen.contains(option))
        .collect();
    if !missing.is_empty() {
        return Err(fail(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        )));
    }
    if !extras.is_empty() {
        return Err(fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
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

fn apply(parsed: &mut Args, option: &str, value: String) -> Result<(), CheckReport> {
    match option {
        "--body" => parsed.body = value,
        "--range" => parsed.range = value,
        "--repo" => parsed.repo = value,
        "--repo-root" => parsed.repo_root = Some(value),
        "--out-body" => parsed.out_body = value,
        "--out-links" => parsed.out_links = value,
        _ => {
            parsed.api_budget = python_int(&value).ok_or_else(|| {
                fail(&format!(
                    "argument {option}: invalid int value: {}",
                    repr(&value)
                ))
            })?;
        }
    }
    Ok(())
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

    fn stderr(text: &str) -> String {
        match parse(&argv(text)) {
            Ok(_) => String::new(),
            Err(report) => report.stderr.lines().last().unwrap_or("").to_owned(),
        }
    }

    #[test]
    fn migration_release_link_argv_matches_argparse() {
        let prefix = "release-notes-link.py: error: ";
        let cases = [
            (
                "--api-budget x --out y",
                "argument --api-budget: invalid int value: 'x'",
            ),
            (
                "zz --out",
                "ambiguous option: --out could match --out-body, --out-links",
            ),
            ("--body --x", "argument --body: expected one argument"),
            (
                "--b=1 --ra 2 --repo-r 3 --api 4 zz",
                "the following arguments are required: --repo, --out-body, --out-links",
            ),
            (
                "--body b --range r --repo o --out-body x --out-links y -- z",
                "unrecognized arguments: -- z",
            ),
            (
                "--help=1",
                "argument -h/--help: ignored explicit argument '1'",
            ),
        ];
        for (args, message) in cases {
            assert_eq!(stderr(args), format!("{prefix}{message}"), "{args}");
        }
        let parsed = parse(&argv(
            "--body= --range r --repo o --out-b x --out-l y --api 0",
        ));
        let parsed = parsed.map_err(|report| report.stderr).expect("valid argv");
        assert_eq!((parsed.body.as_str(), parsed.api_budget), ("", 0));
    }
}
