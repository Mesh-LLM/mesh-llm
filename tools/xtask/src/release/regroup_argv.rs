//! argparse emulation for `scripts/release-notes-regroup.py` under Python
//! 3.13: string options, the `--list`/`--check` flags, unique-prefix
//! abbreviations, `=value`, `-h`, the required `--body`, unrecognized
//! arguments, and the script's own `parser.error` messages.

use crate::ci_operations::build_cache_options::{Kind, classify, help_flag, is_option_like};
use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;

const PROG: &str = "release-notes-regroup.py";
/// The legacy `-h` output with `COLUMNS` unset (80 columns).
const HELP: &str = include_str!("regroup_help.txt");
const USAGE: &str = "\
usage: release-notes-regroup.py [-h] --body BODY [--plan PLAN] [--out OUT]
                                [--list] [--check]
                                [--metadata-from METADATA_FROM]
";
const OPTIONS: [&str; 8] = [
    "-h",
    "--help",
    "--body",
    "--plan",
    "--out",
    "--list",
    "--check",
    "--metadata-from",
];

/// The parsed command line.
#[derive(Debug, Default)]
pub(crate) struct Args {
    pub(crate) body: String,
    pub(crate) plan: Option<String>,
    pub(crate) out: Option<String>,
    pub(crate) list: bool,
    pub(crate) check: bool,
    pub(crate) metadata_from: Option<String>,
}

/// `parser.error(message)`: usage, the message, status 2.
pub(crate) fn fail(message: &str) -> CheckReport {
    error(USAGE, PROG, message)
}

/// Parses `args`, or returns the help/usage report argparse would produce.
pub(crate) fn parse(args: &[String]) -> Result<Args, CheckReport> {
    let mut parsed = Args::default();
    let mut body = None;
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
            Kind::Known("--list", explicit, sep) => {
                help_flag(explicit, sep, "--list", &fail)?;
                parsed.list = true;
            }
            Kind::Known("--check", explicit, sep) => {
                help_flag(explicit, sep, "--check", &fail)?;
                parsed.check = true;
            }
            Kind::Known(option, explicit, _) => {
                let value = take_value(explicit, args, &mut index, option)?;
                match option {
                    "--body" => body = Some(value),
                    "--plan" => parsed.plan = Some(value),
                    "--out" => parsed.out = Some(value),
                    _ => parsed.metadata_from = Some(value),
                }
            }
        }
    }
    let Some(body) = body else {
        return Err(fail("the following arguments are required: --body"));
    };
    if !extras.is_empty() {
        return Err(fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    parsed.body = body;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn stderr(text: &str) -> String {
        let argv: Vec<String> = text.split(' ').map(str::to_owned).collect();
        match parse(&argv) {
            Ok(_) => String::new(),
            Err(report) => report.stderr.lines().last().unwrap_or("").to_owned(),
        }
    }

    #[test]
    fn migration_release_regroup_argv_matches_argparse() {
        let prefix = "release-notes-regroup.py: error: ";
        let cases = [
            ("--body --x", "argument --body: expected one argument"),
            (
                "--plan p zz",
                "the following arguments are required: --body",
            ),
            ("--body b zz", "unrecognized arguments: zz"),
            (
                "--body b --list=1",
                "argument --list: ignored explicit argument \"1\"",
            ),
        ];
        for (args, message) in cases {
            assert_eq!(stderr(args), format!("{prefix}{message}"), "{args}");
        }
        let argv: Vec<String> = ["--body=x", "--check", "--metadata-from", "t", "--list"]
            .map(str::to_owned)
            .to_vec();
        let parsed = parse(&argv).ok();
        assert!(parsed.is_some_and(|args| args.body == "x"
            && args.check
            && args.list
            && args.metadata_from.as_deref() == Some("t")));
    }
}
