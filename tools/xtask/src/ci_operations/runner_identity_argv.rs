//! Typed arguments for `ci-ops runner-identity`.

use crate::ci_operations::runner_identity_help::{PROG, TOP_USAGE, top_help};
use crate::ci_operations::runner_identity_subargs::parse_subcommand;
use crate::repository::check_report::CheckReport;

const COMMANDS: [&str; 6] = [
    "validate", "check", "diagnose", "lookup", "seed-key", "bind",
];

#[derive(Debug, Default)]
pub(crate) struct Args {
    pub(crate) root: Option<String>,
    pub(crate) catalog: Option<String>,
    pub(crate) command: String,
    pub(crate) role: String,
    pub(crate) field: Option<String>,
    pub(crate) recipe_hash: String,
    pub(crate) image_id: String,
    pub(crate) cohort: String,
    pub(crate) anchor: String,
    pub(crate) output: String,
}

pub(crate) fn error(usage: &str, prog: &str, message: &str) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{usage}{prog}: error: {message}\n"),
        code: 2,
    }
}

fn top_error(message: &str) -> CheckReport {
    error(TOP_USAGE, PROG, message)
}

pub(crate) enum Parsed {
    Run(Args),
    Report(CheckReport),
}

pub(crate) fn parse(args: &[String]) -> Parsed {
    match parse_inner(args) {
        Ok(parsed) => Parsed::Run(parsed),
        Err(report) => Parsed::Report(report),
    }
}

fn parse_inner(args: &[String]) -> Result<Args, CheckReport> {
    let mut parsed = Args::default();
    let mut extras = Vec::new();
    let mut index = 0;
    let mut positional_only = false;

    while let Some(arg) = args.get(index) {
        index += 1;
        if !positional_only && arg == "--" {
            positional_only = true;
            continue;
        }
        if positional_only || !arg.starts_with('-') || arg == "-" {
            if !COMMANDS.contains(&arg.as_str()) {
                return Err(top_error(&format!("unknown command: {arg}")));
            }
            parsed.command = arg.clone();
            parse_subcommand(&mut parsed, &args[index..], &mut extras)?;
            return finish(parsed, &extras);
        }
        if arg == "-h" || arg == "--help" {
            return Err(CheckReport::success(top_help()));
        }

        let (option, attached) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(name, value)| {
                (name, Some(value.to_owned()))
            });
        let value = match option {
            "--root" | "--catalog" => Some(take_value(option, attached, args, &mut index)?),
            _ => {
                extras.push(arg.clone());
                None
            }
        };
        match (option, value) {
            ("--root", Some(value)) => parsed.root = Some(value),
            ("--catalog", Some(value)) => parsed.catalog = Some(value),
            _ => {}
        }
    }
    Err(top_error(
        "a command is required: validate, check, diagnose, lookup, seed-key, or bind",
    ))
}

fn take_value(
    option: &str,
    attached: Option<String>,
    args: &[String],
    index: &mut usize,
) -> Result<String, CheckReport> {
    if let Some(value) = attached {
        return Ok(value);
    }
    let Some(value) = args.get(*index) else {
        return Err(top_error(&format!("{option} requires a value")));
    };
    if value == "--" || value.starts_with('-') {
        return Err(top_error(&format!("{option} requires a value")));
    }
    *index += 1;
    Ok(value.clone())
}

fn finish(parsed: Args, extras: &[String]) -> Result<Args, CheckReport> {
    if extras.is_empty() {
        Ok(parsed)
    } else {
        Err(top_error(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{Parsed, parse};

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_command_and_attached_values() {
        let Parsed::Run(parsed) = parse(&args(&[
            "--root",
            "/workspace",
            "--catalog=images.json",
            "lookup",
            "ui-quality",
            "--field=reference",
        ])) else {
            panic!("valid runner identity arguments should parse");
        };
        assert_eq!(parsed.root.as_deref(), Some("/workspace"));
        assert_eq!(parsed.catalog.as_deref(), Some("images.json"));
        assert_eq!(parsed.command, "lookup");
        assert_eq!(parsed.role, "ui-quality");
        assert_eq!(parsed.field.as_deref(), Some("reference"));
    }

    #[test]
    fn rejects_unknown_options_and_commands() {
        for input in [
            args(&["--roo=/workspace", "validate"]),
            args(&["bogus"]),
            args(&["lookup", "role", "--f=reference"]),
        ] {
            let Parsed::Report(error) = parse(&input) else {
                panic!("unknown arguments should be rejected");
            };
            assert_eq!(error.code, 2);
            assert!(error.stdout.is_empty());
            assert!(!error.stderr.is_empty());
        }
    }
}
