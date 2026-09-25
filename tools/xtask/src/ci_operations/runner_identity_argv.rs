//! argparse emulation for `scripts/runner-image-identity.py`: top-level
//! `--root`/`--catalog`, one required subcommand that receives every later
//! argument, unique-prefix option abbreviations, `--name=value`, `-h`, and
//! unknown arguments collected and reported after a successful parse.

use crate::ci_operations::runner_identity_help::{PROG, TOP_USAGE, top_help};
use crate::ci_operations::runner_identity_subargs::parse_subcommand;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;

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

/// Whether argparse classifies `arg` as an option string ('O'): a dash
/// prefix that is neither a lone `-`, a negative number, nor spaced text.
pub(crate) fn is_optional(arg: &str) -> bool {
    let Some(rest) = arg.strip_prefix('-') else {
        return false;
    };
    let negative_number = match rest.split_once('.') {
        Some((whole, fraction)) => {
            whole.bytes().all(|b| b.is_ascii_digit())
                && !fraction.is_empty()
                && fraction.bytes().all(|b| b.is_ascii_digit())
        }
        None => !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()),
    };
    !rest.is_empty() && !negative_number && !arg.contains(' ')
}

/// Resolves an option string to one of `options` (exact or unique prefix
/// of a long option), splitting an attached `=value`.
pub(crate) fn resolve<'o>(arg: &str, options: &[&'o str]) -> Option<(&'o str, Option<String>)> {
    let (name, inline) = match arg.split_once('=') {
        Some((name, value)) if arg.starts_with("--") => (name, Some(value.to_owned())),
        _ => (arg, None),
    };
    if let Some(exact) = options.iter().find(|option| **option == name) {
        return Some((exact, inline));
    }
    if !name.starts_with("--") || name.len() < 3 {
        return None;
    }
    let matches: Vec<&&str> = options
        .iter()
        .filter(|option| option.starts_with(name))
        .collect();
    (matches.len() == 1).then(|| (*matches[0], inline))
}

pub(crate) struct Cursor<'a> {
    pub(crate) args: &'a [String],
    pub(crate) index: usize,
}

impl Cursor<'_> {
    pub(crate) fn value(&mut self, inline: Option<String>) -> Option<String> {
        if inline.is_some() {
            return inline;
        }
        let next = self.args.get(self.index)?;
        if is_optional(next) || next == "--" {
            return None;
        }
        self.index += 1;
        Some(next.clone())
    }
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
    let mut extras: Vec<String> = Vec::new();
    let mut cursor = Cursor { args, index: 0 };
    let mut positional_only = false;
    while let Some(arg) = args.get(cursor.index) {
        cursor.index += 1;
        if !positional_only && arg == "--" {
            positional_only = true;
            continue;
        }
        if positional_only || !is_optional(arg) {
            let command = arg.as_str();
            if !COMMANDS.contains(&command) {
                let choices = COMMANDS
                    .iter()
                    .map(|name| repr(name))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(top_error(&format!(
                    "argument command: invalid choice: {} (choose from {choices})",
                    repr(command)
                )));
            }
            parsed.command = command.to_owned();
            parse_subcommand(&mut parsed, &args[cursor.index..], &mut extras)?;
            return finish(parsed, &extras);
        }
        match resolve(arg, &["-h", "--help", "--root", "--catalog"]) {
            Some(("-h" | "--help", _)) => return Err(CheckReport::success(top_help())),
            Some((option, inline)) => {
                let value = cursor.value(inline).ok_or_else(|| {
                    top_error(&format!("argument {option}: expected one argument"))
                })?;
                if option == "--root" {
                    parsed.root = Some(value);
                } else {
                    parsed.catalog = Some(value);
                }
            }
            None => extras.push(arg.clone()),
        }
    }
    Err(top_error("the following arguments are required: command"))
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
