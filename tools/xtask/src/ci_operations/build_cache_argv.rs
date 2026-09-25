//! argparse emulation for `scripts/manage-build-cache.py`: one required
//! `{status,prune,build}` subcommand, unique-prefix long options, attached
//! `=value`, `-h` bundling, `--` handling and the `build` remainder, with
//! Python 3.13's sequential error order and exit status 2.

use crate::ci_operations::build_cache_help::{
    BUILD_USAGE, PROG, PRUNE_USAGE, STATUS_USAGE, TOP_USAGE, build_help, status_help, top_help,
};
use crate::ci_operations::build_cache_options::{
    Kind, ambiguous, classify, help_flag, is_option_like,
};
use crate::ci_operations::build_cache_values::{Age, parse_age, parse_size};
use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;

pub(crate) const DEFAULT_MAX_BYTES: i128 = 80 * 1024 * 1024 * 1024;
const COMMANDS: [&str; 3] = ["status", "prune", "build"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Status,
    Prune,
    Build,
}

pub(crate) struct Args {
    pub(crate) mode: Mode,
    pub(crate) workspace: Option<String>,
    pub(crate) target_dir: Option<String>,
    pub(crate) max_size: i128,
    pub(crate) max_age: Age,
    pub(crate) json: bool,
    pub(crate) execute: bool,
    pub(crate) build_command: Vec<String>,
}

struct Sub {
    usage: &'static str,
    prog: String,
    options: Vec<&'static str>,
}

impl Sub {
    fn fail(&self, message: &str) -> CheckReport {
        error(self.usage, &self.prog, message)
    }
}

pub(crate) enum Parsed {
    Run(Box<Args>),
    Report(CheckReport),
}

pub(crate) fn parse(args: &[String]) -> Parsed {
    match parse_top(args) {
        Ok(parsed) => Parsed::Run(Box::new(parsed)),
        Err(report) => Parsed::Report(report),
    }
}

fn top_error(message: &str) -> CheckReport {
    error(TOP_USAGE, PROG, message)
}

fn parse_top(args: &[String]) -> Result<Args, CheckReport> {
    let options = ["-h", "--help"];
    let mut extras: Vec<String> = Vec::new();
    let mut positional_only = false;
    for (index, arg) in args.iter().enumerate() {
        if !positional_only && arg == "--" {
            positional_only = true;
            continue;
        }
        let kind = if positional_only {
            Kind::Positional
        } else {
            classify(arg, &options)
        };
        match kind {
            Kind::Positional => {
                let mode = match arg.as_str() {
                    "status" => Mode::Status,
                    "prune" => Mode::Prune,
                    "build" => Mode::Build,
                    _ => {
                        let choices: Vec<String> = COMMANDS.iter().map(|name| repr(name)).collect();
                        return Err(top_error(&format!(
                            "argument command: invalid choice: {} (choose from {})",
                            repr(arg),
                            choices.join(", ")
                        )));
                    }
                };
                let parsed = parse_sub(mode, &args[index + 1..], &mut extras)?;
                if extras.is_empty() {
                    return Ok(parsed);
                }
                return Err(top_error(&format!(
                    "unrecognized arguments: {}",
                    extras.join(" ")
                )));
            }
            Kind::Unknown => extras.push(arg.clone()),
            Kind::Ambiguous(names) => return Err(ambiguous(arg, &names, &top_error)),
            Kind::Known(_, explicit, sep) => {
                help_flag(explicit, sep, "-h/--help", &top_error)?;
                return Err(CheckReport::success(top_help()));
            }
        }
    }
    Err(top_error("the following arguments are required: command"))
}

fn sub_spec(mode: Mode) -> Sub {
    let (usage, name) = match mode {
        Mode::Status => (STATUS_USAGE, "status"),
        Mode::Prune => (PRUNE_USAGE, "prune"),
        Mode::Build => (BUILD_USAGE, "build"),
    };
    let mut options = vec!["-h", "--help", "--workspace", "--target-dir"];
    if mode != Mode::Build {
        options.extend(["--max-size", "--max-age", "--json"]);
    }
    if mode == Mode::Prune {
        options.push("--execute");
    }
    Sub {
        usage,
        prog: format!("{PROG} {name}"),
        options,
    }
}

fn parse_sub(mode: Mode, args: &[String], extras: &mut Vec<String>) -> Result<Args, CheckReport> {
    let sub = sub_spec(mode);
    let mut parsed = Args {
        mode,
        workspace: None,
        target_dir: None,
        max_size: DEFAULT_MAX_BYTES,
        max_age: Age::default_days(),
        json: false,
        execute: false,
        build_command: Vec::new(),
    };
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        index += 1;
        let kind = if arg == "--" {
            Kind::Positional
        } else {
            classify(arg, &sub.options)
        };
        match kind {
            Kind::Positional if mode == Mode::Build => {
                parsed.build_command = args[index - 1..].to_vec();
                return Ok(parsed);
            }
            Kind::Positional if arg == "--" => {
                extras.extend(args[index - 1..].iter().cloned());
                return Ok(parsed);
            }
            Kind::Positional | Kind::Unknown => extras.push(arg.clone()),
            Kind::Ambiguous(names) => return Err(ambiguous(arg, &names, &|m| sub.fail(m))),
            Kind::Known(option, explicit, sep) => {
                let value = match option {
                    "-h" | "--help" | "--json" | "--execute" => {
                        let name = if matches!(option, "-h" | "--help") {
                            "-h/--help"
                        } else {
                            option
                        };
                        help_flag(explicit, sep, name, &|m| sub.fail(m))?;
                        None
                    }
                    _ => Some(take_value(explicit, args, &mut index, &sub, option)?),
                };
                apply(&mut parsed, option, value, &sub)?;
            }
        }
    }
    Ok(parsed)
}

fn take_value(
    explicit: Option<String>,
    args: &[String],
    index: &mut usize,
    sub: &Sub,
    option: &str,
) -> Result<String, CheckReport> {
    if let Some(value) = explicit {
        return Ok(value);
    }
    match args.get(*index) {
        Some(next) if !is_option_like(next, &sub.options) => {
            *index += 1;
            Ok(next.clone())
        }
        _ => Err(sub.fail(&format!("argument {option}: expected one argument"))),
    }
}

fn apply(
    parsed: &mut Args,
    option: &str,
    value: Option<String>,
    sub: &Sub,
) -> Result<(), CheckReport> {
    let value = value.unwrap_or_default();
    let invalid = |message: String| sub.fail(&format!("argument {option}: {message}"));
    match option {
        "-h" | "--help" => {
            let help = match parsed.mode {
                Mode::Status => status_help(false),
                Mode::Prune => status_help(true),
                Mode::Build => build_help(),
            };
            return Err(CheckReport::success(help));
        }
        "--json" => parsed.json = true,
        "--execute" => parsed.execute = true,
        "--workspace" => parsed.workspace = Some(value),
        "--target-dir" => parsed.target_dir = Some(value),
        "--max-size" => parsed.max_size = parse_size(&value).map_err(invalid)?,
        _ => parsed.max_age = parse_age(&value).map_err(invalid)?,
    }
    Ok(())
}
