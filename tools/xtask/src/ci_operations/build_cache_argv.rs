//! Typed arguments for `ci-ops build-cache`.

use crate::ci_operations::build_cache_help::{
    BUILD_USAGE, PROG, PRUNE_USAGE, STATUS_USAGE, TOP_USAGE, build_help, status_help, top_help,
};
use crate::ci_operations::build_cache_values::{Age, parse_size};
use crate::repository::check_report::CheckReport;

pub(crate) const DEFAULT_MAX_BYTES: i128 = 80 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

pub(crate) enum Parsed {
    Run(Box<Args>),
    Report(CheckReport),
}

fn top_error(message: &str) -> CheckReport {
    cli_error(TOP_USAGE, PROG, message)
}

fn sub_error(mode: Mode, message: &str) -> CheckReport {
    let (usage, name) = match mode {
        Mode::Status => (STATUS_USAGE, "status"),
        Mode::Prune => (PRUNE_USAGE, "prune"),
        Mode::Build => (BUILD_USAGE, "build"),
    };
    cli_error(usage, &format!("{PROG} {name}"), message)
}

fn cli_error(usage: &str, prog: &str, message: &str) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{usage}{prog}: error: {message}\n"),
        code: 2,
    }
}

pub(crate) fn parse(args: &[String]) -> Parsed {
    match parse_inner(args) {
        Ok(parsed) => Parsed::Run(Box::new(parsed)),
        Err(report) => Parsed::Report(report),
    }
}

fn parse_inner(args: &[String]) -> Result<Args, CheckReport> {
    let (command_index, mode) = locate_command(args)?;
    parse_options(args, command_index, mode)
}

fn locate_command(args: &[String]) -> Result<(usize, Mode), CheckReport> {
    let Some(command_index) = args
        .iter()
        .position(|arg| !arg.starts_with('-') || arg == "-")
    else {
        if args.iter().any(|arg| arg == "-h" || arg == "--help") {
            return Err(CheckReport::success(top_help()));
        }
        return Err(top_error("expected a command: status, prune, or build"));
    };

    let command = &args[command_index];
    let mode = match command.as_str() {
        "status" => Mode::Status,
        "prune" => Mode::Prune,
        "build" => Mode::Build,
        "-h" | "--help" => return Err(CheckReport::success(top_help())),
        _ => return Err(top_error("expected a command: status, prune, or build")),
    };

    if args[..command_index]
        .iter()
        .any(|arg| arg != "--" && arg != "-h" && arg != "--help")
    {
        return Err(top_error("options must follow the build-cache command"));
    }
    if args[..command_index]
        .iter()
        .any(|arg| arg == "-h" || arg == "--help")
    {
        return Err(CheckReport::success(top_help()));
    }
    Ok((command_index, mode))
}

fn parse_options(args: &[String], command_index: usize, mode: Mode) -> Result<Args, CheckReport> {
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
    let mut index = command_index + 1;
    while let Some(arg) = args.get(index) {
        index += 1;
        if arg == "-h" || arg == "--help" {
            let help = match mode {
                Mode::Status => status_help(false),
                Mode::Prune => status_help(true),
                Mode::Build => build_help(),
            };
            return Err(CheckReport::success(help));
        }
        if mode == Mode::Build && arg == "--" {
            parsed.build_command = args[index..].to_vec();
            return Ok(parsed);
        }
        if mode == Mode::Build && !arg.starts_with('-') {
            parsed.build_command = args[index - 1..].to_vec();
            return Ok(parsed);
        }

        let (option, attached) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(name, value)| {
                (name, Some(value.to_owned()))
            });
        match option {
            "--workspace" => {
                parsed.workspace =
                    Some(take_value(mode, option, attached, args, &mut index, false)?);
            }
            "--target-dir" => {
                parsed.target_dir =
                    Some(take_value(mode, option, attached, args, &mut index, false)?);
            }
            "--max-size" if mode != Mode::Build => {
                let value = take_value(mode, option, attached, args, &mut index, true)?;
                parsed.max_size = parse_size(&value).map_err(|message| {
                    sub_error(mode, &format!("invalid --max-size: {message}"))
                })?;
            }
            "--max-age" if mode != Mode::Build => {
                let value = take_value(mode, option, attached, args, &mut index, true)?;
                parsed.max_age = value
                    .parse::<i128>()
                    .map(Age)
                    .map_err(|_| sub_error(mode, "--max-age must be a whole number of days"))?;
            }
            "--json" if mode != Mode::Build => {
                reject_attached(mode, option, attached)?;
                parsed.json = true;
            }
            "--execute" if mode == Mode::Prune => {
                reject_attached(mode, option, attached)?;
                parsed.execute = true;
            }
            _ => return Err(sub_error(mode, &format!("unknown argument: {arg}"))),
        }
    }
    Ok(parsed)
}

fn take_value(
    mode: Mode,
    option: &str,
    attached: Option<String>,
    args: &[String],
    index: &mut usize,
    allow_negative: bool,
) -> Result<String, CheckReport> {
    if let Some(value) = attached {
        return Ok(value);
    }
    let Some(value) = args.get(*index) else {
        return Err(sub_error(mode, &format!("{option} requires a value")));
    };
    if value == "--" || (value.starts_with('-') && !allow_negative) {
        return Err(sub_error(mode, &format!("{option} requires a value")));
    }
    *index += 1;
    Ok(value.clone())
}

fn reject_attached(mode: Mode, option: &str, attached: Option<String>) -> Result<(), CheckReport> {
    if attached.is_some() {
        return Err(sub_error(mode, &format!("{option} does not take a value")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Mode, Parsed, parse};

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_status_and_prune_options_with_typed_values() {
        let Parsed::Run(status) = parse(&args(&[
            "status",
            "--workspace=/tmp/workspace",
            "--target-dir",
            "/tmp/target",
            "--max-size",
            "10 KiB",
            "--max-age",
            "3",
            "--json",
        ])) else {
            panic!("valid status arguments should parse");
        };
        assert_eq!(status.mode, Mode::Status);
        assert_eq!(status.workspace.as_deref(), Some("/tmp/workspace"));
        assert_eq!(status.target_dir.as_deref(), Some("/tmp/target"));
        assert_eq!(status.max_size, 10 * 1024);
        assert_eq!(status.max_age.0, 3);
        assert!(status.json);

        let Parsed::Run(prune) = parse(&args(&["prune", "--execute", "--max-age", "-1"])) else {
            panic!("negative age is validated by the operation");
        };
        assert!(prune.execute);
        assert_eq!(prune.max_age.0, -1);
    }

    #[test]
    fn build_preserves_the_command_and_its_arguments() {
        let Parsed::Run(build) = parse(&args(&["build", "--", "env", "A=B", "just", "build"]))
        else {
            panic!("valid build arguments should parse");
        };
        assert_eq!(build.mode, Mode::Build);
        assert_eq!(build.build_command, ["env", "A=B", "just", "build"]);

        let Parsed::Run(direct) = parse(&args(&["build", "sh", "-c", "printf 'a b'; exit 4"]))
        else {
            panic!("build command without a delimiter should parse");
        };
        assert_eq!(direct.build_command, ["sh", "-c", "printf 'a b'; exit 4"]);
    }

    #[test]
    fn rejects_abbreviated_options_and_malformed_values() {
        for input in [
            args(&["status", "--max", "10"]),
            args(&["status", "--max-age", "1.5"]),
            args(&["prune", "--execute=yes"]),
            args(&["build", "--json"]),
        ] {
            let Parsed::Report(error) = parse(&input) else {
                panic!("invalid arguments should be rejected");
            };
            assert_eq!(error.code, 2);
            assert!(error.stdout.is_empty());
            assert!(!error.stderr.is_empty());
        }
    }
}
