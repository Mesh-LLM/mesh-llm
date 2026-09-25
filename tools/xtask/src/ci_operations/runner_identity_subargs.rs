//! argparse emulation for the `runner-identity` subcommands: their options,
//! required arguments, `--field` choices and the `lookup` positional.

use crate::ci_operations::runner_identity_argv::{Args, Cursor, error, is_optional, resolve};
use crate::ci_operations::runner_identity_help::{PROG, sub_help, sub_usage};
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;

const FIELDS: [&str; 4] = [
    "reference",
    "native_toolchain_epoch",
    "receipt",
    "provenance",
];

fn sub_options(command: &str) -> &'static [&'static str] {
    match command {
        "lookup" => &["-h", "--help", "--field"],
        "seed-key" => &["-h", "--help", "--recipe-hash"],
        "bind" => &[
            "-h",
            "--help",
            "--image-id",
            "--cohort",
            "--anchor",
            "--output",
        ],
        _ => &["-h", "--help"],
    }
}

pub(crate) fn parse_subcommand(
    parsed: &mut Args,
    args: &[String],
    extras: &mut Vec<String>,
) -> Result<(), CheckReport> {
    let command = parsed.command.clone();
    let prog = format!("{PROG} {command}");
    let usage = sub_usage(&command);
    let fail = |message: &str| error(&usage, &prog, message);
    let options = sub_options(&command);
    let mut cursor = Cursor { args, index: 0 };
    let mut role: Option<String> = None;
    let mut seen: Vec<&str> = Vec::new();
    while let Some(arg) = args.get(cursor.index) {
        cursor.index += 1;
        if arg == "--" {
            let rest = &args[cursor.index..];
            match (command == "lookup" && role.is_none(), rest.split_first()) {
                (true, Some((first, tail))) => {
                    role = Some(first.clone());
                    extras.extend(tail.iter().cloned());
                }
                _ => {
                    extras.push(arg.clone());
                    extras.extend(rest.iter().cloned());
                }
            }
            break;
        }
        if !is_optional(arg) {
            if command == "lookup" && role.is_none() {
                role = Some(arg.clone());
            } else {
                extras.push(arg.clone());
            }
            continue;
        }
        let Some((option, inline)) = resolve(arg, options) else {
            extras.push(arg.clone());
            continue;
        };
        if matches!(option, "-h" | "--help") {
            return Err(CheckReport::success(sub_help(&command)));
        }
        let value = cursor
            .value(inline)
            .ok_or_else(|| fail(&format!("argument {option}: expected one argument")))?;
        store(parsed, option, value, &fail)?;
        seen.push(option);
    }
    let required: &[&str] = match command.as_str() {
        "seed-key" => &["--recipe-hash"],
        "bind" => &["--image-id", "--cohort", "--anchor", "--output"],
        _ => &[],
    };
    let mut missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|name| !seen.contains(name))
        .collect();
    if command == "lookup" && role.is_none() {
        missing.push("role");
    }
    if !missing.is_empty() {
        return Err(fail(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        )));
    }
    parsed.role = role.unwrap_or_default();
    Ok(())
}

fn store(
    parsed: &mut Args,
    option: &str,
    value: String,
    fail: &dyn Fn(&str) -> CheckReport,
) -> Result<(), CheckReport> {
    match option {
        "--field" => {
            if !FIELDS.contains(&value.as_str()) {
                let choices = FIELDS
                    .iter()
                    .map(|name| repr(name))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(fail(&format!(
                    "argument --field: invalid choice: {} (choose from {choices})",
                    repr(&value)
                )));
            }
            parsed.field = Some(value);
        }
        "--recipe-hash" => parsed.recipe_hash = value,
        "--image-id" => parsed.image_id = value,
        "--cohort" => parsed.cohort = value,
        "--anchor" => parsed.anchor = value,
        _ => parsed.output = value,
    }
    Ok(())
}
