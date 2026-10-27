//! Exact options owned by the `runner-identity` subcommands.

use crate::ci_operations::runner_identity_argv::{Args, error};
use crate::ci_operations::runner_identity_help::{PROG, sub_help, sub_usage};
use crate::repository::check_report::CheckReport;

const FIELDS: [&str; 4] = [
    "reference",
    "native_toolchain_epoch",
    "receipt",
    "provenance",
];

pub(crate) fn parse_subcommand(
    parsed: &mut Args,
    args: &[String],
    extras: &mut Vec<String>,
) -> Result<(), CheckReport> {
    let command = parsed.command.clone();
    let prog = format!("{PROG} {command}");
    let usage = sub_usage(&command);
    let fail = |message: &str| error(&usage, &prog, message);
    let mut index = 0;
    let mut role: Option<String> = None;
    let mut seen: Vec<&str> = Vec::new();

    while let Some(arg) = args.get(index) {
        index += 1;
        if arg == "-h" || arg == "--help" {
            return Err(CheckReport::success(sub_help(&command)));
        }
        if arg == "--" {
            if command == "lookup"
                && role.is_none()
                && let Some(value) = args.get(index)
            {
                role = Some(value.clone());
                index += 1;
            }
            extras.extend(args[index..].iter().cloned());
            break;
        }
        if !arg.starts_with('-') || arg == "-" {
            if command == "lookup" && role.is_none() {
                role = Some(arg.clone());
            } else {
                extras.push(arg.clone());
            }
            continue;
        }

        let (option, attached) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(name, value)| {
                (name, Some(value.to_owned()))
            });
        let known = match command.as_str() {
            "lookup" => option == "--field",
            "seed-key" => option == "--recipe-hash",
            "bind" => matches!(option, "--image-id" | "--cohort" | "--anchor" | "--output"),
            _ => false,
        };
        if !known {
            extras.push(arg.clone());
            continue;
        }
        let value = take_value(option, attached, args, &mut index, &fail)?;
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
            "required arguments missing: {}",
            missing.join(", ")
        )));
    }
    parsed.role = role.unwrap_or_default();
    Ok(())
}

fn take_value(
    option: &str,
    attached: Option<String>,
    args: &[String],
    index: &mut usize,
    fail: &dyn Fn(&str) -> CheckReport,
) -> Result<String, CheckReport> {
    if let Some(value) = attached {
        return Ok(value);
    }
    let Some(value) = args.get(*index) else {
        return Err(fail(&format!("{option} requires a value")));
    };
    if value == "--" || value.starts_with('-') {
        return Err(fail(&format!("{option} requires a value")));
    }
    *index += 1;
    Ok(value.clone())
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
                return Err(fail(
                    "--field must be reference, native_toolchain_epoch, receipt, or provenance",
                ));
            }
            parsed.field = Some(value);
        }
        "--recipe-hash" => parsed.recipe_hash = value,
        "--image-id" => parsed.image_id = value,
        "--cohort" => parsed.cohort = value,
        "--anchor" => parsed.anchor = value,
        "--output" => parsed.output = value,
        _ => unreachable!("only supported options reach store"),
    }
    Ok(())
}
