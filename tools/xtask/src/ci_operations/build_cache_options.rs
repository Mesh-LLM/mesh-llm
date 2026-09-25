//! argparse's `_parse_optional` for `ci-ops build-cache`: exact options,
//! `--name=value`, unique long prefixes (ambiguity reported), bundled
//! single-dash values, and the negative-number/space positional rules.

use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;

/// One argument as argparse classifies it.
pub(crate) enum Kind<'o> {
    Positional,
    Unknown,
    Ambiguous(Vec<&'o str>),
    /// Option, explicit value, and whether it was attached with `=`.
    Known(&'o str, Option<String>, bool),
}

pub(crate) fn classify<'o>(arg: &str, options: &[&'o str]) -> Kind<'o> {
    if !arg.starts_with('-') || arg == "-" {
        return Kind::Positional;
    }
    if let Some(exact) = options.iter().find(|option| **option == arg) {
        return Kind::Known(exact, None, false);
    }
    if let Some((name, value)) = arg.split_once('=')
        && let Some(exact) = options.iter().find(|option| **option == name)
    {
        return Kind::Known(exact, Some(value.to_owned()), true);
    }
    let matches = prefix_matches(arg, options);
    match matches.len() {
        0 => {}
        1 => return matches.into_iter().next().unwrap_or(Kind::Unknown),
        _ => {
            return Kind::Ambiguous(
                matches
                    .into_iter()
                    .filter_map(|kind| match kind {
                        Kind::Known(name, ..) => Some(name),
                        _ => None,
                    })
                    .collect(),
            );
        }
    }
    if is_negative_number(arg) || arg.contains(' ') {
        Kind::Positional
    } else {
        Kind::Unknown
    }
}

fn prefix_matches<'o>(arg: &str, options: &[&'o str]) -> Vec<Kind<'o>> {
    let (prefix, explicit) = match arg.split_once('=') {
        Some((prefix, value)) => (prefix, Some(value.to_owned())),
        None => (arg, None),
    };
    let long = arg.starts_with("--");
    let short = arg.get(..2).unwrap_or(arg);
    options
        .iter()
        .filter_map(|option| {
            if !long && *option == short {
                Some(Kind::Known(option, Some(arg[2..].to_owned()), false))
            } else if option.starts_with(prefix) {
                Some(Kind::Known(option, explicit.clone(), explicit.is_some()))
            } else {
                None
            }
        })
        .collect()
}

fn is_negative_number(arg: &str) -> bool {
    let Some(rest) = arg.strip_prefix('-') else {
        return false;
    };
    let digits = |text: &str| text.chars().all(char::is_numeric);
    match rest.split_once('.') {
        Some((whole, fraction)) => digits(whole) && !fraction.is_empty() && digits(fraction),
        None => !rest.is_empty() && digits(rest),
    }
}

pub(crate) fn is_option_like(arg: &str, options: &[&str]) -> bool {
    arg == "--" || !matches!(classify(arg, options), Kind::Positional)
}

pub(crate) fn ambiguous(
    arg: &str,
    names: &[&str],
    fail: &dyn Fn(&str) -> CheckReport,
) -> CheckReport {
    fail(&format!(
        "ambiguous option: {arg} could match {}",
        names.join(", ")
    ))
}

/// A zero-argument option with an attached value: bundled short `-hX`
/// still means `-h`; anything else is "ignored explicit argument".
pub(crate) fn help_flag(
    explicit: Option<String>,
    sep: bool,
    name: &str,
    fail: &dyn Fn(&str) -> CheckReport,
) -> Result<(), CheckReport> {
    match explicit {
        None => Ok(()),
        Some(value)
            if !sep && !value.is_empty() && !value.starts_with('-') && name == "-h/--help" =>
        {
            Ok(())
        }
        Some(value) => Err(fail(&format!(
            "argument {name}: ignored explicit argument {}",
            repr(&value)
        ))),
    }
}
