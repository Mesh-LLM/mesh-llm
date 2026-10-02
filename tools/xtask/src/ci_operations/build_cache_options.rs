//! argparse's `_parse_optional` for `ci-ops build-cache`: exact options,
//! `--name=value`, unique long prefixes (ambiguity reported), bundled
//! single-dash values, and the negative-number/space positional rules.

use crate::repository::check_report::CheckReport;
use crate::repository::text::repr;

/// One argument as argparse classifies it.
pub(crate) enum Kind<'o> {
    Positional,
    Unknown,
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
    if is_negative_number(arg) || arg.contains(' ') {
        Kind::Positional
    } else {
        Kind::Unknown
    }
}

fn is_negative_number(arg: &str) -> bool {
    let Some(rest) = arg.strip_prefix('-') else {
        return false;
    };
    let digits = |text: &str| text.chars().all(|character| character.is_ascii_digit());
    match rest.split_once('.') {
        Some((whole, fraction)) => digits(whole) && !fraction.is_empty() && digits(fraction),
        None => !rest.is_empty() && digits(rest),
    }
}

pub(crate) fn is_option_like(arg: &str, options: &[&str]) -> bool {
    arg == "--" || !matches!(classify(arg, options), Kind::Positional)
}

/// A zero-argument option with an attached value: bundled short `-hX`
/// still means `-h`; anything else is "ignored explicit argument".
pub(crate) fn help_flag(
    explicit: Option<String>,
    _sep: bool,
    name: &str,
    fail: &dyn Fn(&str) -> CheckReport,
) -> Result<(), CheckReport> {
    match explicit {
        None => Ok(()),
        Some(value) => Err(fail(&format!(
            "argument {name}: ignored explicit argument {}",
            repr(&value)
        ))),
    }
}
