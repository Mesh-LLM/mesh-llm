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

/// `classify` plus argparse's `allow_abbrev`: a unique `--` prefix resolves
/// to its option. Callers reject ambiguity first with `first_ambiguity`.
pub(crate) fn classify_abbreviated<'o>(arg: &str, options: &[&'o str]) -> Kind<'o> {
    match classify(arg, options) {
        Kind::Known(option, explicit, attached) => Kind::Known(option, explicit, attached),
        other => match prefix_matches(arg, options).as_slice() {
            [only] => {
                let explicit = arg.split_once('=').map(|(_, value)| value.to_owned());
                let attached = explicit.is_some();
                Kind::Known(only, explicit, attached)
            }
            _ => other,
        },
    }
}

/// argparse's `_get_option_tuples` for `--` arguments.
fn prefix_matches<'o>(arg: &str, options: &[&'o str]) -> Vec<&'o str> {
    if !arg.starts_with("--") {
        return Vec::new();
    }
    let prefix = arg.split_once('=').map_or(arg, |(prefix, _)| prefix);
    options
        .iter()
        .copied()
        .filter(|option| option.starts_with(prefix))
        .collect()
}

/// argparse reports an ambiguous prefix while classifying every argument
/// before `--`, ahead of any value or choice error.
pub(crate) fn first_ambiguity(args: &[String], options: &[&str]) -> Option<String> {
    args.iter()
        .take_while(|arg| *arg != "--")
        .filter(|arg| matches!(classify(arg, options), Kind::Positional | Kind::Unknown))
        .find_map(|arg| {
            let matches = prefix_matches(arg, options);
            (matches.len() > 1)
                .then(|| format!("ambiguous option: {arg} could match {}", matches.join(", ")))
        })
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

#[cfg(test)]
mod tests {
    use super::{Kind, classify, classify_abbreviated, first_ambiguity};

    const OPTIONS: [&str; 5] = ["-h", "--help", "--max-size", "--max-age", "--execute"];

    #[test]
    fn unique_long_prefix_resolves_with_attached_value() {
        assert!(matches!(
            classify_abbreviated("--exec", &OPTIONS),
            Kind::Known("--execute", None, false)
        ));
        assert!(matches!(
            classify_abbreviated("--max-s=1", &OPTIONS),
            Kind::Known("--max-size", Some(value), true) if value == "1"
        ));
        assert!(matches!(classify("--exec", &OPTIONS), Kind::Unknown));
    }

    #[test]
    fn shared_long_prefix_is_ambiguous() {
        let args = ["--max-size", "x", "--ma=1", "--", "--max"].map(String::from);
        assert_eq!(
            first_ambiguity(&args, &OPTIONS).as_deref(),
            Some("ambiguous option: --ma=1 could match --max-size, --max-age")
        );
        assert!(matches!(
            classify_abbreviated("--max", &OPTIONS),
            Kind::Unknown
        ));
    }

    #[test]
    fn unmatched_long_option_stays_unknown() {
        assert!(matches!(
            classify_abbreviated("--json", &OPTIONS),
            Kind::Unknown
        ));
        assert!(first_ambiguity(&["--json".to_owned()], &OPTIONS).is_none());
    }
}
