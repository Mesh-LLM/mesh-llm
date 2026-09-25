//! Closed argument grammar for the ported checks: named value options
//! (`--name value` or `--name=value`), boolean flags and positionals. Errors
//! become argparse-style usage reports with status 2.

use crate::repository::check_report::CheckReport;

/// The options one command accepts.
pub(crate) struct Grammar {
    pub(crate) usage: &'static str,
    pub(crate) values: &'static [&'static str],
    pub(crate) flags: &'static [&'static str],
}

#[derive(Debug, Default)]
pub(crate) struct ParsedArgs {
    values: Vec<(&'static str, String)>,
    flags: Vec<&'static str>,
    pub(crate) positionals: Vec<String>,
}

impl ParsedArgs {
    /// Every value given for `name`, in command-line order.
    pub(crate) fn all(&self, name: &str) -> Vec<&str> {
        self.values
            .iter()
            .filter(|(option, _)| *option == name)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    /// The last value given for `name`, as argparse keeps it.
    pub(crate) fn last(&self, name: &str) -> Option<&str> {
        self.all(name).pop()
    }

    pub(crate) fn flag(&self, name: &str) -> bool {
        self.flags.contains(&name)
    }
}

impl Grammar {
    pub(crate) fn parse(&self, args: &[String]) -> Result<ParsedArgs, CheckReport> {
        let mut parsed = ParsedArgs::default();
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            if arg == "--" {
                parsed.positionals.extend(rest.by_ref().cloned());
                break;
            }
            if !arg.starts_with("--") {
                parsed.positionals.push(arg.clone());
                continue;
            }
            let (name, inline) = match arg.split_once('=') {
                Some((name, value)) => (name, Some(value.to_owned())),
                None => (arg.as_str(), None),
            };
            if let Some(flag) = self.flags.iter().find(|flag| **flag == name) {
                if inline.is_some() {
                    return Err(self.error(&format!("argument {name}: ignored explicit argument")));
                }
                parsed.flags.push(flag);
                continue;
            }
            let Some(option) = self.values.iter().find(|option| **option == name) else {
                return Err(self.error(&format!("unrecognized arguments: {arg}")));
            };
            let value = match inline {
                Some(value) => value,
                None => rest
                    .next()
                    .filter(|value| !looks_like_option(value))
                    .cloned()
                    .ok_or_else(|| {
                        self.error(&format!("argument {name}: expected one argument"))
                    })?,
            };
            parsed.values.push((option, value));
        }
        Ok(parsed)
    }

    pub(crate) fn error(&self, message: &str) -> CheckReport {
        CheckReport::usage(self.usage, message)
    }
}

/// argparse refuses a separate option value that looks like another option:
/// a dash-prefixed word without spaces that is not a negative number. This
/// also keeps such values away from the git argv they are forwarded to.
fn looks_like_option(value: &str) -> bool {
    let Some(rest) = value.strip_prefix('-') else {
        return false;
    };
    let negative_number = match rest.split_once('.') {
        Some((whole, fraction)) => {
            whole.chars().all(|ch| ch.is_ascii_digit())
                && !fraction.is_empty()
                && fraction.chars().all(|ch| ch.is_ascii_digit())
        }
        None => !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()),
    };
    !rest.is_empty() && !value.contains(' ') && !negative_number
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAMMAR: Grammar = Grammar {
        usage: "check [--file FILE] [--quiet] [path]",
        values: &["--file"],
        flags: &["--quiet"],
    };

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn migration_repository_args_parse_values_flags_and_positionals() {
        let parsed = GRAMMAR
            .parse(&strings(&[
                "--file", "a", "p", "--file=b", "--quiet", "--", "--x",
            ]))
            .expect("valid arguments");
        assert_eq!(parsed.all("--file"), ["a", "b"]);
        assert_eq!(parsed.last("--file"), Some("b"));
        assert!(parsed.flag("--quiet"));
        assert_eq!(parsed.positionals, ["p", "--x"]);
    }

    #[test]
    fn migration_repository_args_reject_unknown_and_missing_values() {
        let accepted = GRAMMAR
            .parse(&strings(&["--file", "- x", "--file", "-5", "--file", "-"]))
            .expect("argparse-accepted values");
        assert_eq!(accepted.all("--file"), ["- x", "-5", "-"]);
        for args in [
            &["--bogus"][..],
            &["--file"][..],
            &["--quiet=1"][..],
            &["--file", "-x"][..],
            &["--file", "--"][..],
        ] {
            let report = GRAMMAR
                .parse(&strings(args))
                .expect_err("invalid arguments");
            assert_eq!(report.code, 2, "{args:?}");
            assert!(
                report.stderr.starts_with("usage: check "),
                "{}",
                report.stderr
            );
        }
    }
}
