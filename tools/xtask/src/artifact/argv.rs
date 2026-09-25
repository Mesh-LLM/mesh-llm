//! Python 3.13 argparse semantics for the legacy artifact scripts' shared
//! grammar: required positionals plus the implicit `-h/--help`. Covers
//! option classification, `--` handling, help abbreviations, the "ignored
//! explicit argument" and "ambiguous option" errors, and error precedence.

use crate::repository::check_report::CheckReport;
use crate::repository::python_text::{is_decimal, repr};

/// The legacy program name and positionals that shape usage and help text.
pub(super) struct Program {
    pub(super) name: &'static str,
    pub(super) positionals: &'static [&'static str],
}

/// How argparse reads one command-line word before the first `--`.
enum Token<'a> {
    Positional,
    Separator,
    Unknown,
    /// `-h`/`--help`, with any explicit argument and whether it was short.
    Help {
        explicit: Option<&'a str>,
        short: bool,
        equals: bool,
    },
    Ambiguous,
}

impl Program {
    /// The positional values in declaration order, or the report argparse
    /// would have produced.
    pub(super) fn parse<'a>(&self, args: &'a [String]) -> Result<Vec<&'a str>, CheckReport> {
        let tokens = tokenize(args);
        let mut values = Vec::with_capacity(self.positionals.len());
        let mut extras = Vec::new();
        let mut index = 0;
        while index < tokens.len() {
            let (arg, token) = (&args[index], &tokens[index]);
            match token {
                Token::Positional | Token::Separator => {
                    let end = run_end(&tokens, index);
                    while values.len() < self.positionals.len()
                        && let Some((value, consumed)) = take_positional(&tokens[index..end])
                    {
                        values.push(args[index + value].as_str());
                        index += consumed;
                    }
                    extras.extend(args[index..end].iter().map(String::as_str));
                    index = end;
                    continue;
                }
                Token::Unknown => extras.push(arg.as_str()),
                Token::Help {
                    explicit,
                    short,
                    equals,
                } => {
                    return Err(self.help(*explicit, *short, *equals));
                }
                Token::Ambiguous => {
                    let message = format!("ambiguous option: {arg} could match -h, --help");
                    return Err(self.error(&message));
                }
            }
            index += 1;
        }
        if let Some(missing) = self.positionals.get(values.len()..)
            && !missing.is_empty()
        {
            let message = format!(
                "the following arguments are required: {}",
                missing.join(", ")
            );
            return Err(self.error(&message));
        }
        if !extras.is_empty() {
            return Err(self.error(&format!("unrecognized arguments: {}", extras.join(" "))));
        }
        Ok(values)
    }

    fn usage(&self) -> String {
        format!("usage: {} [-h] {}\n", self.name, self.positionals.join(" "))
    }

    /// argparse's help column: two past the widest invocation, capped at 24.
    fn help_text(&self) -> String {
        const HELP_FLAG: &str = "-h, --help";
        let widest = self
            .positionals
            .iter()
            .map(|name| name.len())
            .chain([HELP_FLAG.len()])
            .max()
            .unwrap_or(HELP_FLAG.len());
        let column = (widest + 4).min(24);
        let listing: String = self
            .positionals
            .iter()
            .map(|name| format!("  {name}\n"))
            .collect();
        let padding = " ".repeat(column - 2 - HELP_FLAG.len());
        format!(
            "{}\npositional arguments:\n{listing}\noptions:\n  {HELP_FLAG}{padding}show this help message and exit\n",
            self.usage()
        )
    }

    fn error(&self, message: &str) -> CheckReport {
        CheckReport {
            stdout: String::new(),
            stderr: format!("{}{}: error: {message}\n", self.usage(), self.name),
            code: 2,
        }
    }

    /// `-h` prints help and exits 0 unless argparse rejects its explicit
    /// argument; bundled short flags (`-hh`, `-hx`) still reach the help.
    fn help(&self, explicit: Option<&str>, short: bool, equals: bool) -> CheckReport {
        match help_rejection(explicit, short, equals) {
            Some(value) => {
                let message = format!(
                    "argument -h/--help: ignored explicit argument {}",
                    repr(value)
                );
                self.error(&message)
            }
            None => CheckReport::success(self.help_text()),
        }
    }
}

/// The explicit argument argparse refuses, if any.
fn help_rejection(explicit: Option<&str>, short: bool, equals: bool) -> Option<&str> {
    let mut rest = explicit?;
    if !short {
        return Some(rest);
    }
    let mut equals = equals;
    loop {
        if equals || rest.starts_with('-') {
            return Some(rest);
        }
        let tail = rest.strip_prefix('h')?;
        if tail.is_empty() {
            return None;
        }
        rest = tail.strip_prefix('=').unwrap_or(tail);
        equals = tail.starts_with('=');
    }
}

/// argparse's pattern string: every word after the first `--` is positional.
fn tokenize(args: &[String]) -> Vec<Token<'_>> {
    let mut tokens = Vec::with_capacity(args.len());
    let mut separated = false;
    for arg in args {
        let token = if separated {
            Token::Positional
        } else {
            classify(arg)
        };
        separated |= matches!(token, Token::Separator);
        tokens.push(token);
    }
    tokens
}

/// The end of the positional run starting at `start`.
fn run_end(tokens: &[Token<'_>], start: usize) -> usize {
    tokens[start..]
        .iter()
        .position(|token| !matches!(token, Token::Positional | Token::Separator))
        .map_or(tokens.len(), |offset| start + offset)
}

/// argparse's `-*A-*` match against one run: the index of the value and how
/// many words the positional consumed, or `None` when the run holds no value.
fn take_positional(run: &[Token<'_>]) -> Option<(usize, usize)> {
    let value = usize::from(matches!(run.first(), Some(Token::Separator)));
    if !matches!(run.get(value), Some(Token::Positional)) {
        return None;
    }
    let trailing = usize::from(matches!(run.get(value + 1), Some(Token::Separator)));
    Some((value, value + 1 + trailing))
}

fn classify(arg: &str) -> Token<'_> {
    if arg == "--" {
        return Token::Separator;
    }
    if !arg.starts_with('-') || arg.len() == 1 {
        return Token::Positional;
    }
    if arg == "-h" || arg == "--help" {
        return Token::Help {
            explicit: None,
            short: arg == "-h",
            equals: false,
        };
    }
    if let Some((name, value)) = arg.split_once('=')
        && (name == "-h" || name == "--help")
    {
        return Token::Help {
            explicit: Some(value),
            short: name == "-h",
            equals: true,
        };
    }
    if let Some(long) = arg.strip_prefix("--") {
        let (prefix, value) = match long.split_once('=') {
            Some((prefix, value)) => (prefix, Some(value)),
            None => (long, None),
        };
        if "help".starts_with(prefix) {
            return Token::Help {
                explicit: value,
                short: false,
                equals: value.is_some(),
            };
        }
    } else if arg[1..].starts_with('=') {
        return Token::Ambiguous;
    } else if let Some(explicit) = arg.strip_prefix("-h") {
        return Token::Help {
            explicit: Some(explicit),
            short: true,
            equals: false,
        };
    }
    if is_negative_number(arg) || arg.contains(' ') {
        Token::Positional
    } else {
        Token::Unknown
    }
}

/// argparse's `^-\d+$|^-\d*\.\d+$`.
fn is_negative_number(arg: &str) -> bool {
    let digits = |text: &str| text.chars().all(is_decimal);
    let rest = &arg[1..];
    match rest.split_once('.') {
        Some((whole, fraction)) => digits(whole) && !fraction.is_empty() && digits(fraction),
        None => !rest.is_empty() && digits(rest),
    }
}
