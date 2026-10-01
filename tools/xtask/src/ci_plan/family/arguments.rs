use super::shard_count::ShardCount;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;
use std::path::PathBuf;

const USAGE: &str = "cargo xtool ci family-plan [--manifest PATH] [--families LABELS] [--shard-count N] [--output PATH] [--github-output PATH] [--verify-plan PATH]";
const OPTIONS: [(&str, OptionName); 7] = [
    ("--manifest", OptionName::Manifest),
    ("--families", OptionName::Families),
    ("--shard-count", OptionName::ShardCount),
    ("--output", OptionName::Output),
    ("--github-output", OptionName::GithubOutput),
    ("--verify-plan", OptionName::VerifyPlan),
    ("--help", OptionName::Help),
];

#[derive(Clone, Copy)]
enum OptionName {
    Manifest,
    Families,
    ShardCount,
    Output,
    GithubOutput,
    VerifyPlan,
    Help,
}

pub(super) struct Arguments<'a> {
    pub(super) manifest: Option<PathBuf>,
    pub(super) families: &'a str,
    pub(super) count: ShardCount,
    pub(super) output: Option<PathBuf>,
    pub(super) github_output: Option<PathBuf>,
    pub(super) verify_plan: Option<PathBuf>,
}

pub(super) enum Parsed<'a> {
    Help,
    Options(Arguments<'a>),
}

enum Token<'a> {
    Value,
    Unknown,
    End,
    Option {
        option: OptionName,
        name: &'static str,
        inline: Option<&'a str>,
    },
}

pub(super) fn parse(args: &[String]) -> Result<Parsed<'_>, CheckReport> {
    let mut parsed = Arguments {
        manifest: None,
        families: "",
        count: ShardCount::from(1),
        output: None,
        github_output: None,
        verify_plan: None,
    };
    let mut extras = Vec::new();
    let mut rest = args.iter().peekable();
    while let Some(arg) = rest.next() {
        match classify(arg) {
            Token::End => {
                extras.push(arg.as_str());
                extras.extend(rest.map(String::as_str));
                break;
            }
            Token::Unknown | Token::Value => extras.push(arg.as_str()),
            Token::Option {
                option: OptionName::Help,
                inline,
                ..
            } => {
                if let Some(value) = inline {
                    return Err(error(&format!(
                        "argument -h/--help: ignored explicit argument {}",
                        repr(value)
                    )));
                }
                return Ok(Parsed::Help);
            }
            Token::Option {
                option,
                name,
                inline,
            } => {
                let value = match inline {
                    Some(value) => value,
                    None => rest
                        .next_if(|next| matches!(classify(next), Token::Value))
                        .map(String::as_str)
                        .ok_or_else(|| error(&format!("argument {name}: expected one argument")))?,
                };
                parsed.store(option, value)?;
            }
        }
    }
    if !extras.is_empty() {
        return Err(error(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    Ok(Parsed::Options(parsed))
}

impl<'a> Arguments<'a> {
    fn store(&mut self, option: OptionName, value: &'a str) -> Result<(), CheckReport> {
        match option {
            OptionName::Manifest => self.manifest = Some(path(value)),
            OptionName::Families => self.families = value,
            OptionName::ShardCount => {
                self.count = ShardCount::parse(value).ok_or_else(|| {
                    error(&format!(
                        "argument --shard-count: invalid int value: {}",
                        repr(value)
                    ))
                })?;
            }
            OptionName::Output => self.output = Some(path(value)),
            OptionName::GithubOutput => self.github_output = Some(path(value)),
            OptionName::VerifyPlan => self.verify_plan = Some(path(value)),
            OptionName::Help => return Err(error("argument -h/--help: ignored explicit argument")),
        }
        Ok(())
    }
}

fn path(value: &str) -> PathBuf {
    PathBuf::from(if value.is_empty() { "." } else { value })
}

fn classify(arg: &str) -> Token<'_> {
    if arg == "--" {
        return Token::End;
    }
    if arg == "-h" {
        return Token::Option {
            option: OptionName::Help,
            name: "--help",
            inline: None,
        };
    }
    let (name, inline) = arg
        .split_once('=')
        .map_or((arg, None), |(name, value)| (name, Some(value)));
    if name.starts_with("--")
        && name.len() > 2
        && let Some((name, option)) = OPTIONS.into_iter().find(|(option, _)| *option == name)
    {
        return Token::Option {
            option,
            name,
            inline,
        };
    }
    if let Some(tail) = arg.strip_prefix("-h") {
        let tail = tail.trim_start_matches('h');
        let inline = if tail.starts_with(['=', '-']) {
            Some(tail.strip_prefix('=').unwrap_or(tail))
        } else {
            None
        };
        return Token::Option {
            option: OptionName::Help,
            name: "--help",
            inline,
        };
    }
    let negative = arg.strip_prefix('-').is_some_and(|rest| {
        rest.strip_prefix('.')
            .unwrap_or(rest)
            .chars()
            .next()
            .filter(char::is_ascii_digit)
            .is_some()
    });
    if !arg.starts_with('-') || arg == "-" || negative || arg.contains(' ') {
        Token::Value
    } else {
        Token::Unknown
    }
}

fn error(message: &str) -> CheckReport {
    CheckReport::usage(USAGE, message)
}

pub(super) fn help() -> CheckReport {
    CheckReport::success(format!(
        "usage: {USAGE}\n\nGenerate or verify a deterministic family policy plan.\n\nOptions:\n  -h, --help             Show this help\n  --manifest PATH        Family manifest (default: selected root's ci/llama-canary/family-certified.json)\n  --families LABELS      Unique comma-separated family labels (default: all)\n  --shard-count N        Positive shard count (default: 1)\n  --output PATH          Write the plan instead of stdout\n  --github-output PATH   Append GitHub outputs; requires --output\n  --verify-plan PATH     Recompute and verify an existing policy plan\n\nCache checking and GGUF inspection remain Python-owned.\n"
    ))
}
