//! argparse emulation for `.github/actions/capture-sccache-stats/capture.py`:
//! required `--artifact-name`/`--output`, optional `--github-output`,
//! `--cache-expectation {cold,warm,opportunistic}` and a `float`
//! `--minimum-hit-rate`, with unique-prefix options, `=value`, `-h`
//! bundling, `--` handling and Python 3.13's sequential error order.

use crate::ci_operations::build_cache_options::{
    Kind, ambiguous, classify, help_flag, is_option_like,
};
use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::{repr, strip};

const PROG: &str = "capture.py";
const USAGE: &str = "\
usage: capture.py [-h] --artifact-name ARTIFACT_NAME --output OUTPUT
                  [--github-output GITHUB_OUTPUT]
                  [--cache-expectation {cold,warm,opportunistic}]
                  [--minimum-hit-rate MINIMUM_HIT_RATE]
";
const OPTIONS: [&str; 7] = [
    "-h",
    "--help",
    "--artifact-name",
    "--output",
    "--github-output",
    "--cache-expectation",
    "--minimum-hit-rate",
];
const EXPECTATIONS: [&str; 3] = ["cold", "warm", "opportunistic"];

pub(crate) struct Args {
    pub(crate) artifact_name: String,
    pub(crate) output: String,
    pub(crate) github_output: Option<String>,
    pub(crate) expectation: &'static str,
    pub(crate) minimum_hit_rate: f64,
}

fn fail(message: &str) -> CheckReport {
    error(USAGE, PROG, message)
}

fn help() -> String {
    format!(
        "{USAGE}\noptions:\n  -h, --help            show this help message and exit\n  \
         --artifact-name ARTIFACT_NAME\n  --output OUTPUT\n  --github-output GITHUB_OUTPUT\n  \
         --cache-expectation {{cold,warm,opportunistic}}\n  \
         --minimum-hit-rate MINIMUM_HIT_RATE\n"
    )
}

#[derive(Default)]
struct Seen {
    artifact_name: Option<String>,
    output: Option<String>,
    github_output: Option<String>,
    expectation: Option<&'static str>,
    minimum_hit_rate: Option<f64>,
}

/// Parses `args`, or returns the help/usage report argparse would produce.
pub(crate) fn parse(args: &[String]) -> Result<Args, CheckReport> {
    let mut seen = Seen::default();
    let mut extras: Vec<String> = Vec::new();
    let mut positional_only = false;
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        index += 1;
        if positional_only || arg == "--" {
            positional_only = true;
            extras.push(arg.clone());
            continue;
        }
        match classify(arg, &OPTIONS) {
            Kind::Positional | Kind::Unknown => extras.push(arg.clone()),
            Kind::Ambiguous(names) => return Err(ambiguous(arg, &names, &fail)),
            Kind::Known("-h" | "--help", explicit, sep) => {
                help_flag(explicit, sep, "-h/--help", &fail)?;
                return Err(CheckReport::success(help()));
            }
            Kind::Known(option, explicit, _) => {
                let value = take_value(explicit, args, &mut index, option)?;
                apply(&mut seen, option, value)?;
            }
        }
    }
    finish(seen, &extras)
}

fn take_value(
    explicit: Option<String>,
    args: &[String],
    index: &mut usize,
    option: &str,
) -> Result<String, CheckReport> {
    if let Some(value) = explicit {
        return Ok(value);
    }
    match args.get(*index) {
        Some(next) if !is_option_like(next, &OPTIONS) => {
            *index += 1;
            Ok(next.clone())
        }
        _ => Err(fail(&format!("argument {option}: expected one argument"))),
    }
}

fn apply(seen: &mut Seen, option: &str, value: String) -> Result<(), CheckReport> {
    match option {
        "--artifact-name" => seen.artifact_name = Some(value),
        "--output" => seen.output = Some(value),
        "--github-output" => seen.github_output = Some(value),
        "--cache-expectation" => {
            let Some(choice) = EXPECTATIONS.iter().find(|choice| **choice == value) else {
                let choices: Vec<String> = EXPECTATIONS.iter().map(|name| repr(name)).collect();
                return Err(fail(&format!(
                    "argument {option}: invalid choice: {} (choose from {})",
                    repr(&value),
                    choices.join(", ")
                )));
            };
            seen.expectation = Some(choice);
        }
        _ => {
            let Some(rate) = python_float(&value) else {
                return Err(fail(&format!(
                    "argument {option}: invalid float value: {}",
                    repr(&value)
                )));
            };
            seen.minimum_hit_rate = Some(rate);
        }
    }
    Ok(())
}

fn finish(seen: Seen, extras: &[String]) -> Result<Args, CheckReport> {
    let missing: Vec<&str> = [
        ("--artifact-name", seen.artifact_name.is_none()),
        ("--output", seen.output.is_none()),
    ]
    .into_iter()
    .filter_map(|(name, absent)| absent.then_some(name))
    .collect();
    if !missing.is_empty() {
        return Err(fail(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        )));
    }
    if !extras.is_empty() {
        return Err(fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    Ok(Args {
        artifact_name: seen.artifact_name.unwrap_or_default(),
        output: seen.output.unwrap_or_default(),
        github_output: seen.github_output,
        expectation: seen.expectation.unwrap_or("opportunistic"),
        minimum_hit_rate: seen.minimum_hit_rate.unwrap_or(0.0),
    })
}

/// Python `float(text)`: surrounding whitespace ignored, underscores only
/// between digits, `inf`/`infinity`/`nan` in any case with a sign.
pub(crate) fn python_float(text: &str) -> Option<f64> {
    let text = strip(text);
    let chars: Vec<char> = text.chars().collect();
    let digit = |index: Option<usize>| {
        index
            .and_then(|index| chars.get(index))
            .is_some_and(char::is_ascii_digit)
    };
    for (index, ch) in chars.iter().enumerate() {
        if *ch == '_' && !(digit(index.checked_sub(1)) && digit(Some(index + 1))) {
            return None;
        }
    }
    let cleaned: String = chars.iter().filter(|ch| **ch != '_').collect();
    if !cleaned.is_ascii() {
        return None;
    }
    cleaned.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::python_float;

    #[test]
    fn migration_ci_operations_python_float_follows_cpython() {
        assert_eq!(python_float(" 0.5\n"), Some(0.5));
        assert_eq!(python_float("1_0"), Some(10.0));
        assert_eq!(python_float("-iNfInity"), Some(f64::NEG_INFINITY));
        assert!(python_float("+nan").is_some_and(f64::is_nan));
        for rejected in ["", "_1", "1_", "1__0", "1_.5", "x", "0x10", "1e"] {
            assert_eq!(python_float(rejected), None, "{rejected}");
        }
    }
}
