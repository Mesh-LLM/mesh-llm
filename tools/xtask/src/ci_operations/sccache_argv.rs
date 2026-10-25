//! Typed arguments for `ci-ops sccache-stats`.

use crate::repository::check_report::CheckReport;

const PROG: &str = "cargo xtool ci-ops sccache-stats";
const USAGE: &str = "usage: cargo xtool ci-ops sccache-stats --artifact-name <name> \
                     --output <path> [--github-output <path>] \
                     [--cache-expectation <cold|warm|opportunistic>] \
                     [--minimum-hit-rate <rate>]\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Expectation {
    Cold,
    Warm,
    Opportunistic,
}

impl Expectation {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
            Self::Opportunistic => "opportunistic",
        }
    }
}

#[derive(Debug)]
pub(crate) struct Args {
    pub(crate) artifact_name: String,
    pub(crate) output: String,
    pub(crate) github_output: Option<String>,
    pub(crate) expectation: Expectation,
    pub(crate) minimum_hit_rate: f64,
}

fn fail(message: &str) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{USAGE}{PROG}: error: {message}\n"),
        code: 2,
    }
}

fn help() -> String {
    format!(
        "{USAGE}\nOptions:\n  -h, --help\n  --artifact-name <name>\n  --output <path>\n  \
         --github-output <path>\n  --cache-expectation <cold|warm|opportunistic>\n  \
         --minimum-hit-rate <rate>\n"
    )
}

#[derive(Default)]
struct Seen {
    artifact_name: Option<String>,
    output: Option<String>,
    github_output: Option<String>,
    expectation: Option<Expectation>,
    minimum_hit_rate: Option<f64>,
}

pub(crate) fn parse(args: &[String]) -> Result<Args, CheckReport> {
    let mut seen = Seen::default();
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        index += 1;
        if arg == "-h" || arg == "--help" {
            return Err(CheckReport::success(help()));
        }

        let (option, attached) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(name, value)| {
                (name, Some(value.to_owned()))
            });
        match option {
            "--artifact-name" => {
                seen.artifact_name = Some(take_value(option, attached, args, &mut index, false)?);
            }
            "--output" => {
                seen.output = Some(take_value(option, attached, args, &mut index, false)?);
            }
            "--github-output" => {
                seen.github_output = Some(take_value(option, attached, args, &mut index, false)?);
            }
            "--cache-expectation" => {
                let value = take_value(option, attached, args, &mut index, false)?;
                seen.expectation = Some(match value.as_str() {
                    "cold" => Expectation::Cold,
                    "warm" => Expectation::Warm,
                    "opportunistic" => Expectation::Opportunistic,
                    _ => {
                        return Err(fail(
                            "--cache-expectation must be cold, warm, or opportunistic",
                        ));
                    }
                });
            }
            "--minimum-hit-rate" => {
                let value = take_value(option, attached, args, &mut index, true)?;
                seen.minimum_hit_rate = Some(value.parse().map_err(|_| {
                    fail("--minimum-hit-rate must be a decimal number between 0 and 1")
                })?);
            }
            _ => return Err(fail(&format!("unknown argument: {arg}"))),
        }
    }

    let mut missing = Vec::new();
    if seen.artifact_name.is_none() {
        missing.push("--artifact-name");
    }
    if seen.output.is_none() {
        missing.push("--output");
    }
    if !missing.is_empty() {
        return Err(fail(&format!(
            "required arguments missing: {}",
            missing.join(", ")
        )));
    }

    Ok(Args {
        artifact_name: seen.artifact_name.unwrap_or_default(),
        output: seen.output.unwrap_or_default(),
        github_output: seen.github_output,
        expectation: seen.expectation.unwrap_or(Expectation::Opportunistic),
        minimum_hit_rate: seen.minimum_hit_rate.unwrap_or(0.0),
    })
}

fn take_value(
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
        return Err(fail(&format!("{option} requires a value")));
    };
    if value == "--" || (value.starts_with('-') && !allow_negative) {
        return Err(fail(&format!("{option} requires a value")));
    }
    *index += 1;
    Ok(value.clone())
}

#[cfg(test)]
mod tests {
    use super::{Expectation, parse};

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_the_capture_action_options_and_defaults() {
        let parsed = parse(&args(&[
            "--artifact-name",
            "build-linux",
            "--cache-expectation",
            "warm",
            "--minimum-hit-rate=0.75",
            "--output",
            "stats.json",
            "--github-output",
            "outputs.txt",
        ]))
        .expect("valid capture arguments");
        assert_eq!(parsed.artifact_name, "build-linux");
        assert_eq!(parsed.output, "stats.json");
        assert_eq!(parsed.github_output.as_deref(), Some("outputs.txt"));
        assert_eq!(parsed.expectation, Expectation::Warm);
        assert_eq!(parsed.minimum_hit_rate, 0.75);

        let defaults = parse(&args(&["--artifact-name", "a", "--output", "o"]))
            .expect("required capture arguments");
        assert_eq!(defaults.expectation, Expectation::Opportunistic);
        assert_eq!(defaults.minimum_hit_rate, 0.0);
    }

    #[test]
    fn rejects_abbreviations_and_malformed_typed_values() {
        for input in [
            args(&["--artifact", "a", "--output", "o"]),
            args(&[
                "--artifact-name",
                "a",
                "--output",
                "o",
                "--cache-expectation",
                "other",
            ]),
            args(&[
                "--artifact-name",
                "a",
                "--output",
                "o",
                "--minimum-hit-rate",
                "0_5",
            ]),
        ] {
            let error = parse(&input).expect_err("malformed arguments must be rejected");
            assert_eq!(error.code, 2);
            assert!(error.stdout.is_empty());
            assert!(!error.stderr.is_empty());
        }
    }

    #[test]
    fn help_shows_supported_flags() {
        let error = parse(&args(&["--help"])).expect_err("help is returned as a report");
        assert_eq!(error.code, 0);
        assert!(error.stdout.contains("--cache-expectation"));
        assert!(error.stdout.contains("--minimum-hit-rate"));
    }
}
