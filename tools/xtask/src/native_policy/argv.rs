//! Exact native-policy options, ordered repeated values, and an optional binary path.

use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;
use crate::repository::text::repr;

/// One native-policy option.
pub(super) struct Opt {
    pub(super) name: &'static str,
    pub(super) flag: bool,
    pub(super) required: bool,
    pub(super) choices: &'static [&'static str],
    /// A bounded unsigned integer, used for CUDA toolkit major choices.
    pub(super) integer: bool,
}

impl Opt {
    pub(super) const fn value(name: &'static str) -> Self {
        Self {
            name,
            flag: false,
            required: false,
            choices: &[],
            integer: false,
        }
    }

    pub(super) const fn required(name: &'static str) -> Self {
        Self {
            required: true,
            ..Self::value(name)
        }
    }

    pub(super) const fn flag(name: &'static str) -> Self {
        Self {
            flag: true,
            ..Self::value(name)
        }
    }

    pub(super) const fn choice(name: &'static str, choices: &'static [&'static str]) -> Self {
        Self {
            choices,
            ..Self::value(name)
        }
    }

    pub(super) const fn int_choice(name: &'static str, choices: &'static [&'static str]) -> Self {
        Self {
            integer: true,
            required: true,
            ..Self::choice(name, choices)
        }
    }
}

/// The native-policy command grammar and its usage output.
pub(super) struct Grammar {
    pub(super) prog: &'static str,
    pub(super) usage: &'static str,
    pub(super) help: &'static str,
    pub(super) options: &'static [Opt],
    pub(super) positional: Option<&'static str>,
}

/// Parsed values: the last value of each option, set flags, the positional.
#[derive(Default)]
pub(super) struct Parsed {
    values: Vec<(&'static str, String)>,
    pub(super) positional: Option<String>,
}

impl Parsed {
    pub(super) fn value(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .rev()
            .find(|(option, _)| *option == name)
            .map(|(_, value)| value.as_str())
    }

    /// Every value of an `action="append"` option, in order.
    pub(super) fn values(&self, name: &str) -> Vec<&str> {
        self.values
            .iter()
            .filter(|(option, _)| *option == name)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    pub(super) fn flag(&self, name: &str) -> bool {
        self.value(name).is_some()
    }
}

impl Grammar {
    fn fail(&self, message: &str) -> CheckReport {
        error(self.usage, self.prog, message)
    }

    pub(super) fn parse(&self, args: &[String]) -> Result<Parsed, CheckReport> {
        let (parsed, extras) = self.parse_known(args)?;
        if !extras.is_empty() {
            return Err(self.fail(&format!("unrecognized arguments: {}", extras.join(" "))));
        }
        Ok(parsed)
    }

    pub(super) fn parse_known(
        &self,
        args: &[String],
    ) -> Result<(Parsed, Vec<String>), CheckReport> {
        if args.len() > 4096 || args.iter().map(String::len).sum::<usize>() > 1024 * 1024 {
            return Err(self.fail("native-policy arguments exceed input limits"));
        }
        let mut parsed = Parsed::default();
        let mut extras = Vec::new();
        let mut separated = false;
        let mut index = 0;
        while let Some(arg) = args.get(index) {
            index += 1;
            if !separated && arg == "--" {
                separated = true;
                continue;
            }
            if separated || !option_like(arg) {
                if self.positional.is_some() && parsed.positional.is_none() {
                    parsed.positional = Some(arg.clone());
                } else {
                    extras.push(arg.clone());
                }
                continue;
            }
            let (name, explicit) = arg
                .split_once('=')
                .map_or((arg.as_str(), None), |(n, v)| (n, Some(v)));
            if matches!(name, "-h" | "--help") {
                if explicit.is_some() {
                    return Err(self.fail("help does not accept a value"));
                }
                return Err(CheckReport::success(self.help.to_owned()));
            }
            let Some(option) = self.options.iter().find(|option| option.name == name) else {
                extras.push(arg.clone());
                continue;
            };
            let value = self.take_value(option, explicit, args, &mut index)?;
            parsed.values.push((option.name, value));
        }
        Ok((self.finish(parsed)?, extras))
    }

    fn take_value(
        &self,
        option: &Opt,
        explicit: Option<&str>,
        args: &[String],
        index: &mut usize,
    ) -> Result<String, CheckReport> {
        if option.flag {
            return if explicit.is_none() {
                Ok(String::new())
            } else {
                Err(self.fail(&format!("argument {} does not accept a value", option.name)))
            };
        }
        let value = match explicit {
            Some(value) => value.to_owned(),
            None => match args.get(*index) {
                Some(next) if !option_like(next) => {
                    *index += 1;
                    next.clone()
                }
                _ => {
                    return Err(
                        self.fail(&format!("argument {}: expected one argument", option.name))
                    );
                }
            },
        };
        let value = if option.integer {
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(self.fail(&format!(
                    "argument {}: expected unsigned integer",
                    option.name
                )));
            }
            value
                .parse::<u32>()
                .map_err(|_| self.fail(&format!("argument {}: integer out of range", option.name)))?
                .to_string()
        } else {
            value
        };
        if !option.choices.is_empty() && !option.choices.contains(&value.as_str()) {
            let choices: Vec<String> = option.choices.iter().map(|name| repr(name)).collect();
            return Err(self.fail(&format!(
                "argument {}: invalid choice: {} (choose from {})",
                option.name,
                repr(&value),
                choices.join(", ")
            )));
        }
        Ok(value)
    }

    fn finish(&self, parsed: Parsed) -> Result<Parsed, CheckReport> {
        let mut missing: Vec<&str> = self
            .options
            .iter()
            .filter(|option| option.required && parsed.value(option.name).is_none())
            .map(|option| option.name)
            .collect();
        if let Some(name) = self.positional
            && parsed.positional.is_none()
        {
            missing.push(name);
        }
        if !missing.is_empty() {
            return Err(self.fail(&format!(
                "the following arguments are required: {}",
                missing.join(", ")
            )));
        }
        Ok(parsed)
    }
}

fn option_like(value: &str) -> bool {
    value.starts_with('-') && value != "-"
}

#[cfg(test)]
mod tests {
    use super::{Grammar, Opt};

    const GRAMMAR: Grammar = Grammar {
        prog: "native-policy",
        usage: "native-policy [options] binary",
        help: "native-policy help\n",
        options: &[
            Opt::value("--scan-dir"),
            Opt::int_choice("--cuda-major", &["12", "13"]),
            Opt::flag("--enabled"),
        ],
        positional: Some("binary"),
    };

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    #[test]
    fn exact_options_keep_repeated_unicode_paths_and_separator() {
        let parsed = GRAMMAR
            .parse(&args(&[
                "--scan-dir",
                "模型 one",
                "--cuda-major=12",
                "--scan-dir=second",
                "--enabled",
                "--",
                "-binary",
            ]))
            .unwrap_or_else(|error| panic!("{}", error.stderr));
        assert_eq!(parsed.values("--scan-dir"), ["模型 one", "second"]);
        assert_eq!(parsed.value("--cuda-major"), Some("12"));
        assert!(parsed.flag("--enabled"));
        assert_eq!(parsed.positional.as_deref(), Some("-binary"));
    }

    #[test]
    fn compatibility_spellings_and_missing_values_are_rejected() {
        for words in [
            vec!["--cuda-major", "+12", "binary"],
            vec!["--cuda-major", "1_2", "binary"],
            vec!["--cuda-major", "１２", "binary"],
            vec!["--cuda-major", "4294967296", "binary"],
            vec!["--cuda-major", "12", "--scan-dir", "--enabled", "binary"],
            vec!["--cuda-major", "12", "--ena", "binary"],
            vec!["--cuda-major", "12", "-hh", "binary"],
            vec!["--cuda-major", "12", "--enabled=true", "binary"],
            vec!["--cuda-major", "12", "binary", "--", "--"],
        ] {
            assert!(GRAMMAR.parse(&args(&words)).is_err(), "{words:?}");
        }
        assert!(GRAMMAR.parse(&vec!["x".to_owned(); 4097]).is_err());
    }
}
