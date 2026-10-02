//! Python 3.13 argparse for the two native-policy scripts: single-value
//! options (optionally required or restricted to choices), `store_true`
//! flags, at most one positional, unique-prefix and `=value` options, `-h`
//! bundling, the first `--` separator, and argparse's sequential error order.

use crate::ci_operations::build_cache_options::{Kind, classify, help_flag, is_option_like};
use crate::ci_operations::ci_metrics_int::python_int_text;
use crate::ci_operations::runner_identity_argv::error;
use crate::repository::check_report::CheckReport;
use crate::repository::text::repr;

/// One optional argument of a legacy parser.
pub(super) struct Opt {
    pub(super) name: &'static str,
    pub(super) flag: bool,
    pub(super) required: bool,
    pub(super) choices: &'static [&'static str],
    /// `type=int`: the value is converted (and shown) as a Python int.
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

/// A legacy parser: program name, usage block, help text and arguments.
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

    fn names(&self) -> Vec<&'static str> {
        let mut names = vec!["-h", "--help"];
        names.extend(self.options.iter().map(|option| option.name));
        names
    }

    /// Parses `args`, or returns the help/usage report argparse would produce.
    pub(super) fn parse(&self, args: &[String]) -> Result<Parsed, CheckReport> {
        let (parsed, extras) = self.parse_known(args)?;
        if !extras.is_empty() {
            return Err(self.fail(&format!("unrecognized arguments: {}", extras.join(" "))));
        }
        Ok(parsed)
    }

    /// `parse_known_args`: the parsed values and the unrecognized words.
    pub(super) fn parse_known(
        &self,
        args: &[String],
    ) -> Result<(Parsed, Vec<String>), CheckReport> {
        let names = self.names();
        let mut parsed = Parsed::default();
        let mut extras: Vec<&str> = Vec::new();
        let mut separated = false;
        let mut index = 0;
        while let Some(arg) = args.get(index) {
            let kind = classify(arg, &names);
            if separated || arg == "--" || matches!(kind, Kind::Positional) {
                index = self.positional_run(args, index, &mut separated, &mut parsed, &mut extras);
                continue;
            }
            index += 1;
            match kind {
                Kind::Positional | Kind::Unknown => extras.push(arg),
                Kind::Known("-h" | "--help", explicit, sep) => {
                    help_flag(explicit, sep, "-h/--help", &|message| self.fail(message))?;
                    return Err(CheckReport::success(self.help.to_owned()));
                }
                Kind::Known(name, explicit, sep) => {
                    let option = self.option(name);
                    let value = if option.flag {
                        help_flag(explicit, sep, name, &|message| self.fail(message))?;
                        String::new()
                    } else {
                        self.take_value(option, explicit, args, &mut index, &names)?
                    };
                    parsed.values.push((option.name, value));
                }
            }
        }
        let extras = extras.into_iter().map(str::to_owned).collect();
        Ok((self.finish(parsed)?, extras))
    }

    fn option(&self, name: &str) -> &Opt {
        self.options
            .iter()
            .find(|option| option.name == name)
            .unwrap_or(&self.options[0])
    }

    /// Consumes one run of positional words (and the first `--`), filling
    /// the positional with argparse's `-*A-*` match; the rest are extras.
    fn positional_run<'a>(
        &self,
        args: &'a [String],
        start: usize,
        separated: &mut bool,
        parsed: &mut Parsed,
        extras: &mut Vec<&'a str>,
    ) -> usize {
        let mut separators = Vec::new();
        let mut end = start;
        while let Some(arg) = args.get(end) {
            let separator = !*separated && arg == "--";
            if !(*separated
                || separator
                || matches!(classify(arg, &self.names()), Kind::Positional))
            {
                break;
            }
            *separated |= separator;
            separators.push(separator);
            end += 1;
        }
        let mut consumed = 0;
        if self.positional.is_some() && parsed.positional.is_none() {
            let value = usize::from(separators.first() == Some(&true));
            if separators.get(value) == Some(&false) {
                parsed.positional = Some(args[start + value].clone());
                let trailing = usize::from(separators.get(value + 1) == Some(&true));
                consumed = value + 1 + trailing;
            }
        }
        extras.extend(args[start + consumed..end].iter().map(String::as_str));
        end
    }

    fn take_value(
        &self,
        option: &Opt,
        explicit: Option<String>,
        args: &[String],
        index: &mut usize,
        names: &[&str],
    ) -> Result<String, CheckReport> {
        let value = match explicit {
            Some(value) => value,
            None => match args.get(*index) {
                Some(next) if !is_option_like(next, names) => {
                    *index += 1;
                    next.clone()
                }
                _ => {
                    let message = format!("argument {}: expected one argument", option.name);
                    return Err(self.fail(&message));
                }
            },
        };
        let value = if option.integer {
            python_int_text(&value).ok_or_else(|| {
                self.fail(&format!(
                    "argument {}: invalid int value: {}",
                    option.name,
                    repr(&value)
                ))
            })?
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
