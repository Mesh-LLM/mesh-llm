//! Frozen comparison flags with bounded native lifecycle budgets.
use super::plan::{self, Mode, Side, Spec};
use crate::command::DynResult;
use std::{collections::BTreeMap, path::PathBuf};

pub(super) const USAGE: &str = "cargo xtool automation event-benchmark-run --binary PATH [--baseline-binary PATH] --model LOCAL_GGUF --output-dir PATH --pairs-primary N --pairs-scenario N --seed U64 --mode production|event-disabled|off [--mode MODE] --scenario LABEL [--scenario LABEL] [--attempt 1|2] [--max-tokens N] [--readiness-timeout-secs N] [--request-timeout-secs N] [--shutdown-timeout-secs N] [--execution-timeout-secs N]";
const REQUIRED: &[&str] = &[
    "--binary",
    "--model",
    "--output-dir",
    "--pairs-primary",
    "--pairs-scenario",
    "--seed",
];
const OPTIONAL: &[&str] = &[
    "--baseline-binary",
    "--attempt",
    "--max-tokens",
    "--readiness-timeout-secs",
    "--request-timeout-secs",
    "--shutdown-timeout-secs",
    "--execution-timeout-secs",
];

pub(super) struct Command {
    pub sides: [Side; 2],
    pub spec: Spec,
    pub model: PathBuf,
    pub output_dir: PathBuf,
    pub attempt: u64,
    pub max_tokens: u64,
    pub readiness_secs: u64,
    pub request_secs: u64,
    pub shutdown_secs: u64,
    pub execution_secs: u64,
}

fn mode(value: &str) -> DynResult<Mode> {
    match value {
        "production" => Ok(Mode::Production),
        "event-disabled" => Ok(Mode::EventDisabled),
        "off" => Ok(Mode::Off),
        _ => Err("benchmark mode must be production, event-disabled or off".into()),
    }
}

fn number(values: &BTreeMap<&str, &str>, flag: &str, default: Option<u64>) -> DynResult<u64> {
    match values.get(flag) {
        Some(value) => Ok(value.parse()?),
        None => default.ok_or_else(|| format!("missing required benchmark option {flag}").into()),
    }
}

impl Command {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        if args.len() > 256
            || !args.len().is_multiple_of(2)
            || args.iter().any(|arg| arg.len() > 16 * 1024)
        {
            return Err("benchmark options require bounded flag/value pairs".into());
        }
        let mut values = BTreeMap::new();
        let mut modes = Vec::new();
        let mut scenarios = Vec::new();
        for pair in args.as_chunks::<2>().0 {
            let flag = pair[0].as_str();
            let value = pair[1].as_str();
            match flag {
                "--mode" => modes.push(mode(value)?),
                "--scenario" => scenarios.push(value.into()),
                _ if REQUIRED.contains(&flag) || OPTIONAL.contains(&flag) => {
                    if values.insert(flag, value).is_some() || value.is_empty() {
                        return Err(format!("duplicate or empty benchmark option {flag}").into());
                    }
                }
                _ => return Err(format!("unknown benchmark option {flag}").into()),
            }
        }
        for flag in REQUIRED {
            if !values.contains_key(flag) {
                return Err(format!("missing required benchmark option {flag}").into());
            }
        }
        let spec = Spec {
            seed: number(&values, "--seed", None)?,
            pairs_primary: usize::try_from(number(&values, "--pairs-primary", None)?)?,
            pairs_scenario: usize::try_from(number(&values, "--pairs-scenario", None)?)?,
            scenarios,
        };
        spec.validate()?;
        let sides = plan::sides(
            values["--binary"].into(),
            values
                .get("--baseline-binary")
                .map(|value| PathBuf::from(*value)),
            &modes,
        )?;
        let result = Self {
            sides,
            spec,
            model: values["--model"].into(),
            output_dir: values["--output-dir"].into(),
            attempt: number(&values, "--attempt", Some(1))?,
            max_tokens: number(&values, "--max-tokens", Some(64))?,
            readiness_secs: number(&values, "--readiness-timeout-secs", Some(120))?,
            request_secs: number(&values, "--request-timeout-secs", Some(120))?,
            shutdown_secs: number(&values, "--shutdown-timeout-secs", Some(15))?,
            execution_secs: number(&values, "--execution-timeout-secs", Some(86400))?,
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> DynResult<()> {
        if !(1..=2).contains(&self.attempt)
            || !(1..=4096).contains(&self.max_tokens)
            || [
                self.readiness_secs,
                self.request_secs,
                self.shutdown_secs,
                self.execution_secs,
            ]
            .iter()
            .any(|n| !(1..=86400).contains(n))
        {
            return Err("benchmark attempts are limited to1/2; token and lifecycle budgets must be bounded positive values".into());
        }
        let (execution, cleanup) = super::lifecycle_budget::cell(
            std::time::Duration::from_secs(self.readiness_secs),
            std::time::Duration::from_secs(self.request_secs),
            std::time::Duration::from_secs(self.shutdown_secs),
        )?;
        if execution + cleanup >= std::time::Duration::from_secs(self.execution_secs) {
            return Err(
                "matrix deadline must cover at least one complete measurement worker and cleanup"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "options_tests.rs"]
mod tests;
