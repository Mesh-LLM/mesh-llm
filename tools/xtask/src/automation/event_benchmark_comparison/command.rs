//! Command admission, bounded local inputs and atomic comparison reports.
use super::{manifest, options::Options, report};
use crate::command::DynResult;
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

pub(super) const USAGE: &str = "cargo xtool automation event-benchmark-compare --production FILE --event-disabled FILE --baseline FILE --output FILE --bootstrap-samples N --seed N --max-degradation-percent PERCENT --min-primary-pairs N --min-scenario-pairs N --max-mdd-percent PERCENT [--report-holm]";
const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const VALUE_FLAGS: [&str; 10] = [
    "--production",
    "--event-disabled",
    "--baseline",
    "--output",
    "--bootstrap-samples",
    "--seed",
    "--max-degradation-percent",
    "--min-primary-pairs",
    "--min-scenario-pairs",
    "--max-mdd-percent",
];

pub(super) struct Command {
    production: PathBuf,
    reference: PathBuf,
    baseline: PathBuf,
    output: PathBuf,
    options: Options,
}

impl Command {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        let mut values = BTreeMap::new();
        let mut holm = false;
        let mut rest = args;
        while let Some((flag, tail)) = rest.split_first() {
            if flag == "--report-holm" {
                if holm {
                    return Err("duplicate --report-holm".into());
                }
                holm = true;
                rest = tail;
                continue;
            }
            if !VALUE_FLAGS.contains(&flag.as_str()) {
                return Err(format!("unknown comparison option {flag}").into());
            }
            let (value, tail) = tail
                .split_first()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            if value.starts_with("--")
                || value.is_empty()
                || values.insert(flag.as_str(), value.as_str()).is_some()
            {
                return Err(format!("missing or repeated comparison option {flag}").into());
            }
            rest = tail;
        }
        for flag in VALUE_FLAGS {
            if !values.contains_key(flag) {
                return Err(format!("required comparison option {flag}").into());
            }
        }
        Ok(Self {
            production: values["--production"].into(),
            reference: values["--event-disabled"].into(),
            baseline: values["--baseline"].into(),
            output: values["--output"].into(),
            options: Options {
                seed: values["--seed"].parse()?,
                bootstrap_samples: values["--bootstrap-samples"].parse()?,
                min_primary_pairs: values["--min-primary-pairs"].parse()?,
                min_scenario_pairs: values["--min-scenario-pairs"].parse()?,
                max_degradation_percent: values["--max-degradation-percent"].parse()?,
                max_mdd_percent: values["--max-mdd-percent"].parse()?,
                report_holm: holm,
            },
        })
    }

    pub fn execute(&self) -> DynResult<serde_json::Value> {
        let inputs = [&self.production, &self.reference, &self.baseline];
        if self.output.exists() {
            let output = fs::canonicalize(&self.output)?;
            for input in inputs {
                if fs::canonicalize(input)? == output {
                    return Err("comparison output cannot overwrite an input manifest".into());
                }
            }
        }
        let production = read_manifest(&self.production)?;
        let reference = read_manifest(&self.reference)?;
        let baseline = read_manifest(&self.baseline)?;
        let report = report::build(&production, &reference, &baseline, &self.options)?;
        let mut bytes = serde_json::to_vec_pretty(&report)?;
        bytes.push(b'\n');
        let parent = self
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let mut staged = tempfile::NamedTempFile::new_in(parent)?;
        staged.write_all(&bytes)?;
        staged.as_file().sync_all()?;
        staged.persist(&self.output)?;
        Ok(
            serde_json::json!({"output_path":self.output,"certification_status":report["certification_status"]}),
        )
    }
}

fn read_manifest(path: &Path) -> DynResult<manifest::Manifest> {
    if !fs::metadata(path)?.is_file() {
        return Err("comparison inputs must be regular local files".into());
    }
    let file = fs::File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err("comparison inputs must be regular local files".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    manifest::decode(&bytes)
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let summary = Command::parse(args)?.execute()?;
    println!("{}", serde_json::to_string(&summary)?);
    if summary["certification_status"] == "pass" {
        Ok(())
    } else {
        Err("event benchmark certification blocked; inspect the published comparison report".into())
    }
}

#[cfg(test)]
#[path = "command_tests.rs"]
mod tests;
