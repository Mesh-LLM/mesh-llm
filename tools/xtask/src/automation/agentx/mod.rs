//! AgentX owns its workload, provenance and lifecycle independently of replay-matrix.
mod command;
mod inputs;
mod lifecycle;
mod results;
mod workload;
use crate::{command::DynResult, repository::check_args::Grammar};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Profile {
    Smoke,
    Benchmark,
}
impl Profile {
    fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Benchmark => "benchmark",
        }
    }
    fn parse(value: &str) -> DynResult<Self> {
        match value {
            "smoke" => Ok(Self::Smoke),
            "benchmark" => Ok(Self::Benchmark),
            _ => Err("profile must be smoke or benchmark".into()),
        }
    }
    fn concurrencies(self) -> &'static [u32] {
        if self == Self::Smoke {
            &[1]
        } else {
            &[1, 2, 4, 8]
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FilePin {
    name: String,
    sha256: String,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Model {
    key: String,
    family: String,
    repo: String,
    revision: String,
    tokenizer: String,
    tokenizer_revision: String,
    files: Vec<FilePin>,
}
#[derive(Deserialize, Serialize)]
struct Prepared {
    root: PathBuf,
    output: PathBuf,
    model: Model,
    weights: PathBuf,
    dataset: PathBuf,
    profile: Profile,
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation agentx {run|worker} --model KEY --profile {smoke|benchmark} --output PATH | worker --input PATH --port PORT --concurrency N",
        values: &[
            "--model",
            "--profile",
            "--output",
            "--input",
            "--port",
            "--concurrency",
        ],
        flags: &["--help"],
    };
    let parsed = GRAMMAR
        .parse(args)
        .map_err(|_| "invalid AgentX arguments")?;
    if parsed.flag("--help") {
        return crate::repository::check_report::CheckReport::success(GRAMMAR.usage.into()).emit();
    }
    match parsed.positionals.as_slice() {
        [verb] if *verb == "run" => {
            let root = std::env::current_dir()?;
            let profile = Profile::parse(parsed.last("--profile").ok_or("missing profile")?)?;
            let output = PathBuf::from(parsed.last("--output").ok_or("missing output")?);
            if !output.is_absolute() {
                return Err("output must be absolute".into());
            }
            std::fs::create_dir(&output)?;
            let prepared = inputs::prepare(
                root,
                output.clone(),
                parsed.last("--model").ok_or("missing model")?,
                profile,
            );
            let result = prepared.and_then(|prepared| lifecycle::campaign(&prepared));
            let archive = results::archive(&output);
            result?;
            archive
        }
        [verb] if *verb == "worker" => {
            let prepared: Prepared = serde_json::from_slice(&std::fs::read(
                parsed.last("--input").ok_or("missing input")?,
            )?)?;
            workload::run(
                &prepared,
                parsed.last("--port").ok_or("missing port")?.parse()?,
                parsed
                    .last("--concurrency")
                    .ok_or("missing concurrency")?
                    .parse()?,
            )
        }
        _ => Err(GRAMMAR.usage.into()),
    }
}
fn write(path: &Path, value: &impl Serialize) -> DynResult<()> {
    crate::command::write_json_file(path, value)
}
