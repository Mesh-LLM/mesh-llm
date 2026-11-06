//! AgentX owns its workload, provenance and lifecycle independently of replay-matrix.
mod agentic_config;
mod cache_decisions;
mod campaign;
mod capacity;
mod command;
mod comparison_cohort;
mod completed_work;
mod contention_cohort;
mod context_support;
mod final_sweep;
#[cfg(test)]
mod final_sweep_tests;
mod hardware;
mod inputs;
mod lifecycle;
mod memory;
mod mlflow_export;
mod mlflow_publish;
mod native_kv_span;
mod overlap;
mod phase_outcome;
mod prepared_launch;
mod prepared_tools;
mod results;
mod scheduler_profile;
mod standard_runtime;
mod trace_context;
mod workload;
use crate::{command::DynResult, repository::check_args::Grammar};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Profile {
    Smoke,
    Benchmark,
    Comparison,
    Contention,
}
impl Profile {
    fn harness_version(self) -> &'static str {
        match self {
            Self::Smoke => "agentx-1.0.12.post2-local-smoke-9",
            Self::Benchmark => "agentx-1.0.12.post2-local-benchmark-8",
            Self::Comparison => "agentx-1.0.12.post2-local-comparison-3",
            Self::Contention => "agentx-1.0.12.post2-local-contention-2",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Benchmark => "benchmark",
            Self::Comparison => "comparison",
            Self::Contention => "contention",
        }
    }
    fn concurrencies(self) -> &'static [u32] {
        // Offered load is independent of engine-planned serving capacity.
        match self {
            Self::Contention => &[4, 8, 16],
            _ => &[1, 2, 4],
        }
    }
}
#[derive(Clone, Copy)]
pub(super) enum Engine {
    Llama,
    Skippy,
}
impl Engine {
    fn label(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Skippy => "skippy",
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
impl Model {
    fn registry_ref(&self) -> String {
        format!("hf:{}@{}#{}", self.repo, self.revision, self.files[0].name)
    }
}
#[derive(Deserialize, Serialize)]
struct Prepared {
    #[serde(default)]
    tools: Option<prepared_tools::Tools>,
    root: PathBuf,
    output: PathBuf,
    model: Model,
    weights: PathBuf,
    dataset: PathBuf,
    profile: Profile,
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        return crate::repository::check_report::CheckReport::success(final_sweep::USAGE.into())
            .emit();
    }
    if matches!(
        args.first().map(String::as_str),
        Some("pin" | "comparison" | "mlflow-export" | "mlflow-publish")
    ) {
        return final_sweep::command(args);
    }
    if args.first().map(String::as_str) != Some("worker") {
        return Err("AgentX accepts comparison --input JSON --output DIR or mlflow-export; legacy unpinned launch modes are not admitted".into());
    }
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation agentx worker --input PATH --port PORT --concurrency N",
        values: &["--input", "--port", "--concurrency"],
        flags: &["--help"],
    };
    let parsed = GRAMMAR
        .parse(args)
        .map_err(|_| "invalid AgentX arguments")?;
    if parsed.flag("--help") {
        return crate::repository::check_report::CheckReport::success(GRAMMAR.usage.into()).emit();
    }
    match parsed.positionals.as_slice() {
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
