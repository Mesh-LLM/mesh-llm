//! Documented manual replay options normalized without launching tools.
use super::replay_profile::Mode;
use crate::{
    command::DynResult,
    repository::check_args::{Grammar, ParsedArgs},
};
use serde::Serialize;
use std::path::PathBuf;
pub(super) const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix {plan|run} --ref LABEL=REF... --model MODEL (--trajectory-manifest PATH | --dataset-file PATH --sessions-per-concurrency N) [--replay-mode checkpoint|final|all] [--output PATH]",
    values: &[
        "--repo",
        "--ref",
        "--model",
        "--model-file",
        "--engine-config",
        "--backend",
        "--passes",
        "--replay-mode",
        "--concurrency",
        "--minimum-worker-waves",
        "--trajectories-per-framework",
        "--sessions-per-concurrency",
        "--expected-model-sha256",
        "--minimum-context-tokens",
        "--minimum-session-prompt-tokens",
        "--max-output-tokens",
        "--warmup-turns",
        "--min-isl",
        "--max-isl",
        "--min-turns",
        "--framework",
        "--source-dataset",
        "--trajectory-manifest",
        "--require-framework",
        "--prompt-token-range",
        "--min-cache-pct",
        "--max-ttft-regression-pct",
        "--dataset-file",
        "--output",
        "--worktree-root",
        "--hf-home",
        "--startup-timeout",
        "--request-timeout",
        "--timeout",
        "--python",
    ],
    flags: &[
        "--help",
        "--require-recurrent-restores",
        "--require-output-match",
        "--skip-build",
        "--resume",
    ],
};
#[derive(Serialize)]
pub(super) struct Ref {
    pub label: String,
    pub reference: String,
}
#[derive(Serialize)]
pub(super) struct Options {
    pub refs: Vec<Ref>,
    pub model: String,
    pub model_file: Option<PathBuf>,
    pub repo: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub manifest: Option<PathBuf>,
    pub dataset: Option<PathBuf>,
    pub engine_config: Option<PathBuf>,
    pub backend: String,
    pub mode: Mode,
    pub concurrency: Vec<usize>,
    pub passes: u32,
    pub worker_waves: usize,
    pub warmup: usize,
    pub sessions: Option<usize>,
    pub frameworks: Vec<String>,
    pub sources: Vec<String>,
    pub required_frameworks: Vec<String>,
    pub max_output: u64,
    pub min_isl: u64,
    pub max_isl: u64,
    pub min_turns: usize,
    pub expected_model_sha256: Option<String>,
    pub minimum_context: u64,
    pub minimum_session: u64,
    pub recurrent: bool,
    pub prompt_range: Option<[u64; 2]>,
    pub min_cache: Option<f64>,
    pub output_match: bool,
    pub max_ttft: Option<f64>,
    pub startup: u64,
    pub request: u64,
    pub timeout: u64,
    pub worktree_root: Option<PathBuf>,
    pub hf_home: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub skip_build: bool,
    pub resume: bool,
}
fn number<T: std::str::FromStr>(parsed: &ParsedArgs, name: &str, default: &str) -> DynResult<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    Ok(parsed.last(name).unwrap_or(default).parse()?)
}
fn list(parsed: &ParsedArgs, name: &str, defaults: &[&str]) -> Vec<String> {
    let values = parsed.all(name);
    if values.is_empty() {
        defaults.iter().map(|v| (*v).into()).collect()
    } else {
        values.into_iter().map(str::to_owned).collect()
    }
}
fn unique(values: &[String]) -> bool {
    values.iter().all(|v| !v.is_empty())
        && values
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == values.len()
}
fn path(parsed: &ParsedArgs, name: &str) -> DynResult<Option<PathBuf>> {
    parsed
        .last(name)
        .map(std::path::absolute)
        .transpose()
        .map_err(Into::into)
}
pub(super) fn parse(parsed: &ParsedArgs, running: bool) -> DynResult<Options> {
    if !parsed.positionals.is_empty() {
        return Err("unexpected manual replay arguments".into());
    }
    let refs = references(parsed)?;
    let model = parsed.last("--model").ok_or("missing --model")?.to_owned();
    if model.is_empty() || model.chars().any(char::is_control) {
        return Err("invalid model reference".into());
    }
    let Cohort {
        manifest,
        dataset,
        frameworks,
        required_frameworks,
        sources,
        sessions,
        concurrency,
        worker_waves,
    } = cohort(parsed, running)?;
    let Gates {
        prompt_range,
        min_cache,
        max_ttft,
    } = gates(parsed)?;
    let options = Options {
        refs,
        model,
        model_file: path(parsed, "--model-file")?,
        repo: path(parsed, "--repo")?,
        output: path(parsed, "--output")?,
        manifest,
        dataset,
        engine_config: path(parsed, "--engine-config")?,
        backend: parsed.last("--backend").unwrap_or("metal").into(),
        mode: Mode::parse(parsed.last("--replay-mode").unwrap_or("checkpoint"))?,
        concurrency,
        passes: number(parsed, "--passes", "1")?,
        worker_waves,
        warmup: number(parsed, "--warmup-turns", "4")?,
        sessions,
        frameworks,
        sources,
        required_frameworks,
        max_output: number(parsed, "--max-output-tokens", "2048")?,
        min_isl: number(parsed, "--min-isl", "8192")?,
        max_isl: number(parsed, "--max-isl", "65536")?,
        min_turns: number(parsed, "--min-turns", "5")?,
        expected_model_sha256: parsed.last("--expected-model-sha256").map(str::to_owned),
        minimum_context: number(parsed, "--minimum-context-tokens", "0")?,
        minimum_session: number(parsed, "--minimum-session-prompt-tokens", "0")?,
        recurrent: parsed.flag("--require-recurrent-restores"),
        prompt_range,
        min_cache,
        output_match: parsed.flag("--require-output-match"),
        max_ttft,
        startup: number(parsed, "--startup-timeout", "1800")?,
        request: number(parsed, "--request-timeout", "900")?,
        timeout: number(parsed, "--timeout", "86400")?,
        worktree_root: path(parsed, "--worktree-root")?,
        hf_home: path(parsed, "--hf-home")?,
        python: path(parsed, "--python")?,
        skip_build: parsed.flag("--skip-build"),
        resume: parsed.flag("--resume"),
    };
    validate(&options, running)?;
    Ok(options)
}

fn references(parsed: &ParsedArgs) -> DynResult<Vec<Ref>> {
    let mut refs = Vec::new();
    let mut labels = std::collections::BTreeSet::new();
    for value in parsed.all("--ref") {
        let (label, reference) = value.split_once('=').ok_or("ref must be LABEL=REF")?;
        if label.is_empty()
            || [".", ".."].contains(&label)
            || !label
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || b"_-".contains(&v))
            || reference.is_empty()
            || reference.starts_with('-')
            || reference.chars().any(char::is_control)
            || !labels.insert(label.to_owned())
        {
            return Err("invalid or duplicate ref identity".into());
        }
        refs.push(Ref {
            label: label.into(),
            reference: reference.into(),
        });
    }
    if refs.is_empty() {
        return Err("at least one --ref required".into());
    }
    Ok(refs)
}

struct Cohort {
    manifest: Option<PathBuf>,
    dataset: Option<PathBuf>,
    frameworks: Vec<String>,
    required_frameworks: Vec<String>,
    sources: Vec<String>,
    sessions: Option<usize>,
    concurrency: Vec<usize>,
    worker_waves: usize,
}
fn cohort(parsed: &ParsedArgs, running: bool) -> DynResult<Cohort> {
    let manifest = path(parsed, "--trajectory-manifest")?;
    let dataset = path(parsed, "--dataset-file")?;
    if manifest.is_some() && dataset.is_some() || running && manifest.is_none() && dataset.is_none()
    {
        return Err("run needs exactly one dataset or captured manifest".into());
    }
    let frameworks = list(
        parsed,
        "--framework",
        &["swe-agent", "mini-swe-agent", "openhands"],
    );
    let required_frameworks = list(parsed, "--require-framework", &[]);
    let sources = list(
        parsed,
        "--source-dataset",
        &[
            "swe-smith-claude-3-7-sonnet",
            "kwai-klear-swe-smith-mini",
            "nebius-swe-rebench-openhands",
        ],
    );
    if !unique(&frameworks) || !unique(&required_frameworks) || !unique(&sources) {
        return Err("framework/source identities must be unique and nonempty".into());
    }
    let per = parsed
        .last("--trajectories-per-framework")
        .map(str::parse::<usize>)
        .transpose()?;
    let total = parsed
        .last("--sessions-per-concurrency")
        .map(str::parse::<usize>)
        .transpose()?;
    if per.is_some() && total.is_some() {
        return Err("choose total or per-framework session count".into());
    }
    let sessions = total.or(per
        .map(|n| {
            n.checked_mul(frameworks.len())
                .ok_or("session count overflow")
        })
        .transpose()?);
    if manifest.is_none() && sessions.is_none() || sessions == Some(0) {
        return Err("dataset selection needs a positive explicit session count".into());
    }
    let concurrency = if parsed.all("--concurrency").is_empty() {
        vec![1, 2, 4]
    } else {
        parsed
            .all("--concurrency")
            .into_iter()
            .map(str::parse)
            .collect::<Result<Vec<usize>, _>>()?
    };
    let worker_waves: usize = number(parsed, "--minimum-worker-waves", "2")?;
    if concurrency.iter().any(|n| !(1..=256).contains(n))
        || concurrency
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != concurrency.len()
        || worker_waves == 0
    {
        return Err("invalid concurrency/wave budget".into());
    }
    let minimum = worker_waves
        .checked_mul(*concurrency.iter().max().ok_or("empty concurrency")?)
        .ok_or("wave count overflow")?;
    if sessions.is_some_and(|n| n < minimum) {
        return Err("cohort needs the minimum complete worker waves".into());
    }
    Ok(Cohort {
        manifest,
        dataset,
        frameworks,
        required_frameworks,
        sources,
        sessions,
        concurrency,
        worker_waves,
    })
}

struct Gates {
    prompt_range: Option<[u64; 2]>,
    min_cache: Option<f64>,
    max_ttft: Option<f64>,
}
fn gates(parsed: &ParsedArgs) -> DynResult<Gates> {
    let prompt_range = parsed
        .last("--prompt-token-range")
        .map(|value| -> DynResult<[u64; 2]> {
            let (a, b) = value
                .split_once(':')
                .ok_or("prompt range must be MIN:MAX")?;
            let pair = [a.parse()?, b.parse()?];
            if pair[0] == 0 || pair[0] > pair[1] {
                return Err("invalid prompt range".into());
            }
            Ok(pair)
        })
        .transpose()?;
    let min_cache = parsed
        .last("--min-cache-pct")
        .map(str::parse::<f64>)
        .transpose()?;
    let max_ttft = parsed
        .last("--max-ttft-regression-pct")
        .map(str::parse::<f64>)
        .transpose()?;
    if min_cache.is_some_and(|n| !n.is_finite() || n <= 0.0 || n > 100.0)
        || max_ttft.is_some_and(|n| !n.is_finite() || n < 0.0)
    {
        return Err("invalid acceptance budget".into());
    }
    Ok(Gates {
        prompt_range,
        min_cache,
        max_ttft,
    })
}

fn validate(options: &Options, running: bool) -> DynResult<()> {
    if !(1..=1000).contains(&options.passes)
        || options.warmup == 0
        || options.max_output == 0
        || options.min_isl == 0
        || options.min_isl >= options.max_isl
        || options.min_turns == 0
        || [options.startup, options.request, options.timeout]
            .iter()
            .any(|n| !(1..=86400).contains(n))
        || options.startup >= options.timeout
        || running && options.output.is_none()
    {
        return Err("invalid manual workload or deadline budget".into());
    }
    if options.mode != Mode::All
        && (options.minimum_context != 0 || options.minimum_session != 0 || options.recurrent)
    {
        return Err("full-session context/recurrent qualification requires replay-mode all".into());
    }
    Ok(())
}
