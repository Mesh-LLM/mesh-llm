use super::{
    family_model,
    manifest_preflight::Requirements,
    run_input::Dataset,
    run_workload::{BuildJob, Input},
};
use crate::{automation::private_state::PrivateState, command::DynResult};
use serde::Deserialize;
use std::{collections::BTreeSet, path::Path};

#[derive(Deserialize)]
struct Matrix {
    replay: Replay,
    models: Vec<family_model::Model>,
}

#[derive(Deserialize)]
struct Replay {
    dataset_revision: String,
    dataset_sha256: String,
    concurrency: Vec<usize>,
    sessions_per_concurrency: usize,
    minimum_worker_waves: usize,
    minimum_context_tokens: u64,
    minimum_session_prompt_tokens: u64,
    min_isl: u64,
    max_isl: u64,
    min_turns: usize,
    passes: u32,
    warmup_turns: usize,
    max_output_tokens: u64,
}

pub(super) struct Context<'a> {
    pub matrix: &'a Path,
    pub family: &'a str,
    pub model: &'a Path,
    pub dataset: &'a Path,
    pub reader: &'a Path,
    pub output: &'a Path,
    pub refs: &'a [String],
    pub repo: &'a Path,
    pub worktree_root: &'a Path,
    pub git: &'a Path,
    pub just: &'a Path,
    pub timeout_seconds: u64,
}

pub(super) struct Prepared {
    pub input: Input,
    pub staging: PrivateState,
}

pub(super) fn prepare(context: &Context<'_>) -> DynResult<Prepared> {
    super::input::load(context.matrix).map_err(|error| error.to_string())?;
    let matrix: Matrix = serde_json::from_slice(&std::fs::read(context.matrix)?)?;
    let replay = matrix.replay;
    if !(1..=86400).contains(&context.timeout_seconds) {
        return Err("family preparation requires a timeout in 1..=86400 seconds".into());
    }
    let inner_timeout = context.timeout_seconds.max(2);
    let model = family_model::verify(
        &matrix.models,
        context.family,
        context.model,
        replay.minimum_context_tokens,
    )?;
    let dataset = Dataset {
        file: context.dataset.canonicalize()?,
        sha256: replay.dataset_sha256,
        revision: replay.dataset_revision,
        reader: context.reader.to_path_buf(),
        timeout_seconds: context.timeout_seconds.min(600),
        sessions_per_cohort: replay.sessions_per_concurrency,
        min_isl: replay.min_isl,
        max_isl: replay.max_isl,
        min_turns: replay.min_turns,
        frameworks: ["swe-agent", "mini-swe-agent", "openhands"]
            .map(str::to_owned)
            .to_vec(),
        source_datasets: [
            "swe-smith-claude-3-7-sonnet",
            "kwai-klear-swe-smith-mini",
            "nebius-swe-rebench-openhands",
        ]
        .map(str::to_owned)
        .to_vec(),
    };
    super::run_input::verify(&dataset)?;
    if !context.output.is_absolute() || context.output.try_exists()? {
        return Err("family output must be an unused absolute directory".into());
    }
    let staging = PrivateState::create(&std::env::temp_dir().canonicalize()?, "replay-family")?;
    let input = Input {
        manifest: staging.root().join("selection.json"),
        requirements: Requirements {
            concurrency: replay.concurrency,
            minimum_worker_waves: replay.minimum_worker_waves,
            warmup_turns: replay.warmup_turns,
            required_frameworks: dataset.frameworks.clone(),
        },
        builds: Vec::new(),
        build_jobs: jobs(context)?,
        engine_config: None,
        engine_config_sha256: None,
        validated_engine_config: None,
        context_qualification: Default::default(),
        replay_mode: super::replay_profile::Mode::All,
        hf_home: None,
        model: model.file,
        model_reference: Some(model.reference),
        model_sha256: model.sha256,
        minimum_context_tokens: replay.minimum_context_tokens,
        minimum_session_prompt_tokens: replay.minimum_session_prompt_tokens,
        require_recurrent_restores: model.recurrent,
        passes: replay.passes,
        max_output_tokens: replay.max_output_tokens,
        request_timeout_seconds: context.timeout_seconds.min(900),
        startup_timeout_seconds: (inner_timeout - 1).min(1800),
        timeout_seconds: inner_timeout,
        output: context.output.to_path_buf(),
        prompt_token_range: None,
        min_cache_pct: None,
        require_output_match: false,
        max_ttft_regression_pct: None,
        resume: false,
        dataset: Some(dataset),
    };
    input.validate()?;
    Ok(Prepared { input, staging })
}

fn jobs(context: &Context<'_>) -> DynResult<Vec<BuildJob>> {
    if context.refs.is_empty() {
        return Err("family requires at least one ordered ref".into());
    }
    if [
        context.repo,
        context.worktree_root,
        context.git,
        context.just,
        context.reader,
    ]
    .iter()
    .any(|path| !path.is_absolute())
    {
        return Err("family repository, worktree root and tools must be absolute".into());
    }
    let mut labels = BTreeSet::new();
    context
        .refs
        .iter()
        .map(|reference| {
            let (label, reference) = reference.split_once('=').ok_or("ref must be label=ref")?;
            if label.is_empty()
                || !label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
                || !labels.insert(label)
                || reference.trim().is_empty()
            {
                return Err(
                    "ref labels must be unique safe components and refs must be nonempty".into(),
                );
            }
            Ok(BuildJob {
                repo: context.repo.to_path_buf(),
                worktree_root: context.worktree_root.to_path_buf(),
                label: label.into(),
                reference: reference.into(),
                backend: "metal".into(),
                git: context.git.to_path_buf(),
                just: context.just.to_path_buf(),
                timeout_seconds: context.timeout_seconds,
                logs: context.output.join("build-logs").join(label),
                skip_build: false,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../tests/replay_family_workload/mod.rs"]
mod tests;
