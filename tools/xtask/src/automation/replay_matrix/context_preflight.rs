use super::manifest_preflight::{Manifest, Requirements};
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Input {
    manifest: PathBuf,
    requirements: Requirements,
    binary: PathBuf,
    native_runtime_root: PathBuf,
    model: PathBuf,
    model_sha256: String,
    minimum_context_tokens: u64,
    minimum_session_prompt_tokens: u64,
    max_output_tokens: u64,
    request_timeout_seconds: u64,
    startup_timeout_seconds: u64,
    timeout_seconds: u64,
    port: u16,
    output: PathBuf,
}

pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix context-preflight --input PATH --output PATH",
        values: &["--input", "--output"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    execute(
        root,
        input,
        Path::new(parsed.last("--output").ok_or("missing --output")?),
    )
}

fn execute(root: Option<&Path>, input: Input, result_path: &Path) -> DynResult<()> {
    if input.output.try_exists()?
        || result_path.try_exists()?
        || !input.output.is_absolute()
        || input.port == 0
    {
        return Err("context preflight requires unused outputs and an explicit port".into());
    }
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&input.manifest)?)?;
    super::manifest_preflight::validate(&manifest, &input.requirements)?;
    let started = std::time::Instant::now();
    let mut progress = super::progress::Record::boundary(
        super::progress::Phase::Preflight,
        super::progress::Event::Started,
        input.output.display().to_string(),
    );
    progress.cohorts = Some(manifest.cohorts.len());
    progress.sessions = Some(manifest.cohorts.values().map(Vec::len).sum());
    progress.turns = Some(
        manifest
            .cohorts
            .values()
            .flatten()
            .map(|trajectory| {
                trajectory
                    .messages
                    .iter()
                    .filter(|message| {
                        message.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
                    })
                    .count()
            })
            .sum(),
    );
    super::progress::emit(&progress)?;
    let mut jobs = Vec::new();
    let base = format!("http://127.0.0.1:{}/v1", input.port);
    for (name, trajectories) in manifest.cohorts {
        let workload = serde_json::json!({"trajectories":trajectories.into_iter().map(|trajectory|trajectory.original).collect::<Vec<_>>(),
            "model":"pending","base_url":base,"concurrency":1,"max_output_tokens":input.max_output_tokens,
            "request_timeout_seconds":input.request_timeout_seconds,"qualification_probe":true,
            "eligibility":{"context_tokens":input.minimum_context_tokens,"maximum_output_tokens":input.max_output_tokens,
                "minimum_session_prompt_tokens":if name=="warmup" {0} else {input.minimum_session_prompt_tokens}}});
        jobs.push((name, workload));
    }
    let (first_name, mut workload) = jobs.remove(0);
    let requests = input.output.join(format!("{first_name}-probes.jsonl"));
    let summary = input.output.join(format!("{first_name}.json"));
    let following=jobs.iter().map(|(name,workload)|serde_json::json!({"workload":workload,
        "requests_output":input.output.join(format!("{name}-probes.jsonl")),"summary_output":input.output.join(format!("{name}.json"))})).collect::<Vec<_>>();
    workload["following_cells"] = following.into();
    workload["model_pin"] = serde_json::json!({"sha256":input.model_sha256,"minimum_context_tokens":input.minimum_context_tokens,
        "output":input.output.join("model-identity.json")});
    workload["runtime_context"] = serde_json::json!({"required_tokens":input.minimum_context_tokens,"output":input.output.join("runtime.json")});
    let path = input.output.join("workload.json");
    crate::command::write_json_file(&path, &workload)?;
    let log = input.output.join("mesh.log");
    let mut arguments = Vec::new();
    for (option, path) in [
        ("--binary", &input.binary),
        ("--native-runtime-root", &input.native_runtime_root),
        ("--model", &input.model),
        ("--workload", &path),
        ("--requests-output", &requests),
        ("--summary-output", &summary),
        ("--server-log", &log),
    ] {
        arguments.extend([
            option.into(),
            path.to_str().ok_or("non-Unicode replay path")?.into(),
        ]);
    }
    arguments.extend([
        "--timeout".into(),
        input.timeout_seconds.to_string(),
        "--startup-timeout".into(),
        input.startup_timeout_seconds.to_string(),
    ]);
    let execution = super::server_cell::run(root, &arguments);
    let mut cohorts = BTreeMap::new();
    let mut tokens = BTreeMap::<String, BTreeMap<String, u64>>::new();
    for name in std::iter::once(&first_name).chain(jobs.iter().map(|(name, _)| name)) {
        let path = input.output.join(format!("{name}.json"));
        if !path.try_exists()? {
            continue;
        }
        let report: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let eligibility = report["eligibility"].clone();
        #[derive(Deserialize)]
        struct Eligibility {
            turns: Vec<Turn>,
        }
        #[derive(Deserialize)]
        struct Turn {
            request_id: String,
            prompt_tokens: Option<u64>,
        }
        let turns: Eligibility = serde_json::from_value(eligibility.clone())?;
        tokens.insert(
            name.clone(),
            turns
                .turns
                .into_iter()
                .filter_map(|turn| Some((turn.request_id, turn.prompt_tokens?)))
                .collect(),
        );
        cohorts.insert(name.clone(), eligibility);
    }
    let passed = execution.is_ok() && cohorts.len() == input.requirements.concurrency.len() + 1;
    let mut result =
        serde_json::json!({"passed":passed,"cohorts":cohorts,"prompt_tokens_by_cohort":tokens});
    result["artifact_sha256"] =
        serde_json::to_value(super::pass_identity::capture(&input.output)?)?;
    let identity = input.output.join("model-identity.json");
    if identity.try_exists()? {
        result["model"] = serde_json::from_slice(&std::fs::read(identity)?)?;
    }
    if let Err(error) = &execution {
        result["error"] = error.to_string().into();
    }
    crate::command::write_json_file(result_path, &result)?;
    progress.event = super::progress::Event::Completed;
    progress.elapsed_seconds = started.elapsed().as_secs_f64();
    progress.outcome = Some(if passed {
        super::progress::Outcome::Success
    } else {
        super::progress::Outcome::Error
    });
    super::progress::emit(&progress)?;
    execution
}
