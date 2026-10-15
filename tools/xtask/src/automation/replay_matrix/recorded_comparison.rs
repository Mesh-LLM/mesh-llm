//! Pinned recorded windows shared by local and Jianyang comparisons.
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    schema_version: u32,
    name: String,
    model_sha256: String,
    dataset_revision: String,
    dataset_sha256: String,
    upstream_commit: String,
    context_size: u64,
    warmup_turns: usize,
    sampling: super::replay_sampling::Sampling,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    session_id: String,
    start: usize,
    count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Bindings {
    source_manifest: PathBuf,
    source_manifest_sha256: String,
    mesh_build: PathBuf,
    llama_server: PathBuf,
    llama_source: PathBuf,
    model: PathBuf,
    hf_home: PathBuf,
}

pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix recorded-comparison --job PATH --input PATH --output PATH [--plan]",
        values: &["--job", "--input", "--output"],
        flags: &["--help", "--plan"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(v) => v,
        Err(e) => return e.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return Err("unexpected positional arguments".into());
    }
    let job_path = parsed.last("--job").ok_or("missing --job")?;
    let job: Job = serde_json::from_slice(&std::fs::read(job_path)?)?;
    let bindings: Bindings = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    let output = std::path::absolute(parsed.last("--output").ok_or("missing --output")?)?;
    validate(&job)?;
    verify_bindings(&job, &bindings)?;
    if output.exists() {
        return Err("comparison output already exists".into());
    }
    let source: Value = serde_json::from_slice(&std::fs::read(&bindings.source_manifest)?)?;
    let manifests = job
        .cases
        .iter()
        .map(|case| manifest(&source, &job, case))
        .collect::<DynResult<Vec<_>>>()?;
    std::fs::create_dir_all(output.join("inputs"))?;
    std::fs::copy(job_path, output.join("job.json"))?;
    let engines = output.join("inputs/engines.json");
    crate::command::write_json_file(
        &engines,
        &json!({"schema_version":1,"comparison":{"model":bindings.model},"arms":[{
        "label":"llama","engine":"llama.cpp","executable":bindings.llama_server,"cwd":bindings.llama_source,
        "model":bindings.model,"served_model":"qwen35-4b","context_size":job.context_size,"max_concurrency":1,
        "prefix_cache":true,"batch_size":2048,"ubatch_size":512,"extra_args":["--flash-attn","on"]}]}),
    )?;
    let build: Value = serde_json::from_slice(&std::fs::read(&bindings.mesh_build)?)?;
    for (case, manifest) in job.cases.iter().zip(manifests) {
        let path = output.join("inputs").join(format!("{}.json", case.name));
        crate::command::write_json_file(&path, &manifest)?;
        let input = json!({"manifest":path,"requirements":{"concurrency":[1],"minimum_worker_waves":1,"warmup_turns":job.warmup_turns,"required_frameworks":[]},
            "builds":[build],"engine_config":engines,"context_qualification":"captured","replay_mode":"window",
            "model":bindings.model,"model_sha256":job.model_sha256,"hf_home":bindings.hf_home,
            "passes":1,"max_output_tokens":null,"require_complete_output":true,
            "request_timeout_seconds":900,"startup_timeout_seconds":1800,"timeout_seconds":19800,"output":output.join(&case.name)});
        let input_path = output
            .join("inputs")
            .join(format!("{}-run.json", case.name));
        crate::command::write_json_file(&input_path, &input)?;
        if !parsed.flag("--plan") {
            // The retained run owner verifies binary/runtime identities and always
            // cleans up its servers. A failure stops subsequent cases.
            super::run_execution::run(
                root,
                &["--input".into(), input_path.to_string_lossy().into_owned()],
            )?;
        }
    }
    if !parsed.flag("--plan") {
        super::comparison_result::write(
            &output,
            &job.cases
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            &build,
        )?;
    }
    Ok(())
}

fn validate(job: &Job) -> DynResult<()> {
    job.sampling.validate()?;
    if job.schema_version != 1
        || job.name.trim().is_empty()
        || job.cases.len() != 2
        || job.warmup_turns != 4
        || job.context_size != 131072
        || job.sampling.enable_thinking != Some(false)
        || job.cases.iter().any(|c| {
            c.count != 50
                || c.name.is_empty()
                || !c
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
        || job.cases[0].name == job.cases[1].name
    {
        return Err("comparison requires two named 50-turn cases, four warmups, 128K context and explicit thinking off".into());
    }
    Ok(())
}

fn verify_bindings(job: &Job, bindings: &Bindings) -> DynResult<()> {
    for path in [
        &bindings.source_manifest,
        &bindings.mesh_build,
        &bindings.llama_server,
        &bindings.llama_source,
        &bindings.model,
        &bindings.hf_home,
    ] {
        if !path.is_absolute() || !path.exists() {
            return Err(format!(
                "comparison binding must exist and be absolute: {}",
                path.display()
            )
            .into());
        }
    }
    for (path, expected) in [
        (&bindings.source_manifest, &bindings.source_manifest_sha256),
        (&bindings.model, &job.model_sha256),
    ] {
        if crate::product::digest::file_sha256(path).map_err(|e| e.error)? != *expected {
            return Err(format!("comparison input digest mismatch: {}", path.display()).into());
        }
    }
    let head = super::git_head::git_head_with_budget(
        &bindings.llama_source,
        std::time::Duration::from_secs(10),
    )?;
    if head.trim() != job.upstream_commit {
        return Err("upstream source pin mismatch".into());
    }
    Ok(())
}

fn manifest(source: &Value, job: &Job, case: &Case) -> DynResult<Value> {
    if source["metadata"]["dataset_revision"] != job.dataset_revision
        || source["metadata"]["dataset_sha256"] != job.dataset_sha256
    {
        return Err("recorded manifest dataset provenance mismatch".into());
    }
    let cohorts = source["cohorts"].as_object().ok_or("missing cohorts")?;
    let mut matches = cohorts
        .iter()
        .filter(|(name, _)| name.as_str() != "warmup")
        .flat_map(|(_, v)| v.as_array().into_iter().flatten())
        .filter(|v| v["session_id"] == case.session_id);
    let mut selected = matches.next().ok_or("recorded session absent")?.clone();
    if matches.any(|v| v != &selected) {
        return Err("conflicting recorded session copies".into());
    }
    selected["replay_window"] = json!({"start":case.start,"count":case.count});
    selected["replay_sampling"] = serde_json::to_value(&job.sampling)?;
    let mut warmup = source["cohorts"]["warmup"]
        .as_array()
        .and_then(|v| v.first())
        .ok_or("missing warmup trajectory")?
        .clone();
    warmup["replay_sampling"] = serde_json::to_value(&job.sampling)?;
    let selection = super::recorded_requests::Selection {
        model: "pending",
        maximum_output_tokens: None,
        turn_limit: None,
        qualification_probe: false,
    };
    super::replay_profile::build(
        &[serde_json::from_value(selected.clone())?],
        &selection,
        super::replay_profile::Mode::Window,
    )?;
    Ok(
        json!({"schema_version":1,"metadata":{"dataset":source["metadata"]["dataset"],
        "dataset_revision":job.dataset_revision,"dataset_sha256":job.dataset_sha256,
        "selection":{"mode":"consecutive-window","session":case.session_id,"start":case.start,"count":case.count,"whole_trajectories":false}},
        "cohorts":{"1":[selected],"warmup":[warmup]}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn committed_job_pins_two_windows_and_rejects_thinking_drift() {
        let mut job: Job = serde_json::from_str(include_str!(
            "../../../../../mesh/evals/recorded-replay/job.json"
        ))
        .unwrap();
        validate(&job).unwrap();
        job.sampling.enable_thinking = None;
        assert!(validate(&job).is_err());
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    #[test]
    fn windows_keep_recorded_history_align_warmups_and_refuse_wrong_or_short_sessions() {
        let job: Job = serde_json::from_str(include_str!(
            "../../../../../mesh/evals/recorded-replay/job.json"
        ))
        .unwrap();
        let trajectory = |id: &str| {
            let mut messages = vec![json!({"role":"user","content":"start"})];
            for turn in 0..80 {
                messages.push(json!({"role":"assistant","content":format!("answer {turn}")}));
                messages.push(json!({"role":"user","content":format!("next {turn}")}));
            }
            json!({"session_id":id,"source_dataset":"fixture","agent_framework":"fixture","messages":messages})
        };
        let mut source = json!({"metadata":{"dataset_revision":job.dataset_revision,"dataset_sha256":job.dataset_sha256},
            "cohorts":{"1":[trajectory(&job.cases[0].session_id)],"2":[trajectory(&job.cases[1].session_id)],"warmup":[trajectory("warmup")]}});
        for case in &job.cases {
            let manifest = manifest(&source, &job, case).unwrap();
            let selected: super::super::recorded_requests::Trajectory =
                serde_json::from_value(manifest["cohorts"]["1"][0].clone()).unwrap();
            let selection = super::super::recorded_requests::Selection {
                model: "fixture",
                maximum_output_tokens: None,
                turn_limit: None,
                qualification_probe: false,
            };
            let requests = super::super::replay_profile::build(
                &[selected],
                &selection,
                super::super::replay_profile::Mode::Window,
            )
            .unwrap();
            assert_eq!(requests[0].len(), 50);
            assert_eq!(requests[0][0].assistant_turn, case.start);
            assert_eq!(
                requests[0][0].body["messages"].as_array().unwrap().len(),
                1 + case.start * 2
            );
            assert_eq!(
                requests[0][0].body["chat_template_kwargs"]["enable_thinking"],
                false
            );
            assert!(requests[0][0].body.get("max_tokens").is_none());
            assert_eq!(
                manifest["cohorts"]["warmup"][0]["replay_sampling"]["enable_thinking"],
                false
            );
        }
        source["metadata"]["dataset_revision"] = "wrong".into();
        assert!(manifest(&source, &job, &job.cases[0]).is_err());
        source["metadata"]["dataset_revision"] = job.dataset_revision.clone().into();
        source["cohorts"]["1"][0]["messages"] = json!([]);
        assert!(manifest(&source, &job, &job.cases[0]).is_err());
    }
}
