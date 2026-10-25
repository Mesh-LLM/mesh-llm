use super::{
    history_input::{Input, Model},
    manifest_preflight::{Manifest, Requirements},
    session_evidence,
};
use crate::command::DynResult;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub(super) struct Cell {
    pub pass: u32,
    pub value: Value,
    pub directory: PathBuf,
    pub modified: std::time::SystemTime,
    pub verified: bool,
}
pub(super) struct Family {
    pub cells: Vec<Cell>,
    pub document: Value,
    pub backend: String,
    pub problems: Vec<String>,
}
pub(super) fn load(input: &Input, model: &Model) -> DynResult<Family> {
    let root = input.root.join(&model.family);
    let document: Value = super::history_input::read(&root.join("run.json"))?;
    let mut cells = Vec::new();
    for entry in sorted(&root.join("data"))? {
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "non-Unicode replay directory")?;
        let Some(pass) = name.strip_prefix("pass-") else {
            continue;
        };
        let pass = pass.parse::<u32>()?;
        directory(&entry.path())?;
        let arm = entry.path().join(&input.label);
        if !arm.try_exists()? {
            continue;
        }
        directory(&arm)?;
        for file in sorted(&arm)? {
            let name = file
                .file_name()
                .into_string()
                .map_err(|_| "non-Unicode cell file")?;
            if name.starts_with("c-") && name.ends_with(".json") {
                cells.push(Cell {
                    pass,
                    value: super::history_input::read(&file.path())?,
                    directory: arm.clone(),
                    modified: file.metadata()?.modified()?,
                    verified: false,
                });
            }
        }
    }
    if cells.is_empty() {
        return Err("no replay cells matched family and label".into());
    }
    let mut family = Family {
        cells,
        document,
        backend: String::new(),
        problems: Vec::new(),
    };
    let verification = verify(input, model, &root, &mut family);
    if let Err(error) = verification {
        family.problems.push(error.to_string());
    }
    coverage(input, &mut family)?;
    Ok(family)
}
fn verify(input: &Input, model: &Model, root: &Path, family: &mut Family) -> DynResult<()> {
    let document = &family.document;
    require_full_replay(document)?;
    let expected_uri = format!("{}@{}/{}", model.repo, model.revision, model.file);
    if document["config"]["model"] != expected_uri {
        return Err("replay model differs from pinned matrix URI".into());
    }
    let builds = document["builds"]
        .as_array()
        .ok_or("missing backend builds")?;
    let mut selected = builds.iter().filter(|build| build["label"] == input.label);
    let build = selected.next().ok_or("missing selected backend build")?;
    if selected.next().is_some() || build["commit"] != input.source_sha {
        return Err("selected backend source differs from history source".into());
    }
    family.backend = build["binary_sha256"]
        .as_str()
        .ok_or("missing backend binary digest")?
        .into();
    super::history_input::pin(&family.backend, 64)?;
    if input
        .backend_sha256
        .as_ref()
        .is_some_and(|expected| expected != &family.backend)
    {
        return Err("backend binary digest differs from caller".into());
    }
    let manifest_name = Path::new(
        document["inputs"]["manifest"]
            .as_str()
            .ok_or("missing retained manifest")?,
    )
    .file_name()
    .ok_or("missing manifest filename")?;
    let manifest_path = root.join("inputs").join(manifest_name);
    let digest =
        crate::product::digest::file_sha256(&manifest_path).map_err(|error| error.error)?;
    if document["inputs"]["manifest_sha256"] != digest {
        return Err("trajectory manifest digest mismatch".into());
    }
    let manifest: Manifest = super::history_input::read(&manifest_path)?;
    super::manifest_preflight::validate(
        &manifest,
        &Requirements {
            concurrency: input.replay.concurrency.clone(),
            minimum_worker_waves: input.replay.minimum_worker_waves,
            warmup_turns: input.replay.warmup_turns,
            required_frameworks: vec![
                "swe-agent".into(),
                "mini-swe-agent".into(),
                "openhands".into(),
            ],
        },
    )?;
    let preflight = &document["context_preflight"][&input.label];
    if preflight["passed"] != true
        || preflight["model"]["sha256"] != model.sha256
        || preflight["model"]["native_context_tokens"]
            .as_u64()
            .is_none_or(|context| context < input.replay.minimum_context_tokens)
    {
        return Err("missing pinned model and successful runtime context qualification".into());
    }
    let preflight_directory = root.join("context-preflight").join(&input.label);
    let runtime: session_evidence::Runtime =
        super::history_input::read(&preflight_directory.join("runtime.json"))?;
    let effective_context = runtime.context(input.replay.minimum_context_tokens)?;
    if preflight.get("artifact_sha256").is_some() {
        super::pass_identity::verify(&preflight_directory, &preflight["artifact_sha256"])?;
        let retained: Value =
            super::history_input::read(&preflight_directory.join("eligibility.json"))?;
        if retained != *preflight {
            return Err("preflight snapshot differs from retained evidence".into());
        }
    }
    for level in &input.replay.concurrency {
        if preflight["cohorts"][&level.to_string()]["context_tokens"].as_u64()
            != Some(effective_context)
        {
            return Err("preflight effective context differs from retained runtime".into());
        }
    }
    for cell in &mut family.cells {
        match verify_cell(input, model, &manifest, preflight, cell) {
            Ok(()) => cell.verified = true,
            Err(error) => family.problems.push(format!("pass {}: {error}", cell.pass)),
        }
    }
    if !accepted_run_gates(&document["gates"]) || document["completed_at"].as_str().is_none() {
        family
            .problems
            .push("run did not complete all acceptance gates".into());
        for cell in &mut family.cells {
            cell.verified = false;
        }
    }
    Ok(())
}
fn verify_cell(
    input: &Input,
    model: &Model,
    manifest: &Manifest,
    preflight: &Value,
    cell: &Cell,
) -> DynResult<()> {
    let level = cell.value["concurrency"]
        .as_u64()
        .ok_or("missing concurrency")?
        .to_string();
    let trajectories = manifest
        .cohorts
        .get(&level)
        .ok_or("foreign measured cohort")?;
    if trajectories.len() != input.replay.sessions_per_concurrency {
        return Err("session count differs from matrix".into());
    }
    let context = preflight["cohorts"][&level]["context_tokens"]
        .as_u64()
        .ok_or("missing effective context")?;
    if preflight["cohorts"][&level]["passed"] != true
        || context < input.replay.minimum_context_tokens
    {
        return Err("runtime context qualification failed".into());
    }
    let raw = records(&cell.directory.join(format!("c-{level}-requests.jsonl")))?;
    let sessions = trajectories
        .iter()
        .map(|trajectory| {
            serde_json::from_value::<session_evidence::Trajectory>(trajectory.original.clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let requests = raw
        .iter()
        .map(|record| serde_json::from_value::<session_evidence::Request>(record.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let complete = session_evidence::complete(&sessions, &requests);
    if !complete.passed
        || serde_json::to_value(&complete)? != cell.value["completeness"]
        || cell.value["requests"].as_u64() != Some(u64::try_from(raw.len())?)
    {
        return Err("raw turn coverage differs from manifest and summary".into());
    }
    let identity = super::cohort_identity::digest(&serde_json::Value::Array(
        trajectories
            .iter()
            .map(|trajectory| trajectory.original.clone())
            .collect(),
    ))?;
    if cell.value["session_cohort_sha256"] != identity {
        return Err("cell trajectory identity mismatch".into());
    }
    let eligibility = super::context_eligibility::evaluate(
        &sessions,
        &requests,
        &super::context_eligibility::Budget {
            context_tokens: context,
            maximum_output_tokens: input.replay.max_output_tokens,
            minimum_session_prompt_tokens: input.replay.minimum_session_prompt_tokens,
        },
    );
    if !eligibility.passed || cell.value["acceptance"]["passed"] != true {
        return Err("measured full-session acceptance failed".into());
    }
    let expected_prompt_tokens: BTreeMap<String, u64> =
        serde_json::from_value(preflight["prompt_tokens_by_cohort"][&level].clone())?;
    if !super::measured_prefix::problems(
        &raw,
        &super::measured_prefix::Qualification {
            expected_prompt_tokens,
            require_later_turn_reuse: true,
        },
    )
    .is_empty()
    {
        return Err("measured prompt prefix or later-turn reuse differs from preflight".into());
    }
    let recomputed = super::cell_summary::summarize(trajectories, &raw, level.parse()?)?;
    for field in [
        "requests",
        "successful_requests",
        "failed_requests",
        "completion_tokens",
        "generation_seconds",
        "workload_window_seconds",
        "ttft_samples",
        "cache_pct",
        "finish_reason_length_requests",
    ] {
        let actual = &recomputed[field];
        let saved = &cell.value[field];
        let equal = if actual.is_number() && saved.is_number() {
            actual.as_f64() == saved.as_f64()
        } else {
            actual == saved
        };
        if !equal {
            return Err(format!("raw metric differs from summary: {field}").into());
        }
    }
    let success: BTreeSet<_> = complete.expected_request_ids.iter().collect();
    let saved: Vec<String> = serde_json::from_value(cell.value["successful_request_ids"].clone())?;
    if saved.len() != success.len() || saved.iter().collect::<BTreeSet<_>>() != success {
        return Err("successful request identities differ from raw evidence".into());
    }
    if model.class == "hybrid-recurrent" {
        let mut paths = vec![
            cell.directory.join("mesh.log"),
            cell.directory.join("mesh.stderr.log"),
        ];
        let native = cell.directory.join("native-runtime");
        if native.try_exists()? {
            collect_logs(&native, &mut paths)?;
        }
        let lookups = super::recurrent_evidence::read_logs(&paths)?;
        let recurrent = super::recurrent_evidence::evaluate(
            &requests,
            &lookups,
            input.replay.minimum_session_prompt_tokens.max(1),
        );
        if !recurrent.passed || serde_json::to_value(recurrent)? != cell.value["recurrent_state"] {
            return Err("recurrent restore evidence differs from native lookup events".into());
        }
    }
    Ok(())
}
pub(super) fn records(path: &Path) -> DynResult<Vec<Value>> {
    use std::io::{BufRead, BufReader};
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("history JSONL must be a regular file".into());
    }
    BufReader::new(std::fs::File::open(path)?)
        .lines()
        .filter_map(|line| match line {
            Ok(text) if text.trim().is_empty() => None,
            other => Some(other),
        })
        .map(|line| -> DynResult<Value> { Ok(serde_json::from_str(&line?)?) })
        .collect()
}
fn coverage(input: &Input, family: &mut Family) -> DynResult<()> {
    let expected = (1..=input.replay.passes)
        .flat_map(|pass| {
            input
                .replay
                .concurrency
                .iter()
                .map(move |level| (pass, *level))
        })
        .collect::<BTreeSet<_>>();
    let observed = family
        .cells
        .iter()
        .map(|cell| {
            Ok((
                cell.pass,
                usize::try_from(
                    cell.value["concurrency"]
                        .as_u64()
                        .ok_or("missing cell concurrency")?,
                )?,
            ))
        })
        .collect::<DynResult<BTreeSet<_>>>()?;
    if observed != expected || family.cells.len() != expected.len() {
        family
            .problems
            .push("incomplete, duplicate or foreign pass/concurrency matrix".into());
    }
    Ok(())
}
fn collect_logs(root: &Path, paths: &mut Vec<PathBuf>) -> DynResult<()> {
    directory(root)?;
    for entry in sorted(root)? {
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("native log evidence must not be symlinked".into());
        }
        if kind.is_dir() {
            collect_logs(&entry.path(), paths)?;
        } else if kind.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "log")
        {
            paths.push(entry.path());
        }
    }
    Ok(())
}
fn directory(path: &Path) -> DynResult<()> {
    if !std::fs::symlink_metadata(path)?.file_type().is_dir() {
        return Err("history artifact directory must be regular".into());
    }
    Ok(())
}
fn sorted(path: &Path) -> DynResult<Vec<std::fs::DirEntry>> {
    directory(path)?;
    let mut entries = std::fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    Ok(entries)
}

fn accepted_run_gates(gates: &Value) -> bool {
    gates["passed"] == true
        || (gates["evaluated"] == false
            && gates["passed"].is_null()
            && gates["checks"].as_array().is_some_and(Vec::is_empty)
            && gates["session_acceptance_failures"]
                .as_array()
                .is_some_and(Vec::is_empty))
}

fn require_full_replay(document: &Value) -> DynResult<()> {
    match document["config"].get("replay_mode") {
        None => Ok(()),
        Some(Value::String(mode)) if mode == "all" => Ok(()),
        _ => Err("history requires complete all-session replay, not selected checkpoints".into()),
    }
}
#[cfg(test)]
mod profile_tests {
    use super::*;
    #[test]
    fn history_profiles_require_all_while_legacy_absence_still_needs_existing_raw_proofs() {
        assert!(require_full_replay(&serde_json::json!({"config":{"replay_mode":"all"}})).is_ok());
        assert!(require_full_replay(&serde_json::json!({"config":{}})).is_ok());
        for mode in [
            serde_json::json!("final"),
            serde_json::json!("checkpoint"),
            Value::Null,
            serde_json::json!({}),
        ] {
            assert!(
                require_full_replay(&serde_json::json!({"config":{"replay_mode":mode}})).is_err()
            );
        }
    }
}
