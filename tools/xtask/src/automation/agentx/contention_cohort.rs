//! Bounded two-turn sessions for concurrency diagnosis.
use super::{Prepared, comparison_cohort, inputs, trace_context, write};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader},
    path::Path,
};

pub(super) const SESSIONS: usize = 16;
const OUTPUT_TOKENS_PER_TURN: u64 = 128;
const INPUT_TOKENS_PER_TURN_CAP: u64 = 8192;
const MAX_SECOND_TURN_DELAY_SECS: f64 = 60.0;

struct Candidate {
    id: String,
    source_requested_output_tokens: u64,
    source_requested_turns: usize,
    profile_requested_output_tokens: u64,
    line: String,
    source_context: Value,
    profile_context: Value,
}

pub(super) fn materialize(source: &Path, destination: &Path) -> DynResult<()> {
    let mut seen = BTreeSet::new();
    let mut eligible = Vec::new();
    for line in BufReader::new(std::fs::File::open(source)?).lines() {
        let line = line?;
        let trace: Value = serde_json::from_str(&line)?;
        let id = trace["id"].as_str().ok_or("trace ID missing")?.to_owned();
        if !seen.insert(id.clone()) {
            return Err("duplicate contention trace".into());
        }
        let source_context = trace_context::describe(&trace, line.as_bytes())?;
        if source_context["status"] != "eligible" {
            continue;
        }
        let Some(profile) = two_turn_profile(&trace)? else {
            continue;
        };
        let turns = comparison_cohort::expected_turns(&id, &trace)?;
        let profile_turns = comparison_cohort::expected_turns(&id, &profile)?;
        let profile_line = serde_json::to_string(&profile)?;
        let profile_context = trace_context::describe(&profile, profile_line.as_bytes())?;
        eligible.push(Candidate {
            id,
            source_requested_output_tokens: turns.values().sum(),
            source_requested_turns: turns.len(),
            profile_requested_output_tokens: profile_turns.values().sum(),
            line: profile_line,
            source_context,
            profile_context,
        });
    }
    eligible.sort_by(|a, b| {
        a.source_requested_output_tokens
            .cmp(&b.source_requested_output_tokens)
            .then_with(|| a.id.cmp(&b.id))
    });
    if eligible.len() < SESSIONS {
        return Err("fewer than 16 two-turn context-compatible AgentX traces".into());
    }
    let selected: Vec<_> = eligible.into_iter().take(SESSIONS).collect();
    let parent = destination.parent().ok_or("contention cohort parent")?;
    let audit: Vec<_> = selected
        .iter()
        .map(|candidate| candidate.profile_context.clone())
        .collect();
    let mut expected = BTreeMap::new();
    for (index, candidate) in selected.iter().enumerate() {
        std::fs::write(
            destination.join(format!("{index:04}.json")),
            &candidate.line,
        )?;
        let trace: Value = serde_json::from_str(&candidate.line)?;
        expected.extend(comparison_cohort::expected_turns(&candidate.id, &trace)?);
    }
    write(&parent.join("expected-work.json"), &expected)?;
    write(
        &parent.join("cohort.json"),
        &json!({
            "dataset_repo":inputs::DATA_REPO,"dataset_revision":inputs::DATA_REVISION,
            "dataset_sha256":inputs::DATA_SHA,
            "trace_ids":selected.iter().map(|candidate| &candidate.id).collect::<Vec<_>>(),
            "trace_context":audit,
            "source_trace_context":selected.iter().map(|candidate| &candidate.source_context).collect::<Vec<_>>(),
            "selection":"16_lowest_source_output_two_root_turns_input8k_output128_v2",
            "source_requested_turns":selected.iter().map(|candidate| candidate.source_requested_turns).sum::<usize>(),
            "source_requested_output_tokens":selected.iter().map(|candidate| candidate.source_requested_output_tokens).sum::<u64>(),
            "profile_requested_turns":SESSIONS * 2,
            "profile_requested_output_tokens":selected.iter().map(|candidate| candidate.profile_requested_output_tokens).sum::<u64>(),
            "output_tokens_per_turn_cap":OUTPUT_TOKENS_PER_TURN,
            "input_tokens_per_turn_cap":INPUT_TOKENS_PER_TURN_CAP,
            "max_second_turn_delay_seconds":MAX_SECOND_TURN_DELAY_SECS,
            "request_policy":"fixed_16_two_turn_sessions_not_full_agentx_qualification",
        }),
    )
}

pub(super) fn configuration(input: &Prepared, port: u16, concurrency: u32) -> DynResult<Value> {
    super::agentic_config::configuration(
        input,
        port,
        concurrency,
        SESSIONS,
        "mesh-agentx-contention-v2",
    )
}

fn two_turn_profile(trace: &Value) -> DynResult<Option<Value>> {
    let requests = trace["requests"]
        .as_array()
        .ok_or("missing trace requests")?;
    let [first, second, ..] = requests.as_slice() else {
        return Ok(None);
    };
    if ![first, second]
        .iter()
        .all(|turn| matches!(turn["type"].as_str(), Some("n" | "s")))
    {
        return Ok(None);
    }
    let (Some(first_at), Some(second_at)) = (first["t"].as_f64(), second["t"].as_f64()) else {
        return Ok(None);
    };
    if !(0.0..=MAX_SECOND_TURN_DELAY_SECS).contains(&(second_at - first_at)) {
        return Ok(None);
    }
    let block_size = trace["block_size"]
        .as_u64()
        .ok_or("invalid trace block size")?;
    if block_size == 0 {
        return Err("zero trace block size".into());
    }
    let mut selected = Vec::with_capacity(2);
    for turn in [first, second] {
        let output = turn["out"].as_u64().ok_or("invalid requested output")?;
        let input = turn["in"].as_u64().ok_or("invalid requested input")?;
        let capped_input = input.min(INPUT_TOKENS_PER_TURN_CAP);
        let mut capped = turn.clone();
        capped["out"] = json!(output.clamp(1, OUTPUT_TOKENS_PER_TURN));
        capped["in"] = json!(capped_input);
        if let Some(hashes) = turn["hash_ids"].as_array() {
            let blocks = capped_input.div_ceil(block_size) as usize;
            capped["hash_ids"] = json!(hashes.iter().take(blocks).collect::<Vec<_>>());
        }
        selected.push(capped);
    }
    let mut profile = trace.clone();
    profile["requests"] = json!(selected);
    Ok(Some(profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_the_same_bounded_two_turn_traces_independent_of_source_order() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.jsonl");
        let rows: Vec<_> = (0..20)
            .map(|index| {
                json!({"id":format!("trace-{index:02}"),"block_size":64,
                    "requests":[{"type":"n","t":0,"in":100,"out":20-index},
                        {"type":"s","t":1,"in":10000,"out":256,"hash_ids":(0..157).collect::<Vec<_>>()},
                        {"type":"n","t":2,"in":300,"out":512}]})
                .to_string()
            })
            .collect();
        std::fs::write(&source, rows.join("\n")).unwrap();
        let first = dir.path().join("first/traces");
        std::fs::create_dir_all(&first).unwrap();
        materialize(&source, &first).unwrap();
        let first_cohort: Value = serde_json::from_slice(
            &std::fs::read(first.parent().unwrap().join("cohort.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            first_cohort["trace_ids"].as_array().unwrap().len(),
            SESSIONS
        );
        assert_eq!(first_cohort["trace_ids"][0], "trace-19");
        assert_eq!(
            first_cohort["source_requested_output_tokens"],
            136 + 16 * 768
        );
        assert_eq!(
            first_cohort["profile_requested_output_tokens"],
            136 + 16 * 128
        );
        assert_eq!(first_cohort["profile_requested_turns"], 32);
        let expected: BTreeMap<String, u64> = serde_json::from_slice(
            &std::fs::read(first.parent().unwrap().join("expected-work.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(expected.len(), 32);
        assert_eq!(expected.values().sum::<u64>(), 136 + 16 * 128);
        let first_trace: Value =
            serde_json::from_slice(&std::fs::read(first.join("0000.json")).unwrap()).unwrap();
        assert_eq!(first_trace["requests"].as_array().unwrap().len(), 2);
        assert_eq!(first_trace["requests"][1]["out"], 128);
        assert_eq!(first_trace["requests"][1]["in"], 8192);
        assert_eq!(
            first_trace["requests"][1]["hash_ids"]
                .as_array()
                .unwrap()
                .len(),
            128
        );
        std::fs::write(
            &source,
            rows.into_iter().rev().collect::<Vec<_>>().join("\n"),
        )
        .unwrap();
        let second = dir.path().join("second/traces");
        std::fs::create_dir_all(&second).unwrap();
        materialize(&source, &second).unwrap();
        let second_cohort: Value = serde_json::from_slice(
            &std::fs::read(second.parent().unwrap().join("cohort.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(first_cohort, second_cohort);
    }

    #[test]
    fn excludes_delayed_or_branching_first_followups() {
        let ordinary = json!({"block_size":64,"requests":[{"type":"n","t":0,"in":100,"out":1},
            {"type":"s","t":2,"in":200,"out":1}]});
        assert!(two_turn_profile(&ordinary).unwrap().is_some());
        let delayed = json!({"block_size":64,"requests":[{"type":"n","t":0,"in":100,"out":1},
            {"type":"s","t":61,"in":200,"out":1}]});
        assert!(two_turn_profile(&delayed).unwrap().is_none());
        let branched = json!({"block_size":64,"requests":[{"type":"n","t":0,"in":100,"out":1},
            {"type":"subagent","requests":[]}]});
        assert!(two_turn_profile(&branched).unwrap().is_none());
    }

    #[test]
    fn config_runs_all_sessions_without_scenario_warmup_or_duration() {
        let models: Vec<super::super::Model> = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/kv-validation/models.json"
        ))
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dataset = dir.path().join("traces");
        std::fs::create_dir(&dataset).unwrap();
        std::fs::write(dir.path().join("cohort.json"), b"bounded source cohort").unwrap();
        let input = Prepared {
            tools: None,
            root: "/repo".into(),
            output: "/results".into(),
            weights: "/weights/model.gguf".into(),
            dataset,
            model: models[1].clone(),
            profile: super::super::Profile::Contention,
        };
        let config = configuration(&input, 8080, 16).unwrap();
        let benchmark = &config["benchmark"];
        assert_eq!(benchmark["dataset"]["entries"], SESSIONS);
        assert_eq!(benchmark["phases"][0]["sessions"], SESSIONS);
        assert_eq!(benchmark["phases"][0]["concurrency"], 16);
        assert!(benchmark["phases"][0]["duration"].is_null());
        assert!(benchmark["phases"][0]["warmup_requests_per_lane"].is_null());
        assert!(benchmark["scenario"].is_null());
    }
}
