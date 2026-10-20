//! A fixed, complete local workload; independent of engine performance.
use super::{Prepared, inputs, trace_context, write};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader},
    path::Path,
};

// Pinned corpus: two <=64K, one 64–96K, one 96–128K peak-input trace.
// Within each band, smallest output totals among traces spanning <=600s.
// All requests, outputs, recorded delays and child topology remain unchanged.
pub(super) const IDS: [&str; 4] = [
    "12cfecf98c158ab54fb65980602421f00d69",
    "1493faffdc8942e99daa22277f44871a40b8",
    "9ec047754f4ddd94a59fcd0d85213d5f3b1b",
    "bbdcb12440a7ab3496b9fac8b5f9824b1672",
];

pub(super) fn materialize(source: &Path, destination: &Path) -> DynResult<()> {
    let mut selected = BTreeMap::new();
    for line in BufReader::new(std::fs::File::open(source)?).lines() {
        let line = line?;
        let trace: Value = serde_json::from_str(&line)?;
        let id = trace["id"].as_str().ok_or("trace ID missing")?;
        if IDS.contains(&id) && selected.insert(id.to_owned(), line).is_some() {
            return Err("duplicate comparison trace".into());
        }
    }
    if selected.len() != IDS.len() {
        return Err("fixed comparison cohort missing complete traces".into());
    }
    let mut audit = Vec::new();
    let mut expected = BTreeMap::new();
    for (index, (id, line)) in selected.iter().enumerate() {
        let trace: Value = serde_json::from_str(line)?;
        let description = trace_context::describe(&trace, line.as_bytes())?;
        if description["status"] != "eligible" {
            return Err("fixed comparison trace exceeds common context budget".into());
        }
        expected.extend(expected_turns(id, &trace)?);
        audit.push(description);
        std::fs::write(destination.join(format!("{index:04}.json")), line)?;
    }
    let parent = destination.parent().ok_or("comparison cohort parent")?;
    write(&parent.join("expected-work.json"), &expected)?;
    write(
        &parent.join("cohort.json"),
        &json!({
            "dataset_repo":inputs::DATA_REPO,"dataset_revision":inputs::DATA_REVISION,
            "dataset_sha256":inputs::DATA_SHA,"trace_ids":IDS,"trace_context":audit,
            "selection":"fixed_complete_context_stratified_v1",
            "expected_requests":expected.len(),"expected_output_tokens":expected.values().sum::<u64>(),
            "request_policy":"each_source_turn_exactly_once_no_dispatch_cutoff"
        }),
    )
}

pub(super) fn expected_turns(id: &str, trace: &Value) -> DynResult<BTreeMap<String, u64>> {
    let mut turns = BTreeMap::new();
    let requests = trace["requests"]
        .as_array()
        .ok_or("missing trace requests")?;
    for (outer, request) in requests.iter().enumerate() {
        if request["type"] == "subagent" {
            let children = request["requests"]
                .as_array()
                .ok_or("missing child requests")?;
            for (inner, child) in children.iter().enumerate() {
                turns.insert(
                    turn_key(id, outer as u64, Some(inner as u64)),
                    output(child)?,
                );
            }
        } else {
            turns.insert(turn_key(id, outer as u64, None), output(request)?);
        }
    }
    Ok(turns)
}
fn output(request: &Value) -> DynResult<u64> {
    if !matches!(request["type"].as_str(), Some("n" | "s")) {
        return Err("unsupported comparison source request topology".into());
    }
    Ok(request["out"]
        .as_u64()
        .ok_or("invalid requested output")?
        .max(1))
}
pub(super) fn turn_key(id: &str, outer: u64, inner: Option<u64>) -> String {
    format!(
        "{id}/{outer}/{}",
        inner.map_or_else(|| "root".into(), |i| i.to_string())
    )
}

pub(super) fn configuration(input: &Prepared, port: u16, concurrency: u32) -> DynResult<Value> {
    if ![1, 2, 4].contains(&concurrency) {
        return Err("comparison supports c=1/2/4".into());
    }
    let cache = std::env::var_os("HF_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| input.root.join(".cache/huggingface"));
    let tokenizer = cache
        .join("hub")
        .join(format!(
            "models--{}",
            input.model.tokenizer.replace('/', "--")
        ))
        .join("snapshots")
        .join(&input.model.tokenizer_revision);
    // Explicit local config avoids AgentX scenario defaults reinstating a
    // duration or advancing traces during one-token cache-pressure warmup.
    Ok(json!({"schema_version":"2.0","random_seed":42,"benchmark":{
        "model":input.model.key,
        "endpoint":{"url":format!("http://127.0.0.1:{port}/v1/chat/completions"),
            "type":"chat","streaming":true,"timeout":1800,
            "use_server_token_count":true,"session_header":"X-Session-ID","extra":{"ignore_eos":true}},
        "tokenizer":{"name":tokenizer,"revision":input.model.tokenizer_revision},
        "dataset":{"type":"file","path":input.dataset,"format":"weka_trace",
            "sampling":"sequential","entries":4,"max_context_length":trace_context::LIMIT,
            "ignore_trace_delays":false,"use_think_time_only":true,
            "cache_bust":{"target":"first_turn_prefix"}},
        "phases":[{"name":"profiling","type":"concurrency","timing_mode":"agentic_replay",
            "concurrency":concurrency,"sessions":4,"trajectory_start_min_ratio":0,
            "trajectory_start_max_ratio":0,"burst_phase_starts":true}],
        // An explicit processor is a required startup service. Auto-scaling
        // can register it after the one-shot dataset notification and strand
        // every metric record, even with dataset mmap caching disabled.
        "runtime":{"record_processors":1},
        "artifacts":{"dir":input.output.join("upstream"),"summary":["json"],
            "records":["jsonl"],"trace":true},"gpu_telemetry":{"enabled":false}
    }}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_finishes_sessions_without_a_duration_or_warmup_cutoff() {
        let models: Vec<super::super::Model> = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/agentx/models.json"
        ))
        .unwrap();
        let input = Prepared {
            root: "/repo".into(),
            output: "/results".into(),
            weights: "/weights/model.gguf".into(),
            dataset: "/traces".into(),
            model: models[0].clone(),
            profile: super::super::Profile::Comparison,
        };
        for concurrency in [1, 2, 4] {
            let config = configuration(&input, 8080, concurrency).unwrap();
            let benchmark = &config["benchmark"];
            assert!(benchmark["scenario"].is_null());
            assert_eq!(
                benchmark["phases"],
                json!([{
                    "name":"profiling", "type":"concurrency", "timing_mode":"agentic_replay",
                    "concurrency":concurrency, "sessions":4, "trajectory_start_min_ratio":0,
                    "trajectory_start_max_ratio":0, "burst_phase_starts":true
                }])
            );
            assert_eq!(benchmark["dataset"]["entries"], IDS.len());
            assert_eq!(benchmark["dataset"]["max_context_length"], 131072);
            assert_eq!(benchmark["dataset"]["use_think_time_only"], true);
            assert_eq!(benchmark["endpoint"]["extra"]["ignore_eos"], true);
            assert_eq!(benchmark["endpoint"]["use_server_token_count"], true);
            assert_eq!(benchmark["artifacts"]["records"], json!(["jsonl"]));
            assert_eq!(benchmark["runtime"]["record_processors"], 1);
        }
        assert!(configuration(&input, 8080, 8).is_err());
    }
    #[test]
    fn expected_work_includes_children_and_zero_output_reservation() {
        let trace = json!({"requests":[{"type":"n","out":3},
            {"type":"subagent","requests":[{"type":"s","out":0},{"type":"n","out":5}]}]});
        assert_eq!(
            expected_turns("t", &trace).unwrap(),
            BTreeMap::from([
                ("t/0/root".into(), 3),
                ("t/1/0".into(), 1),
                ("t/1/1".into(), 5)
            ])
        );
        assert!(
            expected_turns(
                "t",
                &json!({"requests":[{"type":"subagent","requests":[{"type":"subagent"}]}]})
            )
            .is_err()
        );
    }
    #[test]
    fn materialization_retains_exact_rows_and_requires_every_selected_trace() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.jsonl");
        let dest = dir.path().join("traces");
        std::fs::create_dir(&dest).unwrap();
        let rows: Vec<_> = IDS
            .iter()
            .map(|id| json!({"id":id,"requests":[{"type":"n","in":120000,"out":2}]}).to_string())
            .collect();
        std::fs::write(&source, rows.join("\n")).unwrap();
        materialize(&source, &dest).unwrap();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                std::fs::read_to_string(dest.join(format!("{i:04}.json"))).unwrap(),
                *row
            );
        }
        std::fs::write(&source, rows[..3].join("\n")).unwrap();
        assert!(materialize(&source, &dest).is_err());
    }
}
