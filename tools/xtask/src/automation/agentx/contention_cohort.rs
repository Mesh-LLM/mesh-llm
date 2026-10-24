//! A fixed set of complete traces for duration-bounded concurrency diagnosis.
use super::{comparison_cohort, inputs, trace_context, write};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    io::{BufRead, BufReader},
    path::Path,
};

pub(super) const SESSIONS: usize = 16;
pub(super) const WINDOW_SECS: u32 = 180;

struct Candidate {
    id: String,
    requested_output_tokens: u64,
    requested_turns: usize,
    line: String,
    context: Value,
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
        let context = trace_context::describe(&trace, line.as_bytes())?;
        if context["status"] != "eligible" {
            continue;
        }
        let turns = comparison_cohort::expected_turns(&id, &trace)?;
        eligible.push(Candidate {
            id,
            requested_output_tokens: turns.values().sum(),
            requested_turns: turns.len(),
            line,
            context,
        });
    }
    eligible.sort_by(|a, b| {
        a.requested_output_tokens
            .cmp(&b.requested_output_tokens)
            .then_with(|| a.id.cmp(&b.id))
    });
    if eligible.len() < SESSIONS {
        return Err("fewer than 16 complete context-compatible AgentX traces".into());
    }
    let selected: Vec<_> = eligible.into_iter().take(SESSIONS).collect();
    let parent = destination.parent().ok_or("contention cohort parent")?;
    let audit: Vec<_> = selected
        .iter()
        .map(|candidate| candidate.context.clone())
        .collect();
    for (index, candidate) in selected.iter().enumerate() {
        std::fs::write(
            destination.join(format!("{index:04}.json")),
            &candidate.line,
        )?;
    }
    write(
        &parent.join("cohort.json"),
        &json!({
            "dataset_repo":inputs::DATA_REPO,"dataset_revision":inputs::DATA_REVISION,
            "dataset_sha256":inputs::DATA_SHA,
            "trace_ids":selected.iter().map(|candidate| &candidate.id).collect::<Vec<_>>(),
            "trace_context":audit,
            "selection":"16_lowest_total_requested_output_complete_traces_under_128k_v1",
            "source_requested_turns":selected.iter().map(|candidate| candidate.requested_turns).sum::<usize>(),
            "source_requested_output_tokens":selected.iter().map(|candidate| candidate.requested_output_tokens).sum::<u64>(),
            "request_policy":"duration_bounded_diagnostic_not_fixed_work_qualification",
            "window_seconds":WINDOW_SECS,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_the_same_complete_low_output_traces_independent_of_source_order() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.jsonl");
        let rows: Vec<_> = (0..20)
            .map(|index| {
                json!({"id":format!("trace-{index:02}"),
                    "requests":[{"type":"n","in":100,"out":20-index}]})
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
        assert_eq!(first_cohort["source_requested_output_tokens"], 136);
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
}
