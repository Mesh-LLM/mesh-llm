//! Refuse fixed-work throughput if any source request is missing or repeated.
use super::{Prepared, comparison_cohort::turn_key, write};
use crate::command::DynResult;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader},
};

pub(super) fn validate(input: &Prepared) -> DynResult<Value> {
    let expected: BTreeMap<String, u64> = serde_json::from_slice(&std::fs::read(
        input
            .dataset
            .parent()
            .ok_or("cohort parent")?
            .join("expected-work.json"),
    )?)?;
    let file = std::fs::File::open(input.output.join("upstream/profile_export.jsonl"))?;
    let mut observed = BTreeMap::new();
    for line in BufReader::new(file).lines() {
        let record: Value = serde_json::from_str(&line?)?;
        if record["metadata"]["benchmark_phase"] == "profiling" {
            add(&mut observed, &record)?;
        }
    }
    check(&expected, &observed)?;
    let result = json!({"status":"complete","requests":observed.len(),
        "output_tokens":observed.values().sum::<u64>(),
        "source_turn_output_sha256":hex::encode(Sha256::digest(serde_json::to_vec(&observed)?)),
        "source_turns":observed});
    write(&input.output.join("completed-work.json"), &result)?;
    Ok(result)
}
fn add(observed: &mut BTreeMap<String, u64>, record: &Value) -> DynResult<()> {
    let m = &record["metadata"];
    if m["was_cancelled"] != false
        || m["context_overflow_skip"] == true
        || !record["error"].is_null()
    {
        return Err("fixed-work request failed, cancelled or overflowed".into());
    }
    let id = m["source_trace_id"]
        .as_str()
        .ok_or("missing source trace ID")?;
    let outer = m["source_outer_idx"]
        .as_u64()
        .ok_or("missing source turn index")?;
    let inner = if m["source_inner_idx"].is_null() {
        None
    } else {
        Some(
            m["source_inner_idx"]
                .as_u64()
                .ok_or("invalid source child index")?,
        )
    };
    let output = record["metrics"]["output_token_count"]["value"]
        .as_f64()
        .ok_or("missing output count")?;
    if !output.is_finite() || output < 1.0 || output.fract() != 0.0 || output > (1_u64 << 53) as f64
    {
        return Err("invalid actual output count".into());
    }
    if observed
        .insert(turn_key(id, outer, inner), output as u64)
        .is_some()
    {
        return Err("fixed-work source turn repeated".into());
    }
    Ok(())
}
fn check(expected: &BTreeMap<String, u64>, observed: &BTreeMap<String, u64>) -> DynResult<()> {
    if expected.is_empty() || expected != observed {
        return Err(format!("fixed-work completion mismatch: expected {} source turns, observed {}; no comparable throughput",expected.len(),observed.len()).into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_harness_records_cover_complete_roots_and_children() {
        // Reduced actual 1.0.12.post1 export from a four-root mock replay.
        let records = include_str!("../../../tests/fixtures/agentx/complete-trace-records.jsonl");
        let mut observed = BTreeMap::new();
        for line in records.lines() {
            add(&mut observed, &serde_json::from_str(line).unwrap()).unwrap();
        }
        let trace = json!({"requests":[{"type":"n","out":8},
            {"type":"subagent","requests":[{"type":"n","out":3},{"type":"n","out":4}]},
            {"type":"s","out":5}]});
        let expected = ["fixture-a", "fixture-1", "fixture-2", "fixture-3"]
            .into_iter()
            .flat_map(|id| super::super::comparison_cohort::expected_turns(id, &trace).unwrap())
            .collect();
        check(&expected, &observed).unwrap();
        assert_eq!(observed.len(), 16);
        assert_eq!(observed.values().sum::<u64>(), 80);
    }
    #[test]
    fn missing_extra_and_shortened_outputs_fail_closed() {
        let expected = BTreeMap::from([("t/0/root".into(), 8), ("t/1/0".into(), 3)]);
        assert!(check(&expected, &expected).is_ok());
        for observed in [
            BTreeMap::new(),
            BTreeMap::from([("t/0/root".into(), 8)]),
            BTreeMap::from([("t/0/root".into(), 8), ("t/1/0".into(), 2)]),
            BTreeMap::from([
                ("t/0/root".into(), 8),
                ("t/1/0".into(), 3),
                ("t/2/root".into(), 1),
            ]),
        ] {
            assert!(check(&expected, &observed).is_err());
        }
    }
    #[test]
    fn duplicate_and_failed_source_turns_are_rejected() {
        let record = json!({"metadata":{"source_trace_id":"t","source_outer_idx":1,"source_inner_idx":0,
            "was_cancelled":false},"metrics":{"output_token_count":{"value":3.0}}});
        let mut observed = BTreeMap::new();
        add(&mut observed, &record).unwrap();
        assert_eq!(observed["t/1/0"], 3);
        assert!(add(&mut observed, &record).is_err());
        let mut cancelled = record.clone();
        cancelled["metadata"]["was_cancelled"] = json!(true);
        assert!(add(&mut BTreeMap::new(), &cancelled).is_err());
        for (key, value) in [
            ("context_overflow_skip", json!(true)),
            ("was_cancelled", Value::Null),
        ] {
            let mut failed = record.clone();
            failed["metadata"][key] = value;
            assert!(add(&mut BTreeMap::new(), &failed).is_err());
        }
        let mut failed = record;
        failed["error"] = json!({"message":"HTTP 429"});
        assert!(add(&mut BTreeMap::new(), &failed).is_err());
    }
}
