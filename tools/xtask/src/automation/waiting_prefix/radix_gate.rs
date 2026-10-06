//! Original regression gate: conditional N1 correctness and per-round concurrent hit tolerance.
use super::radix_summary::{self, key};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
pub(super) fn evaluate(payload: &str, cells: &[Value], rows: &[Value]) -> Value {
    let mut failures = vec![];
    let required = if payload == "resident-kv" {
        vec!["exact", "divergent", "coding"]
    } else {
        vec!["exact"]
    };
    for cell in cells {
        if cell["suspect_log"].as_bool().unwrap_or(false) {
            failures.push("radix serving arm emitted a suspect runtime diagnostic".into());
        }
        for observation in cell["observations"].as_array().into_iter().flatten() {
            let s = &observation["summary"];
            let n = observation["concurrency"].as_u64().unwrap_or(0);
            let scenario = observation["scenario"].as_str().unwrap_or("");
            if s["successful"] != s["requests"] {
                failures.push(format!("{scenario}/n{n} did not complete every request"));
            }
            if n == 1
                && s["outputs_by_prompt"].as_object().is_none_or(|m| {
                    m.values()
                        .any(|v| v.as_array().is_none_or(|a| a.len() != 1))
                })
            {
                failures.push(format!(
                    "{scenario}/n1 produced missing/nondeterministic prompt output"
                ));
            }
            if cell["cache"] == "warm"
                && required.contains(&scenario)
                && n == 1
                && s["cache_hits"] != s["requests"]
            {
                failures.push(format!(
                    "{scenario}/n1 did not report a cache hit for every request"
                ));
            }
        }
    }
    let (parity, preservation) = radix_summary::comparisons(rows);
    let preserved = preservation
        .iter()
        .map(|r| {
            (
                (
                    r["version"].as_str().unwrap_or(""),
                    r["scenario"].as_str().unwrap_or(""),
                    r["concurrency"].as_u64().unwrap_or(0),
                ),
                r["cache_preserves_output"].as_bool().unwrap_or(false),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for row in &preservation {
        let scenario = row["scenario"].as_str().unwrap_or("");
        if row["version"] == "new"
            && row["concurrency"] == 1
            && row["cache_preserves_output"] == false
            && preserved
                .get(&("old", scenario, 1))
                .copied()
                .unwrap_or(true)
        {
            failures.push(format!(
                "NEW cache introduced an N1 output mismatch absent from OLD for {scenario}"
            ));
        }
    }
    for row in &parity {
        if row["cache"] == "cold" && row["concurrency"] == 1 && row["identical_outputs"] == false {
            failures.push("OLD/NEW cold N1 output differs".into());
        }
    }
    let indexed = rows.iter().map(|r| (key(r), r)).collect::<BTreeMap<_, _>>();
    let tolerance = cells
        .iter()
        .filter(|c| c["version"] == "new" && c["cache"] == "warm")
        .filter_map(|c| c["round"].as_u64())
        .collect::<BTreeSet<_>>()
        .len()
        .max(1) as u64;
    for scenario in &required {
        let levels = rows
            .iter()
            .filter(|r| r["cache"] == "warm" && r["scenario"] == *scenario)
            .filter_map(|r| r["concurrency"].as_u64().filter(|n| *n > 1))
            .collect::<BTreeSet<_>>();
        for n in levels {
            match (
                indexed.get(&("old".into(), "warm".into(), (*scenario).into(), n)),
                indexed.get(&("new".into(), "warm".into(), (*scenario).into(), n)),
            ) {
                (Some(old), Some(new)) => {
                    if new["cache_hits"]
                        .as_u64()
                        .unwrap_or(0)
                        .saturating_add(tolerance)
                        < old["cache_hits"].as_u64().unwrap_or(0)
                    {
                        failures.push(format!("NEW warm hits regressed beyond the {tolerance}-request round tolerance for {scenario}/n{n}"));
                    }
                }
                _ => failures.push(format!(
                    "missing concurrent warm OLD/NEW pair for {scenario}/n{n}"
                )),
            }
        }
    }
    for n in rows
        .iter()
        .filter(|r| r["cache"] == "warm" && r["scenario"] == "divergent")
        .filter_map(|r| r["concurrency"].as_u64())
        .collect::<BTreeSet<_>>()
    {
        let old = indexed.get(&("old".into(), "warm".into(), "divergent".into(), n));
        let new = indexed.get(&("new".into(), "warm".into(), "divergent".into(), n));
        match (
            old.and_then(|r| r["suffix_prefill_tokens_median"].as_f64()),
            new.and_then(|r| r["suffix_prefill_tokens_median"].as_f64()),
        ) {
            (Some(old), Some(new)) if new <= old => {}
            (Some(_), Some(_)) => failures.push(format!(
                "NEW divergent suffix prefill exceeded OLD at concurrency {n}"
            )),
            _ => failures.push(format!(
                "missing divergent suffix telemetry at concurrency {n}"
            )),
        }
    }
    json!({"passed":failures.is_empty(),"failures":failures})
}
