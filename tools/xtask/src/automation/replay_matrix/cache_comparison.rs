//! Paired cache evidence, separate from cross-engine performance results.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

pub(super) fn write(output: &Path, expected: usize) -> DynResult<()> {
    let read = |label| -> DynResult<Vec<Value>> {
        std::fs::read_to_string(output.join(format!("data/pass-1/{label}/c-1-requests.jsonl")))?
            .lines()
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    };
    let report = compare(&read("cache-off")?, &read("cache-on")?, expected)?;
    crate::command::write_json_file(&output.join("cache-comparison.json"), &report)?;
    if report["passed"] != true {
        return Err("cache comparison requires investigation; paired evidence retained".into());
    }
    Ok(())
}

fn indexed(records: &[Value]) -> DynResult<BTreeMap<&str, &Value>> {
    let mut result = BTreeMap::new();
    for record in records {
        let id = record["request_id"]
            .as_str()
            .ok_or("missing request identity")?;
        if result.insert(id, record).is_some() {
            return Err("duplicate request identity".into());
        }
    }
    Ok(result)
}

fn compare(off: &[Value], on: &[Value], expected: usize) -> DynResult<Value> {
    let off = indexed(off)?;
    let on = indexed(on)?;
    let mut problems = Vec::new();
    if off.len() != expected || on.len() != expected || off.keys().ne(on.keys()) {
        problems.push("incomplete or unaligned request sets");
    }
    let mut hits = 0;
    let mut divergences = Vec::new();
    for (id, cold) in &off {
        let Some(warm) = on.get(id) else { continue };
        if super::output_integrity::incomplete(cold) || super::output_integrity::incomplete(warm) {
            problems.push("incomplete response");
        }
        if cold["cached_tokens"].as_u64() != Some(0) {
            problems.push("cache-off did not report zero cached tokens");
        }
        if warm["cached_tokens"].as_u64().is_some_and(|v| v > 0) {
            hits += 1;
        }
        if cold["request_sha256"].as_str().is_none()
            || cold["request_sha256"] != warm["request_sha256"]
            || cold["prompt_tokens"] != warm["prompt_tokens"]
        {
            problems.push("request or prompt-token identity differs");
        }
        if cold["content_sha256"].as_str().is_none()
            || cold["content_sha256"] != warm["content_sha256"]
            || cold["completion_tokens"] != warm["completion_tokens"]
            || cold["finish_reason"] != warm["finish_reason"]
        {
            divergences.push(json!({"request_id":id,"cache_off":cold,"cache_on":warm}));
        }
    }
    if hits == 0 {
        problems.push("cache-on had no measured cache hits");
    }
    problems.sort_unstable();
    problems.dedup();
    Ok(
        json!({"passed":problems.is_empty() && divergences.is_empty(),
        "cache_hit_requests":hits,"paired_requests":off.keys().filter(|id| on.contains_key(**id)).count(),
        "problems":problems,"divergences":divergences,
        "interpretation":"Divergence needs a matched-batching native/logit investigation; it alone does not establish KV corruption."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(id: &str, cached: u64) -> Value {
        json!({"request_id":id,"request_sha256":"same-request","content_sha256":"same-output",
            "prompt_tokens":1000,"cached_tokens":cached,"completion_tokens":20,
            "finish_reason":"stop","output_problems":[]})
    }
    #[test]
    fn requires_actual_reuse_and_aligned_complete_outputs() {
        let off = vec![record("a", 0), record("b", 0)];
        let mut on = vec![record("a", 0), record("b", 512)];
        assert_eq!(compare(&off, &on, 2).unwrap()["passed"], true);
        on[1]["cached_tokens"] = 0.into();
        assert_eq!(compare(&off, &on, 2).unwrap()["passed"], false);
        assert_eq!(compare(&off, &off[..1], 2).unwrap()["passed"], false);
        assert!(compare(&off, &[off[0].clone(), off[0].clone()], 2).is_err());
    }
    #[test]
    fn retains_divergence_and_rejects_wrong_inputs_or_truncation() {
        let off = vec![record("a", 0)];
        for (field, value) in [
            ("request_sha256", json!("different")),
            ("content_sha256", json!("different")),
            ("finish_reason", json!("length")),
            ("output_problems", json!(["truncated"])),
        ] {
            let mut on = vec![record("a", 512)];
            on[0][field] = value;
            let report = compare(&off, &on, 1).unwrap();
            assert_eq!(report["passed"], false, "{field}");
            if field == "content_sha256" {
                assert_eq!(report["divergences"][0]["cache_on"], on[0]);
            }
        }
    }
}
