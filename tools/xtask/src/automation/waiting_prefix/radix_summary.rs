//! Cache lift and per-prompt output preservation are independent reported evidence.
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let i = ((sorted.len() - 1) as f64 * q).round_ties_even() as usize;
    sorted.get(i).copied()
}
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        values[mid - 1] / 2.0 + values[mid] / 2.0
    } else {
        values[mid]
    })
}
pub(super) fn summarize(requests: &[Value], events: &[Value]) -> Value {
    let successful = requests
        .iter()
        .filter(|r| r["error"].is_null())
        .collect::<Vec<_>>();
    let attrs = events.iter().map(|e| &e["attributes"]).collect::<Vec<_>>();
    let numeric = |key: &str| {
        attrs
            .iter()
            .filter_map(|a| a[key].as_u64().map(|n| n as f64))
            .collect::<Vec<_>>()
    };
    let timing = |key: &str| {
        successful
            .iter()
            .filter_map(|r| r[key].as_f64().filter(|n| n.is_finite() && *n >= 0.0))
            .collect::<Vec<_>>()
    };
    let matched = numeric("skippy.kv.matched_prefix_tokens");
    let suffix = numeric("skippy.kv.suffix_prefill_tokens");
    let prompt = numeric("llama_stage.prompt_token_count");
    let ttft = timing("ttft_ms");
    let tpot = timing("tpot_ms");
    let mut outputs = BTreeMap::<String, BTreeSet<String>>::new();
    for r in &successful {
        if let (Some(prompt), Some(content)) =
            (r["prompt_sha256"].as_str(), r["content_sha256"].as_str())
        {
            outputs
                .entry(prompt.into())
                .or_default()
                .insert(content.into());
        }
    }
    let statuses = attrs
        .iter()
        .filter_map(|a| a["skippy.kv.status"].as_str())
        .collect::<Vec<_>>();
    json!({"requests":requests.len(),"successful":successful.len(),"errors":requests.iter().filter(|r|!r["error"].is_null()).collect::<Vec<_>>(),"cache_hits":statuses.iter().filter(|s|**s=="hit").count(),"cache_misses":statuses.iter().filter(|s|**s=="miss").count(),"cache_disabled":statuses.iter().filter(|s|**s=="disabled").count(),"matched_prefix_tokens":matched,"suffix_prefill_tokens":suffix,"prompt_tokens":prompt,"ttft_ms":ttft,"tpot_ms":tpot,"matched_prefix_tokens_median":median(&matched),"suffix_prefill_tokens_median":median(&suffix),"prompt_tokens_median":median(&prompt),"ttft_ms_p50":quantile(&ttft,0.5),"ttft_ms_p99":quantile(&ttft,0.99),"tpot_ms_p50":quantile(&tpot,0.5),"outputs_by_prompt":outputs,"radix_final":attrs.last().map(|a|a.as_object().map(|map|map.iter().filter(|(k,_)|k.starts_with("skippy.kv.radix.")).map(|(k,v)|(k.clone(),v.clone())).collect::<BTreeMap<_,_>>()))})
}
pub(super) fn key(row: &Value) -> (String, String, String, u64) {
    (
        row["version"].as_str().unwrap_or("").into(),
        row["cache"].as_str().unwrap_or("").into(),
        row["scenario"].as_str().unwrap_or("").into(),
        row["concurrency"].as_u64().unwrap_or(0),
    )
}
pub(super) fn aggregate(cells: &[Value]) -> Vec<Value> {
    let mut buckets = BTreeMap::<(String, String, String, u64), Vec<&Value>>::new();
    for cell in cells {
        for observation in cell["observations"].as_array().into_iter().flatten() {
            buckets
                .entry((
                    cell["version"].as_str().unwrap_or("").into(),
                    cell["cache"].as_str().unwrap_or("").into(),
                    observation["scenario"].as_str().unwrap_or("").into(),
                    observation["concurrency"].as_u64().unwrap_or(0),
                ))
                .or_default()
                .push(&observation["summary"]);
        }
    }
    let mut result = vec![];
    for ((version, cache, scenario, n), summaries) in buckets {
        let pool = |field: &str| {
            summaries
                .iter()
                .flat_map(|s| s[field].as_array().into_iter().flatten())
                .filter_map(Value::as_f64)
                .collect::<Vec<_>>()
        };
        let sum = |field: &str| {
            summaries
                .iter()
                .filter_map(|s| s[field].as_u64())
                .sum::<u64>()
        };
        let mut outputs = BTreeMap::<String, BTreeSet<String>>::new();
        for s in &summaries {
            if let Some(map) = s["outputs_by_prompt"].as_object() {
                for (k, values) in map {
                    for output in values
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                    {
                        outputs.entry(k.clone()).or_default().insert(output.into());
                    }
                }
            }
        }
        result.push(json!({"version":version,"cache":cache,"scenario":scenario,"concurrency":n,"requests":sum("requests"),"successful":sum("successful"),"cache_hits":sum("cache_hits"),"cache_misses":sum("cache_misses"),"matched_prefix_tokens_median":median(&pool("matched_prefix_tokens")),"suffix_prefill_tokens_median":median(&pool("suffix_prefill_tokens")),"ttft_ms_p50":quantile(&pool("ttft_ms"),0.5),"ttft_ms_p99":quantile(&pool("ttft_ms"),0.99),"tpot_ms_p50":quantile(&pool("tpot_ms"),0.5),"outputs_by_prompt":outputs}));
    }
    let indexed = result
        .iter()
        .map(|r| (key(r), r["ttft_ms_p50"].as_f64()))
        .collect::<BTreeMap<_, _>>();
    for row in &mut result {
        let (v, c, s, n) = key(row);
        if c == "warm"
            && let (Some(Some(cold)), Some(warm)) = (
                indexed.get(&(v, "cold".into(), s, n)),
                row["ttft_ms_p50"].as_f64(),
            )
        {
            row["cache_lift_ttft_ms"] = json!(cold - warm);
        }
    }
    result
}
pub(super) fn comparisons(rows: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let indexed = rows.iter().map(|r| (key(r), r)).collect::<BTreeMap<_, _>>();
    let mut parity = vec![];
    let mut preservation = vec![];
    for row in rows {
        let (v, c, s, n) = key(row);
        if v == "old"
            && let Some(new) = indexed.get(&("new".into(), c.clone(), s.clone(), n))
        {
            parity.push(json!({"cache":c,"scenario":s,"concurrency":n,"identical_outputs":row["outputs_by_prompt"]==new["outputs_by_prompt"]}));
        }
        if c == "cold"
            && let Some(warm) = indexed.get(&(v.clone(), "warm".into(), s.clone(), n))
        {
            preservation.push(json!({"version":v,"scenario":s,"concurrency":n,"cache_preserves_output":row["outputs_by_prompt"]==warm["outputs_by_prompt"]}));
        }
    }
    (parity, preservation)
}
