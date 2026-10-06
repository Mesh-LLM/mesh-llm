//! Cohort statistics preserve unavailable/singleton percentiles and canonical baseline evidence.
use serde_json::{Value, json};
fn quantile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index])
}
fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut value = 0.0;
    for (index, next) in values.iter().enumerate() {
        value += (next - value) / (index + 1) as f64;
    }
    value.is_finite().then_some(value)
}
pub(super) fn cohort(name: &str, rows: &[Value]) -> Value {
    let selected = rows
        .iter()
        .filter(|r| r["cohort"] == name)
        .collect::<Vec<_>>();
    let successful = selected
        .iter()
        .filter(|r| r["error"].is_null())
        .collect::<Vec<_>>();
    let values = |key: &str| {
        successful
            .iter()
            .filter_map(|r| r[key].as_f64().filter(|v| v.is_finite() && *v >= 0.0))
            .collect::<Vec<_>>()
    };
    let ttft = values("ttft_seconds");
    let sum = |key: &str| -> Option<u64> {
        if successful.is_empty() {
            return None;
        }
        successful
            .iter()
            .try_fold(0_u64, |total, row| total.checked_add(row[key].as_u64()?))
    };
    let prompt = sum("prompt_tokens");
    let cached = sum("cached_tokens");
    let cache_pct = match (prompt, cached) {
        (Some(p), Some(c)) if p > 0 => Some(c as f64 / p as f64 * 100.0),
        _ => None,
    };
    json!({"cohort":name,"requests":selected.len(),"failed":selected.len()-successful.len(),"ttft_p50_seconds":quantile(&ttft,0.5),"ttft_p95_seconds":if ttft.len()>1{quantile(&ttft,0.95)}else{None},"total_seconds_mean":mean(&values("total_seconds")),"prompt_tokens":prompt,"cached_tokens":cached,"cache_pct":cache_pct,"decode_tokens_per_second_mean":mean(&values("decode_tokens_per_second"))})
}
fn shown(value: &Value) -> String {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "—".into())
}
pub(super) fn render(run: &Value) -> String {
    let mut text = format!(
        "# KV restart replay\n\nSource: `{}` (checkout provenance, not binary build attestation)\n\nModel SHA: `{}`\n\nManifest SHA: `{}`\n\n| cohort | requests | failed | TTFT p50 (s) | TTFT p95 (s) | cached % | decode tok/s |\n|---|---:|---:|---:|---:|---:|---:|\n",
        run["binary"]["source_sha"].as_str().unwrap_or("unknown"),
        run["model"]["sha256"].as_str().unwrap_or("unavailable"),
        run["manifest_sha256"].as_str().unwrap_or("unavailable")
    );
    if let Some(rows) = run["cohorts"].as_array() {
        for row in rows {
            text.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                row["cohort"].as_str().unwrap_or("unknown"),
                row["requests"],
                row["failed"],
                shown(&row["ttft_p50_seconds"]),
                shown(&row["ttft_p95_seconds"]),
                shown(&row["cache_pct"]),
                shown(&row["decode_tokens_per_second_mean"])
            ));
        }
    }
    text.push_str("\n`restore` is exactly the first replay after a clean process restart on the same state. Without durable KV it is a cold-prefill reference; warm replays measure resident reuse. Canonical completion hashes are compared with the last fill baseline. No speedup or durable restoration is inferred from a fixture.\n");
    if !run["error"].is_null() {
        text.push_str(&format!("\nRun failed: {}\n", run["error"]));
    }
    text
}
