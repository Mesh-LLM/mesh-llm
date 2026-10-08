//! Client-observed cache goodput and same-protocol typed-output parity.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::collections::BTreeMap;
fn finite(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v| v.is_finite() && *v >= 0.0)
}
fn quantile(mut values: Vec<f64>, percentile: usize) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[(values.len() * percentile / 100).min(values.len() - 1)])
}
fn rows(measurement: &Value) -> DynResult<BTreeMap<u64, &Value>> {
    let list = measurement["rows"]
        .as_array()
        .ok_or("cache measurement roster absent")?;
    let count = measurement["requested_requests"]
        .as_u64()
        .ok_or("cache requested count absent")?;
    if count == 0 || count > 4096 || list.len() as u64 != count {
        return Err("cache measurement roster incomplete".into());
    }
    let mut rows = BTreeMap::new();
    for row in list {
        let id = row["request_id"]
            .as_u64()
            .ok_or("cache request ID absent")?;
        if id >= count || rows.insert(id, row).is_some() {
            return Err("cache request IDs duplicated/outside roster".into());
        }
    }
    Ok(rows)
}
pub(super) fn goodput(measurement: &Value, ttft_slo: u32, tpot_slo: u32) -> DynResult<Value> {
    let rows = rows(measurement)?;
    let seconds = finite(&measurement["makespan_seconds"])
        .filter(|s| *s > 0.0)
        .ok_or("cache makespan absent/nonpositive")?;
    let successful: Vec<_> = rows
        .values()
        .filter(|row| row["status"] == "completed" && row["excluded_warmup"] == false)
        .copied()
        .collect();
    let mut tokens = 0_u64;
    let mut ttfts = Vec::new();
    let mut tpots = Vec::new();
    let mut elapsed = Vec::new();
    let mut good = 0_u32;
    for row in &successful {
        tokens = tokens
            .checked_add(
                row["tokens_predicted"]
                    .as_u64()
                    .ok_or("cache observed tokens absent")?,
            )
            .ok_or("cache output count overflow")?;
        elapsed.push(finite(&row["elapsed_ms"]).ok_or("cache latency absent/nonfinite")?);
        let ttft = finite(&row["ttft_ms"]);
        let tpot = finite(&row["tpot_ms"]);
        if let Some(v) = ttft {
            ttfts.push(v);
        }
        if let Some(v) = tpot {
            tpots.push(v);
        }
        if ttft.is_some_and(|v| v <= f64::from(ttft_slo))
            && tpot.is_some_and(|v| v <= f64::from(tpot_slo))
        {
            good += 1;
        }
    }
    let count = u32::try_from(successful.len())?;
    let throughput = f64::from(count) / seconds;
    let output_rate = crate::automation::openai_exchange::stream::number(tokens) / seconds;
    let goodput = f64::from(good) / seconds;
    if ![throughput, output_rate, goodput]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err("cache metric rate overflow".into());
    }
    Ok(
        json!({"scope":"client_observed_not_runtime_scheduler","throughput_rps":throughput,"goodput_rps":goodput,"output_tokens_per_second":output_rate,"successful_requests":count,"failed_or_unlaunched_requests":rows.len()-successful.len(),"observed_output_tokens":tokens,"ttft_p50_ms":quantile(ttfts.clone(),50),"ttft_p99_ms":quantile(ttfts,99),"tpot_p50_ms":quantile(tpots.clone(),50),"tpot_p99_ms":quantile(tpots,99),"latency_p50_ms":quantile(elapsed.clone(),50),"latency_p99_ms":quantile(elapsed,99),"complete":measurement["status"]=="completed"}),
    )
}
fn digest(value: &Value) -> Option<&str> {
    value.as_str().filter(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}
pub(super) fn parity(old: &Value, new: &Value) -> DynResult<Value> {
    let left = rows(old)?;
    let right = rows(new)?;
    if old["cohort"] != "openai-concurrent"
        || new["cohort"] != "openai-concurrent"
        || old["concurrency"] != new["concurrency"]
        || old["prompt_sha256"] != new["prompt_sha256"]
        || old["output_tokens"] != new["output_tokens"]
        || left.keys().ne(right.keys())
    {
        return Err("cache pair workload/roster mismatch".into());
    }
    let mut mismatches = Vec::new();
    let mut first_mismatches = Vec::new();
    let mut unavailable = Vec::new();
    for (id, l) in &left {
        let r = right[id];
        let hashes = [
            digest(&l["evidence"]["content_sha256"]),
            digest(&r["evidence"]["content_sha256"]),
            digest(&l["evidence"]["first_generated_sha256"]),
            digest(&r["evidence"]["first_generated_sha256"]),
        ];
        if l["status"] != "completed"
            || r["status"] != "completed"
            || hashes.iter().any(Option::is_none)
        {
            unavailable.push(*id);
            continue;
        }
        if hashes[0] != hashes[1] {
            mismatches.push(*id);
        }
        if hashes[2] != hashes[3] {
            first_mismatches.push(*id);
        }
    }
    let complete =
        old["status"] == "completed" && new["status"] == "completed" && unavailable.is_empty();
    Ok(
        json!({"scope":"same_typed_OpenAI_delta_protocol_not_cross_native_chunk_parity","complete":complete,"matches":complete&&mismatches.is_empty()&&first_mismatches.is_empty(),"comparable_requests":left.len()-unavailable.len(),"mismatch_request_ids":mismatches,"first_generated_mismatch_request_ids":first_mismatches,"unavailable_request_ids":unavailable}),
    )
}
/// Accepted parity requires the owning cells' full process/capture/identity gates,
/// not merely successful HTTP measurements inside a subsequently refused cell.
pub(super) fn accepted_parity(
    old: &Value,
    new: &Value,
    old_accepted: bool,
    new_accepted: bool,
) -> DynResult<Value> {
    if !old_accepted || !new_accepted {
        return Err("cache parity owning cell refused".into());
    }
    parity(old, new)
}
