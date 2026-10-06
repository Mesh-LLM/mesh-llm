//! Mixed role metrics and finite paired reports; unavailable telemetry stays unavailable.
use super::mixed_counters::Measured;
use crate::command::DynResult;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
pub(super) const METRICS: [&str; 16] = [
    "makespan_ms",
    "output_tokens_per_second",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "anchor_ttft_ms_p50",
    "anchor_ttft_ms_p95",
    "anchor_gap_ms_p50",
    "anchor_gap_ms_p95",
    "prefill_ttft_ms_p50",
    "prefill_ttft_ms_p95",
    "scheduler_iterations",
    "mixed_iterations",
    "mean_batch_tokens",
    "mean_token_occupancy",
    "prefill_chunk_count_median",
    "prefill_max_chunk_size_median",
];
fn number(value: &Value) -> DynResult<f64> {
    let n = value
        .as_f64()
        .ok_or("mixed required numeric evidence absent")?;
    if !n.is_finite() || n < 0.0 {
        return Err("mixed numeric evidence invalid".into());
    }
    Ok(n)
}
fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[((sorted.len() - 1) as f64 * fraction).round_ties_even() as usize])
}
fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    Some(if n.is_multiple_of(2) {
        values[n / 2 - 1] / 2.0 + values[n / 2] / 2.0
    } else {
        values[n / 2]
    })
}
fn delta(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    let (b, a) = (before?, after?);
    if b == 0.0 {
        return None;
    }
    let d = (a - b) / b * 100.0;
    d.is_finite().then_some(d)
}
pub(super) fn requests(worker: &Value) -> DynResult<Value> {
    let rows = worker["requests"]
        .as_array()
        .ok_or("mixed request rows absent")?;
    let wall = number(&worker["makespan_ms"])?;
    if wall == 0.0 {
        return Err("mixed makespan must be positive".into());
    }
    let mut ids = BTreeSet::new();
    let mut ttft = Vec::new();
    let mut anchor = Vec::new();
    let mut prefill = Vec::new();
    let mut gaps = Vec::new();
    let mut tokens = 0_u64;
    let mut errors = 0;
    for row in rows {
        let id = row["request_index"]
            .as_u64()
            .ok_or("mixed request index absent")?;
        if !ids.insert(id) {
            return Err("mixed request index duplicated".into());
        }
        let role = row["role"].as_str().ok_or("mixed request role absent")?;
        if !matches!(role, "anchor" | "prefill") {
            return Err("mixed request role invalid".into());
        }
        if row["error"].is_string() {
            errors += 1;
            continue;
        }
        if !row["error"].is_null() {
            return Err("mixed request error shape invalid".into());
        }
        let time = number(&row["ttft_ms"])?;
        ttft.push(time);
        tokens = tokens
            .checked_add(
                row["completion_tokens"]
                    .as_u64()
                    .ok_or("mixed completion usage absent")?,
            )
            .ok_or("mixed completion usage overflow")?;
        if role == "anchor" {
            anchor.push(time);
            for gap in row["content_gaps_ms"]
                .as_array()
                .ok_or("mixed anchor gap roster absent")?
            {
                gaps.push(number(gap)?);
            }
        } else {
            prefill.push(time);
        }
    }
    let throughput = tokens as f64 / (wall / 1000.0);
    if !throughput.is_finite() {
        return Err("mixed throughput overflow".into());
    }
    Ok(
        json!({"requests":rows.len(),"successful_requests":rows.len()-errors,"errors":errors,"completion_tokens":tokens,"makespan_ms":wall,"output_tokens_per_second":throughput,"ttft_ms_p50":percentile(&ttft,0.5),"ttft_ms_p95":percentile(&ttft,0.95),"anchor_ttft_ms_p50":percentile(&anchor,0.5),"anchor_ttft_ms_p95":percentile(&anchor,0.95),"anchor_gap_ms_p50":percentile(&gaps,0.5),"anchor_gap_ms_p95":percentile(&gaps,0.95),"prefill_ttft_ms_p50":percentile(&prefill,0.5),"prefill_ttft_ms_p95":percentile(&prefill,0.95),"scheduler_available":false,"scheduler_breakdown_available":false,"scheduler_iterations":null,"mixed_iterations":null,"mean_batch_tokens":null,"mean_token_occupancy":null,"prefill_chunk_count_median":null,"prefill_max_chunk_size_median":null,"prefill_bottleneck_stage_median":null}),
    )
}
pub(super) fn counters(
    summary: &mut Value,
    worker: &Value,
    measured: &Measured,
    n_batch: u32,
    prefills: usize,
) -> DynResult<()> {
    if n_batch == 0
        || prefills == 0
        || measured.scheduler.is_empty()
        || measured.request_sha256
            != worker["input_sha256"]
                .as_str()
                .ok_or("mixed worker input identity absent")?
        || !worker["error"].is_null()
        || summary["errors"] != 0
    {
        return Err("mixed measured counter input/worker status refusal".into());
    }
    let mut tokens = 0.0;
    let mut detailed = 0;
    let mut mixed = 0;
    let mut prefill_only = 0;
    let mut decode_only = 0;
    for (index, row) in measured.scheduler.iter().enumerate() {
        let total = row.tokens()? as f64;
        tokens += (total - tokens) / (index + 1) as f64;
        if let Some((p, r, d)) = row.detailed() {
            detailed += 1;
            let pr = p.checked_add(r).ok_or("mixed prefill total overflow")?;
            mixed += usize::from(d > 0 && pr > 0);
            prefill_only += usize::from(d == 0 && pr > 0);
            decode_only += usize::from(d > 0 && pr == 0);
        }
    }
    if !tokens.is_finite() {
        return Err("mixed mean tokens overflow".into());
    }
    let mut ranked = measured.prefills.clone();
    ranked.sort_by_key(|p| std::cmp::Reverse(p.token_count));
    if ranked.len() < prefills {
        return Err("mixed prefill roster too short".into());
    }
    ranked.truncate(prefills);
    for (key, value) in [
        ("scheduler_available", json!(true)),
        (
            "scheduler_phase_qualified",
            json!(matches!(
                measured.phase_provenance,
                "owner-snapshotted-verified-warmup-barrier-required"
                    | "owned-local-producer-timestamps-with-complete-shutdown-tail"
            )),
        ),
        ("scheduler_breakdown_available", json!(detailed > 0)),
        ("scheduler_iterations", json!(measured.scheduler.len())),
        (
            "mixed_iterations",
            json!(if detailed > 0 { Some(mixed) } else { None }),
        ),
        (
            "prefill_only_iterations",
            json!(if detailed > 0 {
                Some(prefill_only)
            } else {
                None
            }),
        ),
        (
            "decode_only_iterations",
            json!(if detailed > 0 {
                Some(decode_only)
            } else {
                None
            }),
        ),
        ("mean_batch_tokens", json!(tokens)),
        ("mean_token_occupancy", json!(tokens / f64::from(n_batch))),
        (
            "prefill_chunk_count_median",
            json!(median(ranked.iter().map(|p| p.chunks as f64).collect())),
        ),
        (
            "prefill_max_chunk_size_median",
            json!(median(ranked.iter().map(|p| p.maximum as f64).collect())),
        ),
        (
            "prefill_bottleneck_stage_median",
            json!(median(
                ranked
                    .iter()
                    .filter_map(|p| p.bottleneck_stage.map(|s| s as f64))
                    .collect()
            )),
        ),
        (
            "scheduler_phase_provenance",
            json!(measured.phase_provenance),
        ),
        (
            "prefill_role_provenance",
            json!(measured.prefill_role_provenance),
        ),
    ] {
        summary[key] = value;
    }
    Ok(())
}
fn find<'a>(cells: &'a [Value], round: u64, version: &str) -> DynResult<&'a Value> {
    let mut matching = cells
        .iter()
        .filter(|c| c["round"] == round && c["version"] == version);
    let row = matching.next().ok_or("mixed round pair missing")?;
    if matching.next().is_some() {
        return Err("mixed round pair duplicated".into());
    }
    Ok(row)
}
fn aggregate(cells: &[Value], version: &str) -> Value {
    let keys = METRICS
        .into_iter()
        .chain(["successful_requests", "errors", "completion_tokens"]);
    json!(
        keys.map(|key| {
            let values = cells
                .iter()
                .filter(|c| c["version"] == version)
                .map(|c| c["summary"][key].as_f64())
                .collect::<Option<Vec<_>>>();
            (key, values.and_then(median))
        })
        .collect::<BTreeMap<_, _>>()
    )
}
pub(super) fn compare(cells: &[Value], rounds: u64) -> DynResult<Value> {
    if !(1..=128).contains(&rounds) || cells.len() != rounds as usize * 2 {
        return Err("mixed exact bounded paired roster required".into());
    }
    let mut mismatches = Vec::new();
    let mut comparable = 0;
    let mut qualified = true;
    for round in 1..=rounds {
        let old = find(cells, round, "old")?;
        let new = find(cells, round, "new")?;
        let before = old["requests"]
            .as_array()
            .ok_or("old mixed request roster absent")?;
        let after = new["requests"]
            .as_array()
            .ok_or("new mixed request roster absent")?;
        if before.is_empty() || before.len() != after.len() {
            return Err("mixed request pair incomplete".into());
        }
        for (a, b) in before.iter().zip(after) {
            if a["request_index"] != b["request_index"]
                || a["role"] != b["role"]
                || a["prompt_sha256"] != b["prompt_sha256"]
                || a["prompt_provenance"] != b["prompt_provenance"]
            {
                return Err("mixed request pair identity differs".into());
            }
            if a["error"].is_string() || b["error"].is_string() {
                qualified = false;
                continue;
            }
            comparable += 1;
            if a["content_sha256"] != b["content_sha256"] {
                mismatches.push(json!({"round":round,"request":a["request_index"]}));
            }
        }
        qualified &= old["summary"]["scheduler_available"] == true
            && new["summary"]["scheduler_available"] == true
            && old["summary"]["scheduler_phase_qualified"] == true
            && new["summary"]["scheduler_phase_qualified"] == true;
    }
    let old = aggregate(cells, "old");
    let new = aggregate(cells, "new");
    let mut paired = BTreeMap::new();
    let mut deltas = BTreeMap::new();
    for key in METRICS {
        deltas.insert(key, delta(old[key].as_f64(), new[key].as_f64()));
        let mut values = Vec::new();
        for round in 1..=rounds {
            if let Some(d) = delta(
                find(cells, round, "old")?["summary"][key].as_f64(),
                find(cells, round, "new")?["summary"][key].as_f64(),
            ) {
                values.push(d);
            }
        }
        if !values.is_empty() {
            let ci = crate::automation::event_benchmark_comparison::resampling::median_interval(
                &values,
                10000,
                0,
                "mixed-prefill-decode",
                key,
            )?;
            paired.insert(
                key,
                json!({"round_deltas":values,"median":ci[0],"ci95":[ci[1],ci[2]]}),
            );
        }
    }
    Ok(
        json!({"qualified":qualified&&mismatches.is_empty(),"aggregate":{"old":old,"new":new,"delta_percent":deltas},"paired_delta_percent":paired,"output_parity":{"comparable_requests":comparable,"exact_matches":comparable-mismatches.len(),"mismatches":mismatches},"bootstrap_algorithm":"sha256-counter-paired-median-bootstrap-v1","bootstrap_resamples":10000}),
    )
}
pub(super) fn render(value: &Value) -> String {
    let mut out = String::from(
        "Mixed scheduling\n\n| Metric | Before | After | Delta | Paired 95% CI |\n| --- | ---: | ---: | ---: | ---: |\n",
    );
    for (label, key) in [
        ("Makespan ms", "makespan_ms"),
        ("Output tok/s", "output_tokens_per_second"),
        ("Anchor TTFT p50 ms", "anchor_ttft_ms_p50"),
        ("Anchor TTFT p95 ms", "anchor_ttft_ms_p95"),
        ("Anchor stream gap p50 ms", "anchor_gap_ms_p50"),
        ("Anchor stream gap p95 ms", "anchor_gap_ms_p95"),
        ("Prefill TTFT p50 ms", "prefill_ttft_ms_p50"),
        ("Prefill TTFT p95 ms", "prefill_ttft_ms_p95"),
        ("Scheduler iterations", "scheduler_iterations"),
        ("Mixed iterations", "mixed_iterations"),
        ("Mean batch tokens", "mean_batch_tokens"),
        ("Token-budget occupancy", "mean_token_occupancy"),
        ("Prefill chunks / request", "prefill_chunk_count_median"),
        ("Maximum prefill chunk", "prefill_max_chunk_size_median"),
    ] {
        let (Some(a), Some(b)) = (
            value["aggregate"]["old"][key].as_f64(),
            value["aggregate"]["new"][key].as_f64(),
        ) else {
            continue;
        };
        let change = delta(Some(a), Some(b)).map_or_else(|| "n/a".into(), |v| format!("{v:+.1}%"));
        let ci = &value["paired_delta_percent"][key]["ci95"];
        let interval = match (ci[0].as_f64(), ci[1].as_f64()) {
            (Some(a), Some(b)) => format!("[{a:+.1}%, {b:+.1}%]"),
            _ => "n/a".into(),
        };
        out.push_str(&format!(
            "| {label} | {a:.3} | {b:.3} | {change} | {interval} |\n"
        ));
    }
    out.push_str(&format!(
        "\nOutput parity: {}/{} exact typed completion hashes.\n",
        value["output_parity"]["exact_matches"], value["output_parity"]["comparable_requests"]
    ));
    out
}
