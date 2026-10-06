//! Adaptive client/prefill summaries, paired round medians and exact typed-output parity.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::collections::BTreeMap;
pub(super) const METRICS: [&str; 7] = [
    "ttft_ms_p50",
    "ttft_ms_p95",
    "prefill_elapsed_ms_p95",
    "makespan_ms",
    "output_tokens_per_second",
    "prefill_chunk_count_median",
    "prefill_max_chunk_size_median",
];
fn number(value: &Value) -> DynResult<f64> {
    let n = value
        .as_f64()
        .ok_or("required adaptive numeric evidence absent")?;
    if !n.is_finite() || n < 0.0 {
        return Err("adaptive measurement invalid".into());
    }
    Ok(n)
}
fn percentile(mut values: Vec<f64>, q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[((values.len() - 1) as f64 * q).round_ties_even() as usize])
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
pub(super) fn summarize(cell: &Value) -> DynResult<Value> {
    let requests = cell["requests"]["requests"]
        .as_array()
        .ok_or("adaptive requests absent")?;
    let prefill = cell["measured_prefill"]
        .as_array()
        .ok_or("required adaptive prefill absent")?;
    if requests.is_empty()
        || requests.len() != prefill.len()
        || cell["prefill_telemetry_available"] != true
    {
        return Err("adaptive measured roster incomplete".into());
    }
    let mut ttft = Vec::new();
    let mut tokens = 0_u64;
    for (index, row) in requests.iter().enumerate() {
        if row["request_id"].as_u64() != Some(index as u64) || !row["error"].is_null() {
            return Err("adaptive request failed or reordered".into());
        }
        ttft.push(number(&row["ttft_ms"])?);
        tokens = tokens
            .checked_add(
                row["tokens_predicted"]
                    .as_u64()
                    .ok_or("completion usage absent")?,
            )
            .ok_or("completion usage overflow")?;
    }
    let wall = number(&cell["requests"]["makespan_ms"])?;
    if wall == 0.0 {
        return Err("adaptive makespan must be positive".into());
    }
    let throughput = tokens as f64 / (wall / 1000.0);
    if !throughput.is_finite() {
        return Err("adaptive throughput overflow".into());
    }
    let field = |key: &str| -> DynResult<Vec<f64>> {
        prefill.iter().map(|row| number(&row[key])).collect()
    };
    Ok(
        json!({"requests":requests.len(),"successful_requests":requests.len(),"errors":0,"ttft_ms_p50":percentile(ttft.clone(),0.5),"ttft_ms_p95":percentile(ttft,0.95),"makespan_ms":wall,"output_tokens_per_second":throughput,"prefill_chunk_count_median":median(field("chunks")?),"prefill_min_chunk_size_median":median(field("minimum")?),"prefill_max_chunk_size_median":median(field("maximum")?),"prefill_elapsed_ms_p95":percentile(field("elapsed_ms")?,0.95)}),
    )
}
fn find<'a>(cells: &'a [Value], round: u64, version: &str) -> DynResult<&'a Value> {
    let mut matches = cells
        .iter()
        .filter(|c| c["round"] == round && c["version"] == version);
    let cell = matches.next().ok_or("adaptive pair missing")?;
    if matches.next().is_some() {
        return Err("adaptive pair duplicated".into());
    }
    Ok(cell)
}
fn aggregate(cells: &[Value], version: &str) -> Value {
    let keys = METRICS.into_iter().chain([
        "successful_requests",
        "errors",
        "prefill_min_chunk_size_median",
    ]);
    let summary: BTreeMap<_, _> = keys
        .map(|key| {
            let rows = cells
                .iter()
                .filter(|c| c["version"] == version)
                .filter_map(|c| c["summary"][key].as_f64())
                .collect();
            (key, median(rows))
        })
        .collect();
    json!(summary)
}
pub(super) fn compare(cells: &[Value], rounds: u64) -> DynResult<Value> {
    if rounds == 0 || rounds > 128 || cells.len() != rounds as usize * 2 {
        return Err("adaptive comparison requires exact bounded round roster".into());
    }
    let mut mismatches = Vec::new();
    let mut comparable = 0;
    for round in 1..=rounds {
        let old = find(cells, round, "old")?;
        let new = find(cells, round, "new")?;
        let before = old["requests"]["requests"]
            .as_array()
            .ok_or("old request roster absent")?;
        let after = new["requests"]["requests"]
            .as_array()
            .ok_or("new request roster absent")?;
        if before.len() != after.len() || before.is_empty() {
            return Err("adaptive request pair incomplete".into());
        }
        for (index, (a, b)) in before.iter().zip(after).enumerate() {
            if a["request_id"] != b["request_id"]
                || a["prompt_sha256"] != b["prompt_sha256"]
                || a["prompt_provenance"] != b["prompt_provenance"]
                || a["error"].is_string()
                || b["error"].is_string()
            {
                return Err("adaptive request pairing or status invalid".into());
            }
            for row in [a, b] {
                let hash = row["content_sha256"]
                    .as_str()
                    .ok_or("output identity absent")?;
                if hash.len() != 64 || !hash.bytes().all(|x| x.is_ascii_hexdigit()) {
                    return Err("output identity invalid".into());
                }
            }
            comparable += 1;
            if a["content_sha256"] != b["content_sha256"] {
                mismatches.push(json!({"round":round,"request":index}));
            }
        }
    }
    let old = aggregate(cells, "old");
    let new = aggregate(cells, "new");
    let mut paired = BTreeMap::new();
    let mut deltas = BTreeMap::new();
    for metric in METRICS {
        deltas.insert(metric, delta(old[metric].as_f64(), new[metric].as_f64()));
        let mut values = Vec::new();
        for round in 1..=rounds {
            if let Some(d) = delta(
                find(cells, round, "old")?["summary"][metric].as_f64(),
                find(cells, round, "new")?["summary"][metric].as_f64(),
            ) {
                values.push(d);
            }
        }
        if !values.is_empty() {
            let interval =
                crate::automation::event_benchmark_comparison::resampling::median_interval(
                    &values,
                    10_000,
                    0,
                    "adaptive-prefill",
                    metric,
                )?;
            paired.insert(metric,json!({"round_deltas":values,"median":interval[0],"ci95":[interval[1],interval[2]]}));
        }
    }
    Ok(
        json!({"aggregate":{"old":old,"new":new,"delta_percent":deltas},"paired_delta_percent":paired,"bootstrap_algorithm":"sha256-counter-paired-median-bootstrap-v1","bootstrap_resamples":10000,"output_parity":{"comparable_requests":comparable,"exact_matches":comparable-mismatches.len(),"mismatches":mismatches}}),
    )
}
pub(super) fn render(comparison: &Value) -> String {
    let mut out = String::from(
        "| Metric | Before | After | Delta | Paired 95% CI |\n| --- | ---: | ---: | ---: | ---: |\n",
    );
    let fmt = |value: Option<f64>| value.map_or_else(|| "n/a".into(), |v| format!("{v:.1}"));
    for key in METRICS {
        let ci = &comparison["paired_delta_percent"][key]["ci95"];
        let interval = match (ci[0].as_f64(), ci[1].as_f64()) {
            (Some(a), Some(b)) => format!("[{a:+.1}%, {b:+.1}%]"),
            _ => "n/a".into(),
        };
        out.push_str(&format!(
            "| {key} | {} | {} | {} | {interval} |\n",
            fmt(comparison["aggregate"]["old"][key].as_f64()),
            fmt(comparison["aggregate"]["new"][key].as_f64()),
            comparison["aggregate"]["delta_percent"][key]
                .as_f64()
                .map_or_else(|| "n/a".into(), |v| format!("{v:+.1}%"))
        ));
    }
    out.push_str(&format!(
        "\nOutput parity: {}/{} exact typed completion hashes.\n",
        comparison["output_parity"]["exact_matches"],
        comparison["output_parity"]["comparable_requests"]
    ));
    out
}
