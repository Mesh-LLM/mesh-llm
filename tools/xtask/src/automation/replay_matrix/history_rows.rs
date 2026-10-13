use super::{
    history_artifacts::{Cell, Family},
    history_input::{Input, Model},
};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn build(
    input: &Input,
    model: (&Model, &Value),
    family: &Family,
) -> DynResult<Vec<Value>> {
    let mut groups = BTreeMap::<usize, Vec<&Cell>>::new();
    for cell in &family.cells {
        let level = usize::try_from(
            cell.value["concurrency"]
                .as_u64()
                .ok_or("missing concurrency")?,
        )?;
        groups.entry(level).or_default().push(cell);
    }
    groups
        .into_iter()
        .map(|(level, cells)| row(input, model, family, level, &cells))
        .collect()
}
fn row(
    input: &Input,
    model: (&Model, &Value),
    family: &Family,
    level: usize,
    cells: &[&Cell],
) -> DynResult<Value> {
    let metrics = cells
        .iter()
        .map(|cell| serde_json::from_value::<super::pooled_metrics::Cell>(cell.value.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let pooled = super::pooled_metrics::aggregate(&metrics.iter().collect::<Vec<_>>())?;
    let requests = count(cells, "requests")?;
    let successful = count(cells, "successful_requests")?;
    let failed = count(cells, "failed_requests")?;
    let lengths = cells.iter().try_fold(0_u64, |total, cell| {
        let count = cell.value["finish_reason_length_requests"]
            .as_u64()
            .or_else(|| cell.value["budget_exhausted_requests"].as_u64())
            .unwrap_or(0);
        total
            .checked_add(count)
            .ok_or("history length count overflow")
    })?;
    let output = sum(cells, "completion_tokens")?;
    let output_count = raw_output_tokens(cells, level)?;
    let window = sum(cells, "workload_window_seconds")?;
    let mut ttft = cells
        .iter()
        .flat_map(|cell| cell.value["ttft_samples"].as_array().into_iter().flatten())
        .map(|value| {
            let value = value.as_f64().ok_or("invalid history TTFT")? * 1000.0;
            if !value.is_finite() || value < 0.0 {
                return Err("invalid history TTFT");
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    ttft.sort_by(f64::total_cmp);
    let mut replay = input.replay_value.clone();
    replay["concurrency"] = level.into();
    let cohort = json!({"model":model.0.family,"quant":model.0.quant,"concurrency":level,
        "backend":"mesh","runner":input.hardware["machine_model"]});
    let cohort_key =
        super::cohort_identity::digest(&json!({"cohort":cohort,"source":input.source_sha,
        "replay":replay,"model_revision":model.0.revision,"model_sha":model.0.sha256}))?;
    let expected: BTreeSet<_> = (1..=input.replay.passes).collect();
    let complete = cells.iter().all(|cell| cell.verified)
        && cells.len() == expected.len()
        && cells.iter().map(|cell| cell.pass).collect::<BTreeSet<_>>() == expected
        && requests > 0
        && failed == 0
        && successful.checked_add(failed) == Some(requests);
    let newest = cells
        .iter()
        .map(|cell| cell.modified)
        .max()
        .ok_or("history group has no cells")?;
    let instant = crate::ci_operations::ci_metrics_time::Instant::from_system_time(newest)
        .ok_or("history cell timestamp is outside supported range")?;
    let date = crate::ci_operations::ci_metrics_time::isoformat(instant);
    let created = format!("{}Z", date.get(..19).ok_or("invalid history timestamp")?);
    let recurrent_restores = if model.0.class == "hybrid-recurrent" {
        Some(cells.iter().try_fold(0_u64, |total, cell| {
            total
                .checked_add(
                    cell.value["recurrent_state"]["restores"]
                        .as_u64()
                        .unwrap_or(0),
                )
                .ok_or("history restore count overflow")
        })?)
    } else {
        None
    };
    let identities = cells
        .iter()
        .filter_map(|cell| cell.value["session_cohort_sha256"].as_str())
        .collect::<BTreeSet<_>>();
    Ok(
        json!({"schema_version":3,"created_utc":created,"source_sha":input.source_sha,"cohort_key":cohort_key,
        "cohort":cohort,"backend_binary_sha256":if family.backend.is_empty() {None} else {Some(&family.backend)},
        "hardware_fingerprint":input.hardware,"model":model.1,"replay":replay,"prompt_count":requests,
        "successful_requests":successful,"failed_requests":failed,"output_tokens":output_count,
        "measured_wall_ms":1000.0*window,"decode_tokens_per_second":pooled["decode_tokens_per_second"].as_f64().unwrap_or(0.0),
        "end_to_end_tokens_per_second":if window>0.0 {output/window} else {0.0},
        "ttft_ms_mean":mean(&ttft).unwrap_or(0.0),
        "ttft_ms_p90":if ttft.is_empty() {0.0} else {ttft[(ttft.len()-1)*9/10]},
        "cache_hit_pct":samples_mean(cells,"cache_pct")?,
        "finish_reason_length_pct":if successful==0 {None} else {Some(100.0*lengths as f64/successful as f64)},
        "complete":complete,"artifact_result":if complete {"ok"} else {"incomplete"},
        "session_cohort_sha256":identities,"recurrent_restores":recurrent_restores}),
    )
}
fn count(cells: &[&Cell], key: &str) -> DynResult<u64> {
    Ok(cells.iter().try_fold(0_u64, |total, cell| {
        total
            .checked_add(cell.value[key].as_u64().ok_or("missing history count")?)
            .ok_or("history count overflow")
    })?)
}
fn sum(cells: &[&Cell], key: &str) -> DynResult<f64> {
    let mut total = 0.0;
    for cell in cells {
        let value = cell.value[key].as_f64().ok_or("missing history metric")?;
        if !value.is_finite() || value < 0.0 {
            return Err("invalid history metric".into());
        }
        total += value;
    }
    if !total.is_finite() {
        return Err("history metric overflow".into());
    }
    Ok(total)
}
fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}
fn samples_mean(cells: &[&Cell], key: &str) -> DynResult<Option<f64>> {
    let samples = cells
        .iter()
        .filter_map(|cell| cell.value[key].as_f64())
        .collect::<Vec<_>>();
    if samples
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("invalid history sample".into());
    }
    Ok(mean(&samples))
}

fn raw_output_tokens(cells: &[&Cell], level: usize) -> DynResult<u64> {
    let mut total = 0_u64;
    for cell in cells {
        for record in super::history_artifacts::records(
            &cell.directory.join(format!("c-{level}-requests.jsonl")),
        )? {
            if record.get("error").is_none() {
                let tokens = record["completion_tokens"]
                    .as_u64()
                    .ok_or("raw request is missing completion tokens")?;
                total = total
                    .checked_add(tokens)
                    .ok_or("history output token count overflow")?;
            }
        }
    }
    Ok(total)
}
