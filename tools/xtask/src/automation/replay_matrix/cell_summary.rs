use super::{
    recorded_requests::Trajectory,
    session_evidence::{self, Request},
};
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) fn summarize(
    trajectories: &[Trajectory],
    records: &[Value],
    concurrency: usize,
) -> DynResult<Value> {
    let cohort_sha256 = super::cohort_identity::digest(&Value::Array(
        trajectories
            .iter()
            .map(|trajectory| trajectory.original.clone())
            .collect(),
    ))?;
    let trajectories = trajectories
        .iter()
        .map(|trajectory| {
            serde_json::from_value::<session_evidence::Trajectory>(
                json!({"session_id":trajectory.session_id,"messages":trajectory.messages}),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let requests = records
        .iter()
        .map(|record| serde_json::from_value::<Request>(record.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let completeness = session_evidence::complete(&trajectories, &requests);
    let sessions = super::session_summary::summarize(&trajectories, records)?;
    let complete_sessions = sessions
        .iter()
        .filter(|session| session["complete"] == true)
        .count();
    let successful = records
        .iter()
        .filter(|record| record.get("error").is_none())
        .collect::<Vec<_>>();
    let sum = |field: &str| {
        successful
            .iter()
            .map(|record| {
                record[field]
                    .as_f64()
                    .ok_or_else(|| format!("missing numeric metric {field}"))
            })
            .sum::<Result<f64, _>>()
    };
    let completion = sum("completion_tokens")?;
    let prompt = sum("prompt_tokens")?;
    let cached = sum("cached_tokens")?;
    let generation = sum("generation_seconds")?;
    let elapsed = sum("elapsed_seconds")?;
    let started = successful
        .iter()
        .filter_map(|record| record["started"].as_f64())
        .min_by(f64::total_cmp);
    let completed = successful
        .iter()
        .filter_map(|record| record["completed"].as_f64())
        .max_by(f64::total_cmp);
    let window = started
        .zip(completed)
        .map(|(started, completed)| completed - started)
        .unwrap_or(0.0);
    let ratio = |numerator, denominator| {
        if denominator > 0.0 {
            Some(numerator / denominator)
        } else {
            None
        }
    };
    let mut ttft = successful
        .iter()
        .filter_map(|record| record["ttft_seconds"].as_f64())
        .collect::<Vec<_>>();
    ttft.sort_by(f64::total_cmp);
    let percentile = |values: &[f64], percent: usize| -> DynResult<Option<f64>> {
        if values.is_empty() {
            return Ok(None);
        }
        let rank = values
            .len()
            .checked_mul(percent)
            .ok_or("percentile rank overflow")?
            .div_ceil(100);
        Ok(Some(values[rank.saturating_sub(1).min(values.len() - 1)]))
    };
    let mut decode_intervals = successful
        .iter()
        .flat_map(|record| {
            record["decode_inter_token_seconds"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_f64)
        })
        .collect::<Vec<_>>();
    decode_intervals.sort_by(f64::total_cmp);
    let p95 = percentile(&ttft, 95)?;
    let decode_p99 = percentile(&decode_intervals, 99)?;
    let median = if ttft.is_empty() {
        None
    } else {
        let middle = ttft.len() / 2;
        Some(if ttft.len() % 2 == 0 {
            (ttft[middle - 1] + ttft[middle]) / 2.0
        } else {
            ttft[middle]
        })
    };
    let ids = |failed: bool| {
        let mut ids = records
            .iter()
            .filter(|record| record.get("error").is_some() == failed)
            .filter_map(|record| record["request_id"].as_str())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    };
    let mean = ratio(elapsed, window);
    let offered = number(concurrency)?;
    let content_hashes = successful
        .iter()
        .filter_map(|record| {
            Some((
                record["request_id"].as_str()?,
                record["content_sha256"].as_str()?,
            ))
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let prompt_min = successful
        .iter()
        .filter_map(|record| record["prompt_tokens"].as_u64())
        .min();
    let prompt_max = successful
        .iter()
        .filter_map(|record| record["prompt_tokens"].as_u64())
        .max();
    let exhausted = successful
        .iter()
        .filter(|record| record["completion_tokens"] == record["requested_output_tokens"])
        .count();
    let length_finishes = successful
        .iter()
        .filter(|record| record["finish_reason"] == "length")
        .count();
    Ok(json!({
        "concurrency":concurrency,"trajectories":trajectories.len(),"requests":records.len(),
        "session_cohort_sha256":cohort_sha256,
        "sessions":sessions,"successful_trajectories":complete_sessions,
        "failed_trajectories":trajectories.len()-complete_sessions,
        "recorded_assistant_turns":completeness.expected_turns,
        "successful_requests":successful.len(),"failed_requests":records.len()-successful.len(),
        "successful_request_ids":ids(false),"failed_request_ids":ids(true),
        "completion_tokens":completion,"prompt_tokens":prompt,"cached_tokens":cached,
        "generation_seconds":generation,"workload_window_seconds":window,
        "ttft_samples":ttft,"ttft_p50_seconds":median,"ttft_p95_seconds":p95,
        "decode_inter_token_p99_seconds":decode_p99,
        "prompt_tokens_min":prompt_min,"prompt_tokens_max":prompt_max,
        "content_sha256_by_request":content_hashes,
        "budget_exhausted_requests":exhausted,"finish_reason_length_requests":length_finishes,
        "agent_steps_per_second":ratio(number(successful.len())?,window),
        "workload_output_tokens_per_second":ratio(completion,window),
        "decode_tokens_per_second":ratio(completion,generation),"mean_in_flight":mean,
        "concurrency_utilization_pct":mean.map(|mean|100.0*mean/offered),
        "cache_pct":ratio(100.0*cached,prompt),"ordered_replay":true,"replay_mode":"all",
        "completeness":completeness,"acceptance":{"passed":completeness.passed,"problems":completeness.problems}
    }))
}

fn number(value: usize) -> DynResult<f64> {
    Ok(f64::from(u32::try_from(value)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_uses_token_weighted_generation_and_shared_workload_window() {
        let trajectories: Vec<Trajectory> = serde_json::from_value(json!([
            {"session_id":"s","source_dataset":"fixture","agent_framework":"fixture",
             "messages":[{"role":"assistant"},{"role":"assistant"}]}
        ]))
        .unwrap();
        let records = vec![
            json!({"session_id":"s","request_id":"s:0","prompt_tokens":100,"cached_tokens":0,
                "completion_tokens":20,"generation_seconds":2,"elapsed_seconds":3,"started":0,"completed":3,"ttft_seconds":1}),
            json!({"session_id":"s","request_id":"s:1","prompt_tokens":200,"cached_tokens":150,
                "completion_tokens":80,"generation_seconds":4,"elapsed_seconds":5,"started":3,"completed":8,"ttft_seconds":1}),
        ];
        let report = summarize(&trajectories, &records, 2).unwrap();
        assert_eq!(report["completeness"]["passed"], true);
        assert_eq!(report["completion_tokens"], 100.0);
        assert_eq!(report["workload_window_seconds"], 8.0);
        assert_eq!(report["cache_pct"], 50.0);
        assert!((report["decode_tokens_per_second"].as_f64().unwrap() - 100.0 / 6.0).abs() < 1e-9);
        assert_eq!(report["workload_output_tokens_per_second"], 12.5);
        let retained: Value =
            serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        assert_eq!(
            retained, report,
            "retained metrics must preserve exact numeric evidence"
        );
    }
}
