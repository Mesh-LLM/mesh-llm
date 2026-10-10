use super::session_evidence::{Request, Trajectory, complete};
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) fn summarize(trajectories: &[Trajectory], records: &[Value]) -> DynResult<Vec<Value>> {
    trajectories.iter().map(|trajectory| {
        let rows = records.iter().filter(|record| record["session_id"].as_str() == Some(&trajectory.session_id)).collect::<Vec<_>>();
        let requests = rows.iter().map(|record| serde_json::from_value::<Request>((*record).clone())).collect::<Result<Vec<_>, _>>()?;
        let completeness = complete(std::slice::from_ref(trajectory), &requests);
        let successful = rows.iter().filter(|record| record.get("error").is_none()).collect::<Vec<_>>();
        let sum = |key: &str| successful.iter().try_fold(0_u64, |sum, record| {
            let value = record[key].as_u64().ok_or("missing session token metric")?;
            sum.checked_add(value).ok_or("session token sum overflow")
        });
        let prompt = sum("prompt_tokens")?;
        let cached = sum("cached_tokens")?;
        let turns = rows.iter().map(|record| json!({
            "request_id":record["request_id"], "prompt_tokens":record["prompt_tokens"],
            "cached_tokens":record["cached_tokens"], "cache_pct":record["cache_pct"],
            "ttft_seconds":record["ttft_seconds"], "error":record["error"]
        })).collect::<Vec<_>>();
        Ok(json!({
            "session_id":trajectory.session_id, "expected_request_ids":completeness.expected_request_ids,
            "complete":completeness.passed, "prompt_tokens":prompt, "cached_tokens":cached,
            "cache_pct":if prompt == 0 { None } else { Some(100.0 * super::stream_evidence::number(cached) / super::stream_evidence::number(prompt)) },
            "max_prompt_tokens":successful.iter().filter_map(|record|record["prompt_tokens"].as_u64()).max().unwrap_or(0),
            "turns":turns
        }))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_turns_remain_visible_but_do_not_contribute_token_metrics() {
        let trajectories = serde_json::from_value::<Vec<Trajectory>>(json!([
            {"session_id":"s","messages":[{"role":"assistant"},{"role":"assistant"}]}
        ]))
        .unwrap();
        let rows = vec![
            json!({"session_id":"s","request_id":"s:0","prompt_tokens":40,"cached_tokens":30,"cache_pct":75.0,"ttft_seconds":1}),
            json!({"session_id":"s","request_id":"s:1","error":"HTTP 500"}),
        ];
        let sessions = summarize(&trajectories, &rows).unwrap();
        assert_eq!(sessions[0]["complete"], false);
        assert_eq!(sessions[0]["prompt_tokens"], 40);
        assert_eq!(sessions[0]["cache_pct"], 75.0);
        assert_eq!(sessions[0]["turns"][1]["error"], "HTTP 500");
    }
}
