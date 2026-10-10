use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
pub(super) struct Qualification {
    pub expected_prompt_tokens: BTreeMap<String, u64>,
    pub require_later_turn_reuse: bool,
}

pub(super) fn problems(
    records: &[serde_json::Value],
    qualification: &Qualification,
) -> Vec<String> {
    let mut problems = Vec::new();
    let mut sessions = BTreeSet::new();
    let mut reused = BTreeSet::new();
    let mut observed = BTreeSet::new();
    for record in records {
        let Some(request) = record["request_id"].as_str() else {
            problems.push("measured record lacks request identity".into());
            continue;
        };
        observed.insert(request);
        if record["prompt_tokens"].as_u64()
            != qualification.expected_prompt_tokens.get(request).copied()
        {
            problems.push(format!(
                "{request}: measured prompt token count differs from context preflight"
            ));
        }
        if let Some(session) = record["session_id"].as_str() {
            sessions.insert(session);
            if record.get("error").is_none()
                && record["assistant_turn"]
                    .as_u64()
                    .is_some_and(|turn| turn > 0)
                && record["cached_tokens"]
                    .as_u64()
                    .is_some_and(|tokens| tokens > 0)
            {
                reused.insert(session);
            }
        } else {
            problems.push(format!("{request}: missing session identity"));
        }
    }
    if observed
        != qualification
            .expected_prompt_tokens
            .keys()
            .map(String::as_str)
            .collect()
    {
        problems.push("measured request identities differ from context preflight".into());
    }
    if qualification.require_later_turn_reuse {
        problems.extend(
            sessions
                .difference(&reused)
                .map(|session| format!("{session}: no later-turn prompt reuse")),
        );
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualification() -> Qualification {
        Qualification {
            expected_prompt_tokens: BTreeMap::from([("s:0".into(), 40), ("s:1".into(), 41)]),
            require_later_turn_reuse: true,
        }
    }

    #[test]
    fn matching_prefixes_with_later_reuse_pass() {
        let records = serde_json::json!([
            {"request_id":"s:0","session_id":"s","assistant_turn":0,"prompt_tokens":40,"cached_tokens":0},
            {"request_id":"s:1","session_id":"s","assistant_turn":1,"prompt_tokens":41,"cached_tokens":30}
        ]);
        assert!(problems(records.as_array().unwrap(), &qualification()).is_empty());
    }

    #[test]
    fn first_turn_cache_cannot_substitute_for_later_reuse() {
        let records = serde_json::json!([
            {"request_id":"s:0","session_id":"s","assistant_turn":0,"prompt_tokens":40,"cached_tokens":30},
            {"request_id":"s:1","session_id":"s","assistant_turn":1,"prompt_tokens":41,"cached_tokens":0}
        ]);
        assert!(!problems(records.as_array().unwrap(), &qualification()).is_empty());
    }

    #[test]
    fn changed_prefix_counts_and_missing_requests_fail() {
        let records = serde_json::json!([{ "request_id":"s:0","session_id":"s","assistant_turn":0,"prompt_tokens":39,"cached_tokens":30 }]);
        assert_eq!(
            problems(records.as_array().unwrap(), &qualification()).len(),
            3
        );
    }
}
