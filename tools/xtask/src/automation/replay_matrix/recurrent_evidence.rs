use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Deserialize, PartialEq)]
pub(super) struct Lookup {
    pub event: String,
    pub start_time_unix_nanos: u64,
    pub attributes: Attributes,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
    #[serde(skip)]
    raw_identity: Option<Vec<u8>>,
}

pub(super) fn read_logs(paths: &[std::path::PathBuf]) -> crate::command::DynResult<Vec<Lookup>> {
    use std::io::{BufRead, BufReader, Read};
    let mut lookups = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for path in paths {
        let mut reader = BufReader::new(std::fs::File::open(path)?);
        loop {
            let mut line = Vec::new();
            let length = reader
                .by_ref()
                .take(1024 * 1024 + 1)
                .read_until(b'\n', &mut line)?;
            if length == 0 {
                break;
            }
            if length > 1024 * 1024 {
                return Err("recurrent log line exceeds 1 MiB".into());
            }
            while line
                .last()
                .is_some_and(|byte| matches!(byte, b'\r' | b'\n'))
            {
                line.pop();
            }
            let text = String::from_utf8_lossy(&line);
            let Ok(event) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            if event["event"] != "stage.openai_kv_lookup_decision" {
                continue;
            }
            if !seen.insert(text.as_bytes().to_vec()) {
                continue;
            }
            let mut lookup: Lookup = serde_json::from_value(event)?;
            lookup.raw_identity = Some(text.as_bytes().to_vec());
            lookups.push(lookup);
        }
    }
    Ok(lookups)
}

#[derive(Deserialize, Serialize, PartialEq)]
pub(super) struct Attributes {
    #[serde(rename = "openai.prompt_cache_key")]
    pub session: String,
    #[serde(rename = "skippy.kv.decision")]
    pub decision: String,
    #[serde(rename = "skippy.exact_cache.payload_kind")]
    pub payload: Option<String>,
    #[serde(default, rename = "skippy.exact_cache.restored_tokens")]
    pub restored_tokens: u64,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Serialize)]
pub(super) struct Turn<'a> {
    request_id: &'a str,
    state_restored: bool,
    restored_tokens: u64,
    lookup: &'a Attributes,
}

#[derive(Serialize)]
pub(super) struct Evidence<'a> {
    pub passed: bool,
    pub problems: Vec<String>,
    restores: usize,
    minimum_restored_tokens: u64,
    turns: Vec<Turn<'a>>,
}

pub(super) fn evaluate<'a>(
    requests: &'a [super::session_evidence::Request],
    lookups: &'a [Lookup],
    minimum: u64,
) -> Evidence<'a> {
    let mut grouped = BTreeMap::<&str, Vec<&super::session_evidence::Request>>::new();
    for request in requests {
        grouped
            .entry(&request.session_id)
            .or_default()
            .push(request);
    }
    let mut decisions = BTreeMap::<&str, Vec<&Lookup>>::new();
    let mut seen = Vec::new();
    for lookup in lookups {
        if lookup.event == "stage.openai_kv_lookup_decision" && !seen.contains(&lookup) {
            seen.push(lookup);
            decisions
                .entry(&lookup.attributes.session)
                .or_default()
                .push(lookup);
        }
    }
    let mut evidence = Evidence {
        passed: !grouped.is_empty() && minimum > 0,
        problems: Vec::new(),
        restores: 0,
        minimum_restored_tokens: minimum,
        turns: Vec::new(),
    };
    for (session, requests) in grouped {
        let mut events = decisions.remove(session).unwrap_or_default();
        events.sort_by_key(|event| event.start_time_unix_nanos);
        if events.len() != requests.len() {
            evidence.problems.push(format!(
                "{session}: {} lookup events for {} turns",
                events.len(),
                requests.len()
            ));
            continue;
        }
        let mut restores = 0;
        for (request, event) in requests.iter().zip(events) {
            let attributes = &event.attributes;
            let restored = attributes.decision == "exact_hit"
                && matches!(
                    attributes.payload.as_deref(),
                    Some("kv-recurrent" | "recurrent-only")
                )
                && attributes.restored_tokens >= minimum
                && request
                    .prompt_tokens
                    .is_some_and(|tokens| attributes.restored_tokens <= tokens);
            if restored && request.assistant_turn > 0 {
                restores += 1;
            }
            evidence.turns.push(Turn {
                request_id: &request.request_id,
                state_restored: restored,
                restored_tokens: if restored {
                    attributes.restored_tokens
                } else {
                    0
                },
                lookup: attributes,
            });
        }
        evidence.restores += restores;
        if restores == 0 {
            evidence.problems.push(format!(
                "{session}: no recurrent restore on a later recorded-history turn"
            ));
        }
    }
    evidence.passed &= evidence.problems.is_empty();
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_lookup_extensions_do_not_collapse_matching_core_fields() {
        let requests = serde_json::from_value::<Vec<super::super::session_evidence::Request>>(
            serde_json::json!([
                {"session_id":"s","request_id":"s:0","assistant_turn":0,"prompt_tokens":40000},
                {"session_id":"s","request_id":"s:1","assistant_turn":1,"prompt_tokens":41000}
            ]),
        )
        .unwrap();
        let lookups = serde_json::from_value::<Vec<Lookup>>(serde_json::json!([
            {"event":"stage.openai_kv_lookup_decision","start_time_unix_nanos":1,"span_id":"first","attributes":{"openai.prompt_cache_key":"s","skippy.kv.decision":"exact_hit","skippy.exact_cache.payload_kind":"kv-recurrent","skippy.exact_cache.restored_tokens":39000,"slot":1}},
            {"event":"stage.openai_kv_lookup_decision","start_time_unix_nanos":1,"span_id":"second","attributes":{"openai.prompt_cache_key":"s","skippy.kv.decision":"exact_hit","skippy.exact_cache.payload_kind":"kv-recurrent","skippy.exact_cache.restored_tokens":39000,"slot":2}}
        ])).unwrap();
        let evidence = evaluate(&requests, &lookups, 32768);
        assert!(evidence.passed, "{:?}", evidence.problems);
        assert_eq!(evidence.turns.len(), 2);
        assert_eq!(evidence.turns[1].lookup.extensions["slot"], 2);
    }

    #[test]
    fn only_correlated_later_turn_recurrent_payloads_certify() {
        let requests = serde_json::from_value::<Vec<super::super::session_evidence::Request>>(
            serde_json::json!([
                {"session_id":"s","request_id":"s:0","assistant_turn":0,"prompt_tokens":40000},
                {"session_id":"s","request_id":"s:1","assistant_turn":1,"prompt_tokens":41000}
            ]),
        )
        .unwrap();
        let mut lookups = serde_json::from_value::<Vec<Lookup>>(serde_json::json!([
            {"event":"stage.openai_kv_lookup_decision","start_time_unix_nanos":0,"attributes":{"openai.prompt_cache_key":"s","skippy.kv.decision":"miss"}},
            {"event":"stage.openai_kv_lookup_decision","start_time_unix_nanos":1,"attributes":{"openai.prompt_cache_key":"s","skippy.kv.decision":"exact_hit","skippy.exact_cache.payload_kind":"kv-recurrent","skippy.exact_cache.restored_tokens":39000}}
        ])).unwrap();
        assert!(evaluate(&requests, &lookups, 32768).passed);
        lookups[1].attributes.decision = "miss".into();
        assert!(!evaluate(&requests, &lookups, 32768).passed);
        lookups[1].attributes.decision = "exact_hit".into();
        lookups[1].attributes.restored_tokens = 999999;
        assert!(!evaluate(&requests, &lookups, 32768).passed);
        lookups[1].attributes.restored_tokens = 39000;
        lookups[1].attributes.payload = Some("full-state".into());
        assert!(!evaluate(&requests, &lookups, 32768).passed);
        lookups[1].attributes.payload = Some("kv-recurrent".into());
        lookups[1].attributes.restored_tokens = 1;
        assert!(!evaluate(&requests, &lookups, 32768).passed);
        assert!(!evaluate(&requests, &lookups[..1], 1).passed);
    }
}
