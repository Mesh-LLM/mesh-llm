//! Audit the whole-trace budget used by the pinned upstream Weka loader.
//! Selection itself remains owned by AgentX's --max-context-length option.
use super::{Prepared, Profile, write};
use crate::command::DynResult;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(super) const LIMIT: u64 = 131_072;

pub(super) fn describe(trace: &Value, bytes: &[u8]) -> DynResult<Value> {
    let peak = peak_context(
        trace["requests"]
            .as_array()
            .ok_or("missing trace requests")?,
    )?;
    Ok(json!({
        "trace_id":trace["id"].as_str().ok_or("missing trace ID")?,
        "trace_sha256":hex::encode(Sha256::digest(bytes)),
        "recorded_peak_input_plus_output_tokens":peak,
        "status":if peak <= LIMIT { "eligible" } else { "unsupported_context" },
    }))
}

fn peak_context(requests: &[Value]) -> DynResult<u64> {
    let mut peak = 0;
    for request in requests {
        let tokens = match request["type"].as_str() {
            Some("subagent") => peak_context(
                request["requests"]
                    .as_array()
                    .ok_or("missing child requests")?,
            )?,
            Some("n" | "s") => {
                let input = request["in"].as_u64().ok_or("invalid input token count")?;
                // AgentX emits at least one output token even for recorded out=0.
                let output = request["out"]
                    .as_u64()
                    .ok_or("invalid output token count")?
                    .max(1);
                input
                    .checked_add(output)
                    .ok_or("context token count overflow")?
            }
            _ => return Err("unknown Weka request type".into()),
        };
        peak = peak.max(tokens);
    }
    Ok(peak)
}

pub(super) fn record(input: &Prepared, concurrency: u32) -> DynResult<Value> {
    let source = input
        .dataset
        .parent()
        .ok_or("missing cohort parent")?
        .join("cohort.json");
    let cohort: Value = serde_json::from_slice(&std::fs::read(source)?)?;
    let traces = cohort["trace_context"]
        .as_array()
        .ok_or("missing trace context audit")?;
    let selection = selection(traces, input.profile, concurrency)?;
    write(&input.output.join("cohort.json"), &selection)?;
    Ok(selection)
}

fn selection(traces: &[Value], profile: Profile, concurrency: u32) -> DynResult<Value> {
    let eligible: Vec<_> = traces
        .iter()
        .filter(|trace| trace["status"] == "eligible")
        .collect();
    let limit = match profile {
        Profile::Smoke => concurrency.max(4) as usize,
        Profile::Contention => super::contention_cohort::SESSIONS,
        _ => eligible.len(),
    };
    if limit == 0 || eligible.len() < limit {
        return Err("not enough complete context-compatible traces for offered concurrency".into());
    }
    let selected = &eligible[..limit];
    // IDs plus original row digests make independent engine jobs comparable.
    let digest = hex::encode(Sha256::digest(serde_json::to_vec(selected)?));
    Ok(json!({
        "harness_version":profile.harness_version(),
        "policy":"common_whole_trace_recorded_input_plus_output",
        "max_context_tokens":LIMIT,
        "token_count_scope":"recorded_counts_not_rendered_chat_template_tokens",
        "source_traces":traces.len(),
        "eligible_traces":eligible.len(),
        "excluded_unsupported_context_traces":traces.len()-eligible.len(),
        "selected_traces":selected,
        "selected_cohort_sha256":digest,
        "exclusion_manifest":"../cohort.json",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_output_and_every_child_and_accepts_exact_boundary() {
        for (requests, expected) in [
            (json!([{"type":"n","in":131071,"out":1}]), "eligible"),
            (
                json!([{"type":"s","in":131071,"out":2}]),
                "unsupported_context",
            ),
            (
                json!([{"type":"s","in":131072,"out":0}]),
                "unsupported_context",
            ),
            (
                json!([{"type":"n","in":8,"out":2},{"type":"subagent","requests":[{"type":"s","in":131070,"out":3}]}]),
                "unsupported_context",
            ),
        ] {
            let trace = json!({"id":"trace","requests":requests});
            assert_eq!(
                describe(&trace, b"unchanged row").unwrap()["status"],
                expected
            );
        }
    }

    #[test]
    fn malformed_counts_fail_closed() {
        for request in [
            json!({"type":"n","in":-1,"out":2}),
            json!({"type":"n","in":1,"out":1.5}),
            json!({"type":"n","in":u64::MAX,"out":1}),
            json!({"type":"subagent"}),
            json!({"type":"unknown"}),
        ] {
            assert!(peak_context(&[request]).is_err());
        }
    }

    #[test]
    fn context_filter_precedes_smoke_cap_and_is_identical_across_concurrencies() {
        let traces: Vec<_> = (0..8)
            .map(|index| {
                json!({
                    "trace_id":index.to_string(),
                    "status":if index < 3 { "unsupported_context" } else { "eligible" },
                })
            })
            .collect();
        let first = selection(&traces, Profile::Smoke, 1).unwrap();
        assert_eq!(first["selected_traces"][0]["trace_id"], "3");
        assert_eq!(first["selected_traces"].as_array().unwrap().len(), 4);
        assert_eq!(first["excluded_unsupported_context_traces"], 3);
        for concurrency in [2, 4] {
            assert_eq!(
                first,
                selection(&traces, Profile::Smoke, concurrency).unwrap()
            );
        }
        assert_eq!(
            selection(&traces, Profile::Benchmark, 4).unwrap()["selected_traces"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
        assert!(selection(&traces[..6], Profile::Smoke, 4).is_err());
        assert!(selection(&[], Profile::Benchmark, 4).is_err());
    }
}
