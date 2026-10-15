//! Request-scoped metrics projected from the retained exchange evidence.
use super::{
    progress::Metrics,
    stream_evidence::{Evidence, number},
};

pub(super) fn from_evidence(evidence: &Evidence) -> Metrics {
    Metrics {
        cached_tokens: Some(evidence.cached_tokens),
        unavailable_metrics: unavailable(evidence),
        prompt_tokens: Some(evidence.prompt_tokens),
        completion_tokens: Some(evidence.completion_tokens),
        cache_pct: Some(evidence.cache_pct),
        decode_tokens_per_second: decode_rate(
            evidence.completion_tokens,
            evidence.generation_seconds,
        ),
        ttft_seconds: Some(evidence.ttft_seconds),
        output_stream_tokens_per_second: decode_rate(
            evidence.completion_tokens,
            evidence.output_stream_seconds,
        ),
        terminal_seconds: Some(evidence.terminal_seconds),
        ..Default::default()
    }
}

fn unavailable(evidence: &Evidence) -> std::collections::BTreeMap<String, String> {
    let mut reasons = std::collections::BTreeMap::from([
        ("native_prefill_tokens_per_second".into(), "Not exposed by the common OpenAI stream; TTFT includes more than prefill.".into()),
        ("native_decode_tokens_per_second".into(), "Not exposed by the common OpenAI stream; client stream throughput is a separate metric.".into()),
    ]);
    for (name, seconds) in [
        ("decode_tokens_per_second", evidence.generation_seconds),
        (
            "output_stream_tokens_per_second",
            evidence.output_stream_seconds,
        ),
    ] {
        if decode_rate(evidence.completion_tokens, seconds).is_none() {
            reasons.insert(name.into(), "No finite positive observed interval.".into());
        }
    }
    reasons
}

fn decode_rate(tokens: u64, seconds: f64) -> Option<f64> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    let rate = number(tokens) / seconds;
    rate.is_finite().then_some(rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_metrics_include_observed_latency_and_cache_without_aggregate_labels() {
        let evidence = Evidence {
            output: None,
            ttft_seconds: 10.0,
            elapsed_seconds: 18.0,
            generation_seconds: 8.0,
            output_stream_seconds: 3.0,
            terminal_seconds: 5.0,
            content_characters: 120,
            reasoning_characters: 0,
            tool_calls: 0,
            decode_inter_token_seconds: vec![],
            completion_tokens: 120,
            prompt_tokens: 1000,
            cached_tokens: 900,
            cache_pct: 90.0,
            content_events: 120,
            finish_reason: Some("stop".into()),
            output_problems: vec![],
            content_sha256: String::new(),
            first_generated_sha256: None,
        };
        let value = serde_json::to_value(from_evidence(&evidence)).unwrap();
        assert_eq!(value["decode_tokens_per_second"], 15.0);
        assert_eq!(value["ttft_seconds"], 10.0);
        assert_eq!(value["cache_pct"], 90.0);
        assert!(value.get("ttft_p50_seconds").is_none());
        assert!(value.get("workload_output_tokens_per_second").is_none());
    }
    #[test]
    fn unavailable_generation_intervals_do_not_invent_throughput() {
        for seconds in [0.0, -1.0, f64::INFINITY, f64::NAN, f64::MIN_POSITIVE] {
            assert_eq!(decode_rate(120, seconds), None);
        }
    }
}
