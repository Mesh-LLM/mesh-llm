#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::network::openai) struct CacheCostObservation {
    pub(in crate::network::openai) queue_delay_micros: u64,
    pub(in crate::network::openai) restore_micros: u64,
    pub(in crate::network::openai) prefill_micros_per_token: Option<u64>,
}

fn timing_micros(timings: &serde_json::Value, name: &str) -> Option<u64> {
    let millis = timings.get(name)?.as_f64()?;
    if !millis.is_finite() || millis < 0.0 || millis > u64::MAX as f64 / 1_000.0 {
        return None;
    }
    Some((millis * 1_000.0).round() as u64)
}

/// Parse bounded, provider-reported cache cost measurements. Third-party
/// providers simply omit this Skippy extension and retain conservative routing
/// defaults.
pub(in crate::network::openai) fn parse_cache_cost_from_json_body(
    body: &[u8],
) -> Option<CacheCostObservation> {
    let json = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    let timings = json.get("timings")?;
    let queue_delay_micros = timing_micros(timings, "queue_wait_ms").unwrap_or_default();
    let restore_micros = timing_micros(timings, "cache_restore_ms").unwrap_or_default();
    let prompt_micros = timing_micros(timings, "prompt_ms");
    let suffix_prefill_tokens = timings
        .get("suffix_prefill_n")
        .and_then(serde_json::Value::as_u64);
    let prefill_micros_per_token = match (prompt_micros, suffix_prefill_tokens) {
        (Some(prompt), Some(tokens)) if tokens > 0 => Some(
            prompt
                .saturating_sub(restore_micros)
                .div_ceil(tokens)
                .max(1),
        ),
        _ => None,
    };
    Some(CacheCostObservation {
        queue_delay_micros,
        restore_micros,
        prefill_micros_per_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_separates_restore_from_suffix_prefill() {
        let response = serde_json::json!({
            "timings": {
                "prompt_ms": 12.5,
                "queue_wait_ms": 1.25,
                "cache_restore_ms": 2.5,
                "suffix_prefill_n": 5
            }
        });

        assert_eq!(
            parse_cache_cost_from_json_body(response.to_string().as_bytes()),
            Some(CacheCostObservation {
                queue_delay_micros: 1_250,
                restore_micros: 2_500,
                prefill_micros_per_token: Some(2_000),
            })
        );
        assert_eq!(parse_cache_cost_from_json_body(br#"{}"#), None);
    }
}
