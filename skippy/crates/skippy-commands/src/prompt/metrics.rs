//! Per-response usage, streaming latency, and the human prompt footer.

use std::time::Duration;

use serde_json::Value;

#[derive(Default)]
pub(super) struct PromptMetrics {
    first_token: Option<Duration>,
    last_token: Option<Duration>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    server_tps: Option<f64>,
}

impl PromptMetrics {
    pub(super) fn observe(&mut self, chunk: &Value, raw: bool, elapsed: Duration) {
        let choice = &chunk["choices"][0];
        let generated = if raw {
            nonempty(&choice["text"])
        } else {
            nonempty(&choice["delta"]["content"]) || nonempty(&choice["delta"]["reasoning_content"])
        };
        if generated {
            self.first_token.get_or_insert(elapsed);
            self.last_token = Some(elapsed);
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| usage.is_object()) {
            self.input_tokens = usage["prompt_tokens"].as_u64();
            self.output_tokens = usage["completion_tokens"].as_u64();
            self.cached_tokens = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
        }
        if let Some(tps) = chunk["timings"]["predicted_per_second"].as_f64()
            && tps.is_finite()
            && tps > 0.0
        {
            self.server_tps = Some(tps);
        }
    }

    fn tokens_per_second(&self) -> Option<f64> {
        if let Some(tps) = self.server_tps {
            return Some(tps);
        }
        let tokens = self.output_tokens?.checked_sub(1)?;
        let elapsed = self
            .last_token?
            .checked_sub(self.first_token?)?
            .as_secs_f64();
        (tokens > 0 && elapsed > 0.0).then(|| tokens as f64 / elapsed)
    }

    fn cached_label(&self) -> String {
        let (Some(cached), Some(input)) = (self.cached_tokens, self.input_tokens) else {
            return "—".to_owned();
        };
        if cached > input {
            return "—".to_owned();
        }
        let percent = if input == 0 {
            0.0
        } else {
            cached as f64 * 100.0 / input as f64
        };
        format!("{} ({percent:.0}%)", token_count(Some(cached)))
    }

    pub(super) fn footer(&self, total: Duration, interrupted: bool) -> String {
        let tps = self
            .tokens_per_second()
            .map_or_else(|| "— tok/s".to_owned(), |tps| format!("{tps:.1} tok/s"));
        let ttft = self.first_token.map_or_else(
            || "—".to_owned(),
            |elapsed| format!("{:.0} ms", elapsed.as_secs_f64() * 1000.0),
        );
        let state = if interrupted { "Interrupted · " } else { "" };
        format!(
            "⚡ {state}{tps} · ⏱ TTFT {ttft} · Total {:.2} s · 📝 {} in / {} out · ♻️ Cached {}",
            total.as_secs_f64(),
            token_count(self.input_tokens),
            token_count(self.output_tokens),
            self.cached_label(),
        )
    }
}

fn nonempty(value: &Value) -> bool {
    value.as_str().is_some_and(|text| !text.is_empty())
}

fn token_count(count: Option<u64>) -> String {
    let Some(count) = count else {
        return "—".to_owned();
    };
    let digits = count.to_string();
    let mut formatted = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(digit);
    }
    formatted
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn footer_matches_the_compact_emoji_layout() {
        let mut metrics = PromptMetrics::default();
        metrics.observe(
            &json!({"choices": [{"delta": {"content": "hello"}}]}),
            false,
            Duration::from_millis(184),
        );
        metrics.observe(
            &json!({
                "choices": [],
                "usage": {
                    "prompt_tokens": 1280, "completion_tokens": 128,
                    "prompt_tokens_details": {"cached_tokens": 1024}
                },
                "timings": {"predicted_per_second": 43.2}
            }),
            false,
            Duration::from_secs(3),
        );
        assert_eq!(
            metrics.footer(Duration::from_millis(3120), false),
            "⚡ 43.2 tok/s · ⏱ TTFT 184 ms · Total 3.12 s · 📝 1,280 in / 128 out · ♻️ Cached 1,024 (80%)"
        );
    }

    #[test]
    fn throughput_fallback_excludes_initial_wait_and_stream_tail() {
        for raw in [false, true] {
            let mut metrics = PromptMetrics::default();
            let delta = if raw {
                json!({"choices": [{"text": "hi"}]})
            } else {
                json!({"choices": [{"delta": {"content": "hi"}}]})
            };
            metrics.observe(&delta, raw, Duration::from_secs(2));
            metrics.observe(&delta, raw, Duration::from_millis(2500));
            metrics.observe(
                &json!({"usage": {"prompt_tokens": 20, "completion_tokens": 6}}),
                raw,
                Duration::from_secs(3),
            );
            assert_eq!(metrics.tokens_per_second(), Some(10.0));
            assert!(
                metrics
                    .footer(Duration::from_secs(3), false)
                    .contains("TTFT 2000 ms")
            );
        }
    }

    #[test]
    fn empty_role_events_do_not_start_ttft_but_reasoning_does() {
        let mut metrics = PromptMetrics::default();
        metrics.observe(
            &json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            false,
            Duration::from_millis(10),
        );
        assert_eq!(metrics.first_token, None);
        metrics.observe(
            &json!({"choices": [{"delta": {"reasoning_content": "thinking"}}]}),
            false,
            Duration::from_millis(50),
        );
        assert_eq!(metrics.first_token, Some(Duration::from_millis(50)));
    }

    #[test]
    fn missing_usage_is_unknown_and_interrupted_output_is_labelled() {
        let metrics = PromptMetrics::default();
        assert_eq!(
            metrics.footer(Duration::from_millis(250), true),
            "⚡ Interrupted · — tok/s · ⏱ TTFT — · Total 0.25 s · 📝 — in / — out · ♻️ Cached —"
        );
    }

    #[test]
    fn cache_data_distinguishes_zero_unknown_and_invalid_counts() {
        let mut metrics = PromptMetrics::default();
        for (input, cached, expected) in [
            (10, Value::Null, "—"),
            (10, json!(0), "0 (0%)"),
            (0, json!(0), "0 (0%)"),
            (10, json!(11), "—"),
        ] {
            metrics.observe(
                &json!({"usage": {"prompt_tokens": input, "prompt_tokens_details": {"cached_tokens": cached}}}),
                false,
                Duration::ZERO,
            );
            assert_eq!(metrics.cached_label(), expected);
        }
    }

    #[test]
    fn single_token_and_zero_time_do_not_produce_invalid_throughput() {
        let mut metrics = PromptMetrics::default();
        metrics.observe(
            &json!({"choices": [{"text": "hello"}], "usage": {"completion_tokens": 1}}),
            true,
            Duration::from_millis(10),
        );
        assert_eq!(metrics.tokens_per_second(), None);
        metrics.observe(
            &json!({"usage": {"completion_tokens": 10}, "timings": {"predicted_per_second": -1}}),
            true,
            Duration::from_millis(10),
        );
        assert_eq!(metrics.tokens_per_second(), None);
    }
}
