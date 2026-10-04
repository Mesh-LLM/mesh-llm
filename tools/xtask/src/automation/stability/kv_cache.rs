use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) struct Metrics {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
}

impl Metrics {
    pub fn from_response(response: &Value) -> Self {
        let usage = &response["usage"];
        Self {
            prompt_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
            cached_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
        }
    }

    pub fn validate(self, minimum_cached: u64, suffix_limit: u64) -> Result<String, String> {
        if self.cached_tokens < minimum_cached {
            return Err(format!(
                "cached_tokens={} below required minimum {minimum_cached}; prompt_tokens={}",
                self.cached_tokens, self.prompt_tokens
            ));
        }
        let suffix = self.prompt_tokens.saturating_sub(self.cached_tokens);
        if suffix > suffix_limit {
            return Err(format!(
                "suffix_prefill_tokens={suffix} above limit {suffix_limit}; prompt_tokens={} cached_tokens={}",
                self.prompt_tokens, self.cached_tokens
            ));
        }
        Ok(format!(
            "prompt_tokens={} cached_tokens={} min_cached_tokens={minimum_cached}",
            self.prompt_tokens, self.cached_tokens
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn measured_cache_thresholds_include_the_exact_boundary_and_retain_observations() {
        let metrics = Metrics {
            prompt_tokens: 2304,
            cached_tokens: 2048,
        };
        assert!(metrics.validate(2048, 256).is_ok());
        let failure = metrics.validate(2049, 256).unwrap_err();
        assert!(failure.contains("cached_tokens=2048"));
        assert!(failure.contains("prompt_tokens=2304"));
        assert!(
            metrics
                .validate(2048, 255)
                .unwrap_err()
                .contains("suffix_prefill_tokens=256")
        );
    }

    #[test]
    fn absent_and_malformed_usage_cannot_manufacture_a_positive_cache_count() {
        for value in [
            json!(null),
            json!(true),
            json!(-1),
            json!("2048"),
            json!(2048.5),
        ] {
            let reply = json!({"usage":{"prompt_tokens":value,
                "prompt_tokens_details":{"cached_tokens":value}}});
            let metrics = Metrics::from_response(&reply);
            assert_eq!(
                metrics,
                Metrics {
                    prompt_tokens: 0,
                    cached_tokens: 0
                }
            );
            assert!(metrics.validate(2048, 256).is_err());
        }
        assert_eq!(
            Metrics::from_response(&json!({})),
            Metrics {
                prompt_tokens: 0,
                cached_tokens: 0
            }
        );
        assert_eq!(
            Metrics::from_response(&json!({"usage":{"prompt_tokens":2304,
            "prompt_tokens_details":{"cached_tokens":2048}}})),
            Metrics {
                prompt_tokens: 2304,
                cached_tokens: 2048
            }
        );
    }

    #[test]
    fn zero_thresholds_and_cached_counts_above_prompt_use_saturating_suffix_geometry() {
        assert!(
            Metrics {
                prompt_tokens: 0,
                cached_tokens: 0
            }
            .validate(0, 0)
            .is_ok()
        );
        assert!(
            Metrics {
                prompt_tokens: 2,
                cached_tokens: 3
            }
            .validate(3, 0)
            .is_ok()
        );
    }
}
