//! Backend-neutral evidence for token billing. Estimates are admission hints,
//! never final usage or a ceiling on native/media token counts.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The two inclusive backend counters used for final billing. Reasoning and
/// cached-token breakdowns must not be added to these counters again.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeteredUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl MeteredUsage {
    /// Accept only a complete, nonnegative integer usage report. Missing,
    /// malformed, or partial reports are unknown, not zero-token bills.
    pub fn from_response(response: &Value) -> Option<Self> {
        let usage = response.get("usage")?;
        let prompt_tokens = usage.get("prompt_tokens")?.as_u64()?;
        let completion_tokens = usage.get("completion_tokens")?.as_u64()?;
        prompt_tokens.checked_add(completion_tokens)?;
        Some(Self {
            prompt_tokens,
            completion_tokens,
        })
    }
}

/// Smallest Lightning payment the metered flow sends: 1 sat. Sub-sat
/// payments are not reliably routable (live: 60/90 msat found no route).
pub const PAYMENT_FLOOR_MSAT: u64 = 1_000;

/// Flat deposit asked of a peer outside the provider's trust cache. It only
/// proves the buyer can pay before backend work starts; it is not sized to
/// the request and is applied against the final bill.
pub const DEPOSIT_MSAT: u64 = PAYMENT_FLOOR_MSAT;

/// What the seller invoices after the deposit: `bill - deposit`, waived when
/// it is below [`PAYMENT_FLOOR_MSAT`] (the seller absorbs it).
pub fn final_invoice_msat(bill_msat: u64, deposit_applied_msat: u64) -> u64 {
    let remaining = bill_msat.saturating_sub(deposit_applied_msat);
    if remaining < PAYMENT_FLOOR_MSAT {
        0
    } else {
        remaining
    }
}

impl MeteredUsage {
    /// The bill for this usage at `pricing`: input plus output charges.
    pub fn bill_msat(&self, pricing: &crate::pricing::Pricing) -> Result<u64> {
        pricing
            .input_charge(self.prompt_tokens)?
            .checked_add(pricing.output_charge(self.completion_tokens)?)
            .ok_or_else(|| anyhow::anyhow!("bill overflow"))
    }
}

/// Parsed-content estimate with explicit media uncertainty. `input_tokens`
/// uses UTF-8 bytes / 4 plus modest message overhead; it is not a tokenizer
/// result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptEstimate {
    pub input_tokens: u64,
    #[serde(default)]
    pub has_media: bool,
}

impl PromptEstimate {
    /// Estimate chat content/tool definitions or a text completion prompt.
    /// Do not price the raw HTTP/JSON envelope, model name, output allowance,
    /// request IDs, or payment policy metadata as input text.
    pub fn from_request(body: &Value) -> Result<Self> {
        let mut bytes = 0_u64;
        let mut overhead = 0_u64;
        let mut has_media = false;
        if let Some(messages) = body.get("messages") {
            let messages = messages
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("invalid messages"))?;
            for message in messages {
                ensure!(message.is_object(), "invalid message");
                overhead = overhead
                    .checked_add(4)
                    .ok_or_else(|| anyhow::anyhow!("estimate overflow"))?;
                if let Some(content) = message.get("content") {
                    count_content(content, &mut bytes, &mut has_media)?;
                }
                for key in ["tool_calls", "function_call", "name", "tool_call_id"] {
                    if let Some(value) = message.get(key) {
                        add_json_bytes(value, &mut bytes)?;
                    }
                }
            }
        } else if let Some(prompt) = body.get("prompt") {
            match prompt {
                Value::String(text) => add_bytes(&mut bytes, text.len())?,
                Value::Array(prompts) if prompts.iter().all(Value::is_string) => {
                    for prompt in prompts {
                        add_bytes(&mut bytes, prompt.as_str().expect("checked string").len())?;
                    }
                }
                _ => anyhow::bail!("tokenized prompts require backend counting"),
            }
        } else {
            anyhow::bail!("missing prompt content");
        }
        for key in ["tools", "functions"] {
            if let Some(value) = body.get(key) {
                add_json_bytes(value, &mut bytes)?;
            }
        }
        let input_tokens = bytes
            .div_ceil(4)
            .checked_add(overhead)
            .ok_or_else(|| anyhow::anyhow!("estimate overflow"))?;
        Ok(Self {
            input_tokens,
            has_media,
        })
    }
}

fn add_bytes(total: &mut u64, bytes: usize) -> Result<()> {
    *total = total
        .checked_add(u64::try_from(bytes)?)
        .ok_or_else(|| anyhow::anyhow!("estimate overflow"))?;
    Ok(())
}

fn add_json_bytes(value: &Value, bytes: &mut u64) -> Result<()> {
    add_bytes(bytes, serde_json::to_vec(value)?.len())
}

fn count_content(content: &Value, bytes: &mut u64, has_media: &mut bool) -> Result<()> {
    match content {
        Value::Null => Ok(()),
        Value::String(text) => add_bytes(bytes, text.len()),
        Value::Array(parts) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text" | "input_text") => {
                        let text = part
                            .get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| anyhow::anyhow!("invalid text part"))?;
                        add_bytes(bytes, text.len())?;
                    }
                    _ => *has_media = true,
                }
            }
            Ok(())
        }
        _ => anyhow::bail!("invalid message content"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_estimate_excludes_envelope_and_counts_utf8_and_tools() {
        let mut body = json!({"messages":[{"role":"user","content":"éé"}],"model":"m"});
        assert_eq!(PromptEstimate::from_request(&body).unwrap().input_tokens, 5);
        body["model"] = json!("x".repeat(10000));
        body["mesh_payment"] = json!({"budget":999999});
        assert_eq!(PromptEstimate::from_request(&body).unwrap().input_tokens, 5);
        body["tools"] =
            json!([{"type":"function","function":{"name":"hello","parameters":{"type":"object"}}}]);
        assert!(PromptEstimate::from_request(&body).unwrap().input_tokens > 5);
    }

    #[test]
    fn media_is_uncertain_not_a_byte_token_ceiling() {
        let body = json!({"messages":[{"content":[{"type":"text","text":"abcd"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]});
        assert_eq!(
            PromptEstimate::from_request(&body).unwrap(),
            PromptEstimate {
                input_tokens: 5,
                has_media: true
            }
        );
        assert_eq!(
            PromptEstimate::from_request(&json!({"prompt":"abcd"}))
                .unwrap()
                .input_tokens,
            1
        );
        assert!(PromptEstimate::from_request(&json!({"prompt":[1,2,3]})).is_err());
        assert!(PromptEstimate::from_request(&json!({"messages":"bad"})).is_err());
    }

    #[test]
    fn usage_is_backend_reported_inclusive_and_may_be_zero() {
        assert_eq!(
            MeteredUsage::from_response(
                &json!({"usage":{"prompt_tokens":10,"completion_tokens":20,"completion_tokens_details":{"reasoning_tokens":15},"prompt_tokens_details":{"cached_tokens":5}}})
            ),
            Some(MeteredUsage {
                prompt_tokens: 10,
                completion_tokens: 20
            })
        );
        assert_eq!(
            MeteredUsage::from_response(
                &json!({"usage":{"prompt_tokens":0,"completion_tokens":0}})
            ),
            Some(MeteredUsage {
                prompt_tokens: 0,
                completion_tokens: 0
            })
        );
        for value in [
            json!({}),
            json!({"usage":null}),
            json!({"usage":{"completion_tokens":10}}),
            json!({"usage":{"prompt_tokens":-1,"completion_tokens":10}}),
            json!({"usage":{"prompt_tokens":1.5,"completion_tokens":10}}),
            json!({"usage":{"prompt_tokens":u64::MAX,"completion_tokens":1}}),
        ] {
            assert_eq!(MeteredUsage::from_response(&value), None);
        }
    }
}
