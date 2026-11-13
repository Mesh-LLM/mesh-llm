//! Provider-profile padding for eligible Open Responses delta events.
//!
//! This is wire padding only. It does not redact or encrypt model output and
//! does not claim to prevent traffic analysis. The profile deliberately uses
//! coarse 256-byte compact-JSON body buckets; the Open Responses schema does
//! not mandate that algorithm.

use chacha20poly1305::aead::Generate;
use serde_json::{Map, Value};

use crate::errors::InferenceError;

const PROFILE_BLOCK_BYTES: usize = 256;
const URL_SAFE_BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

#[cfg(test)]
pub(super) type RandomSource = fn() -> Result<[u8; PROFILE_BLOCK_BYTES], ObfuscationError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::responses) struct ObfuscationPolicy {
    enabled: bool,
}

impl ObfuscationPolicy {
    pub(in crate::responses) const fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Capture and remove the Responses-only transport control before the
    /// request is translated into a Chat Completions backend body.
    pub(in crate::responses) fn capture_and_strip(
        request: &mut Map<String, Value>,
    ) -> Result<Self, InferenceError> {
        let Some(stream_options) = request.get_mut("stream_options") else {
            return Ok(Self::enabled());
        };
        if stream_options.is_null() {
            return Ok(Self::enabled());
        }
        let options = stream_options.as_object_mut().ok_or_else(|| {
            InferenceError::invalid_request("stream_options must be an object or null")
                .with_param("stream_options")
        })?;
        let enabled = match options.remove("include_obfuscation") {
            None => true,
            Some(Value::Bool(enabled)) => enabled,
            Some(_) => {
                return Err(InferenceError::invalid_request(
                    "stream_options.include_obfuscation must be true or false",
                )
                .with_param("stream_options.include_obfuscation"));
            }
        };
        Ok(Self { enabled })
    }

    pub(super) fn apply(self, event: &str, data: &mut Value) -> Result<(), ObfuscationError> {
        if !self.enabled || !eligible(event) {
            return Ok(());
        }
        apply_with(event, data, || {
            <[u8; PROFILE_BLOCK_BYTES]>::try_generate().map_err(|_| ObfuscationError)
        })
    }

    #[cfg(test)]
    pub(super) fn apply_with_random_source(
        self,
        event: &str,
        data: &mut Value,
        random: RandomSource,
    ) -> Result<(), ObfuscationError> {
        if !self.enabled || !eligible(event) {
            return Ok(());
        }
        apply_with(event, data, random)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ObfuscationError;

fn eligible(event: &str) -> bool {
    matches!(
        event,
        "response.output_text.delta"
            | "response.reasoning.delta"
            | "response.reasoning_summary_text.delta"
            | "response.function_call_arguments.delta"
    )
}

fn apply_with(
    event: &str,
    data: &mut Value,
    random: impl FnOnce() -> Result<[u8; PROFILE_BLOCK_BYTES], ObfuscationError>,
) -> Result<(), ObfuscationError> {
    debug_assert!(eligible(event));
    let previous = data
        .as_object_mut()
        .ok_or(ObfuscationError)?
        .insert("obfuscation".into(), Value::String(String::new()));
    let placeholder_bytes = match serde_json::to_vec(data) {
        Ok(bytes) => bytes.len(),
        Err(_) => {
            restore_previous(data, previous);
            return Err(ObfuscationError);
        }
    };
    let padding_len = PROFILE_BLOCK_BYTES - (placeholder_bytes % PROFILE_BLOCK_BYTES);
    let random = match random() {
        Ok(random) => random,
        Err(error) => {
            restore_previous(data, previous);
            return Err(error);
        }
    };
    let padding = random[..padding_len]
        .iter()
        .map(|byte| URL_SAFE_BASE64_ALPHABET[usize::from(*byte) % 64] as char)
        .collect();
    data.as_object_mut()
        .ok_or(ObfuscationError)?
        .insert("obfuscation".into(), Value::String(padding));
    Ok(())
}

fn restore_previous(data: &mut Value, previous: Option<Value>) {
    let Some(object) = data.as_object_mut() else {
        return;
    };
    match previous {
        Some(previous) => {
            object.insert("obfuscation".into(), previous);
        }
        None => {
            object.remove("obfuscation");
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn padded(event: &str, delta: &str) -> Value {
        let mut data = json!({
            "type": event,
            "sequence_number": 17,
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
        });
        apply_with(event, &mut data, || Ok([255; PROFILE_BLOCK_BYTES])).unwrap();
        data
    }

    #[test]
    fn eligible_events_use_the_coarse_json_body_profile() {
        for event in [
            "response.output_text.delta",
            "response.reasoning.delta",
            "response.reasoning_summary_text.delta",
            "response.function_call_arguments.delta",
        ] {
            let data = padded(event, "semantic payload");
            let padding = data["obfuscation"].as_str().unwrap();
            assert!((1..=PROFILE_BLOCK_BYTES).contains(&padding.len()));
            assert!(
                padding
                    .bytes()
                    .all(|byte| URL_SAFE_BASE64_ALPHABET.contains(&byte))
            );
            assert_eq!(
                serde_json::to_vec(&data).unwrap().len() % PROFILE_BLOCK_BYTES,
                0
            );
            assert_eq!(data["delta"], "semantic payload");
        }
    }

    #[test]
    fn an_aligned_placeholder_receives_a_full_random_block() {
        let data = (0..PROFILE_BLOCK_BYTES)
            .map(|length| padded("response.output_text.delta", &"x".repeat(length)))
            .find(|data| data["obfuscation"].as_str().unwrap().len() == PROFILE_BLOCK_BYTES)
            .expect("one delta length aligns the placeholder body");
        assert_eq!(
            data["obfuscation"].as_str().unwrap().len(),
            PROFILE_BLOCK_BYTES
        );
    }

    #[test]
    fn ineligible_or_disabled_events_are_unchanged() {
        let original = json!({"type":"response.output_text.done","delta":"x"});
        let mut ineligible = original.clone();
        ObfuscationPolicy::enabled()
            .apply("response.output_text.done", &mut ineligible)
            .unwrap();
        assert_eq!(ineligible, original);

        let mut request = Map::from_iter([(
            "stream_options".into(),
            json!({"include_obfuscation":false}),
        )]);
        let disabled = ObfuscationPolicy::capture_and_strip(&mut request).unwrap();
        let mut eligible = json!({"delta":"x"});
        disabled
            .apply("response.output_text.delta", &mut eligible)
            .unwrap();
        assert_eq!(eligible, json!({"delta":"x"}));
    }

    #[test]
    fn random_source_failures_are_returned_without_partial_padding() {
        let mut data = json!({"delta":"x"});
        assert_eq!(
            apply_with("response.output_text.delta", &mut data, || Err(
                ObfuscationError
            )),
            Err(ObfuscationError)
        );
        assert!(data.get("obfuscation").is_none());
    }
}
