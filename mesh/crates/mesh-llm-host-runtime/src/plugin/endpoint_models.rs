//! Preserve bounded, per-model output declarations from endpoint discovery.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn parse_models(
    body: &Value,
    protocol: Option<&str>,
) -> (Vec<String>, BTreeMap<String, Vec<String>>) {
    let mut models = Vec::new();
    let mut declarations = BTreeMap::new();
    for entry in body
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| model_matches_protocol(entry, protocol))
    {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        let outputs = parse_output_modalities(entry.get("output_modalities"));
        models.push(id.to_string());
        declarations
            .entry(id.to_string())
            .and_modify(|previous| {
                // A duplicate model ID with conflicting declarations is unknown.
                if previous != &outputs {
                    *previous = Vec::new();
                }
            })
            .or_insert(outputs);
    }
    declarations.retain(|_, outputs| !outputs.is_empty());
    (models, declarations)
}

fn parse_output_modalities(value: Option<&Value>) -> Vec<String> {
    let Some(values) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    // Preserve future modality names without letting an endpoint grow each
    // model's gossip declaration without bound. Invalid declarations are unknown.
    if values.len() > 8 {
        return Vec::new();
    }
    let mut outputs = BTreeSet::new();
    for value in values {
        let Some(token) = value.as_str() else {
            return Vec::new();
        };
        if token.is_empty()
            || token.len() > 32
            || !token.as_bytes()[0].is_ascii_lowercase()
            || !token.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            })
        {
            return Vec::new();
        }
        outputs.insert(token.to_string());
    }
    outputs.into_iter().collect()
}

pub(super) fn model_matches_protocol(entry: &Value, protocol: Option<&str>) -> bool {
    if entry
        .get("availability")
        .and_then(Value::as_str)
        .is_some_and(|state| !matches!(state, "ready" | "loadable" | "busy"))
    {
        return false;
    }
    let expected = match protocol {
        Some("document_ocr_v1") => "document_ocr",
        Some("transcription_v1") => "speech_recognition",
        Some("image_text_similarity_v1") => "image_text_similarity",
        Some("depth_estimation_v1") => "depth_estimation",
        _ => {
            return entry
                .get("workload_class")
                .and_then(Value::as_str)
                .is_none_or(|class| class == "causal_generation" || class == "encoder_decoder");
        }
    };
    // Single-task legacy OCR endpoints omit class; modern multi-task providers
    // declare every entry, so none can leak onto another endpoint.
    entry
        .get("workload_class")
        .and_then(Value::as_str)
        .map_or(protocol == Some("document_ocr_v1"), |class| {
            class == expected
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn output_modalities_stay_with_their_declaring_model() {
        let (models, outputs) = parse_models(
            &json!({"data": [
                {"id": "legacy"},
                {"id": "text", "output_modalities": ["text"]},
                {"id": "image", "output_modalities": ["text", "image", "image"]},
                {"id": "other-task", "workload_class": "document_ocr", "output_modalities": ["image"]},
                {"id": "unavailable", "availability": "unavailable", "output_modalities": ["image"]}
            ]}),
            Some("openai_compatible"),
        );
        assert_eq!(models, ["legacy", "text", "image"]);
        assert_eq!(outputs["image"], ["image", "text"]);
        assert_eq!(outputs["text"], ["text"]);
        assert!(!outputs.contains_key("legacy"));
        assert!(!outputs.contains_key("other-task"));
        assert!(!outputs.contains_key("unavailable"));
    }

    #[test]
    fn declarations_are_bounded_and_future_tokens_survive() {
        assert_eq!(
            parse_output_modalities(Some(&json!(["future_output", "text"]))),
            ["future_output", "text"]
        );
        for value in [
            json!("image"),
            json!(["text", 1]),
            json!(["Image"]),
            json!([""]),
            json!(["x".repeat(33)]),
            json!(vec!["text"; 9]),
        ] {
            assert!(parse_output_modalities(Some(&value)).is_empty());
        }
    }

    #[test]
    fn conflicting_duplicate_ids_do_not_claim_outputs() {
        let (_, outputs) = parse_models(
            &json!({"data": [
                {"id": "duplicate", "output_modalities": ["text", "image"]},
                {"id": "duplicate", "output_modalities": ["text"]},
                {"id": "duplicate", "output_modalities": ["text", "image"]}
            ]}),
            None,
        );
        assert!(!outputs.contains_key("duplicate"));
    }
}
