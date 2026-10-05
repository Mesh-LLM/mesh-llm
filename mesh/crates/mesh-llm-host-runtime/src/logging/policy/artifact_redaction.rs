//! Mandatory artifact-body redaction.
//!
//! Valid JSON is handled structurally so key matching is independent of
//! formatting and nesting. Invalid or opaque bytes use the conservative text
//! sanitizer and can never bypass token/path bounds.

use serde_json::Value;

use super::{apply_artifact_redaction, sanitize_json_body, sanitize_paths_in_text};

const SENSITIVE_JSON_KEYS: &[&str] = &[
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "private_key",
    "secret",
    "secret_key",
    "secrets",
    "session_id",
    "token",
];

/// Redact arbitrary artifact bytes with the canonical privacy pipeline.
/// Parsed JSON receives recursive case-insensitive key redaction. Invalid
/// UTF-8 and malformed JSON-shaped input fail closed; only non-JSON text uses
/// the bounded text sanitizer.
pub fn redact_artifact_bytes(content: &[u8]) -> Vec<u8> {
    let Ok(text) = std::str::from_utf8(content) else {
        return b"[REDACTED]".to_vec();
    };
    if let Ok(mut value) = serde_json::from_str::<Value>(text) {
        // Preserve exact client-visible JSON bytes only when the canonical
        // recursive privacy scan proves that it made no redaction change.
        if !redact_json_value(&mut value) {
            return content.to_vec();
        }
        return serde_json::to_vec(&value).unwrap_or_else(|_| b"[REDACTED]".to_vec());
    }
    if text.trim_start().starts_with(['{', '[']) {
        return b"[REDACTED]".to_vec();
    }
    if contains_sensitive_json_key_syntax(text) {
        return b"[REDACTED]".to_vec();
    }
    let sanitized = sanitize_json_body(text);
    let path_safe = sanitize_paths_in_text(&sanitized);
    apply_artifact_redaction(&path_safe).into_bytes()
}

fn contains_sensitive_json_key_syntax(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    SENSITIVE_JSON_KEYS.iter().any(|key| {
        let quoted = format!("\"{key}\"");
        lower
            .match_indices(&quoted)
            .any(|(index, _)| lower[index + quoted.len()..].trim_start().starts_with(':'))
    })
}

/// Redact a parsed JSON value and return whether canonicalization changed it.
fn redact_json_value(value: &mut Value) -> bool {
    match value {
        Value::Array(items) => {
            let mut changed = false;
            for item in items {
                changed |= redact_json_value(item);
            }
            changed
        }
        Value::Object(fields) => {
            let mut changed = false;
            for (key, value) in fields {
                if SENSITIVE_JSON_KEYS
                    .iter()
                    .any(|sensitive| key.eq_ignore_ascii_case(sensitive))
                {
                    let redacted = Value::String("[REDACTED]".to_string());
                    if *value != redacted {
                        *value = redacted;
                        changed = true;
                    }
                } else {
                    changed |= redact_json_value(value);
                }
            }
            changed
        }
        Value::String(text) => {
            let path_safe = sanitize_paths_in_text(text);
            let redacted = apply_artifact_redaction(&path_safe);
            if *text == redacted {
                false
            } else {
                *text = redacted;
                true
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redacted_json(input: &str) -> Value {
        serde_json::from_slice(&redact_artifact_bytes(input.as_bytes())).expect("redacted JSON")
    }

    #[test]
    fn safe_json_preserves_its_exact_original_bytes() {
        let content = br#"{ "type":"invalid_request_error", "message":"safe" }"#;
        assert_eq!(redact_artifact_bytes(content), content);
    }

    #[test]
    fn nested_case_variant_sensitive_keys_are_redacted() {
        let value = redacted_json(
            r#"{"safe":"kept","nested":{"API_KEY":"secret","deeper":[{"Authorization":"Bearer value"}]}}"#,
        );
        assert_eq!(value["safe"], "kept");
        assert_eq!(value["nested"]["API_KEY"], "[REDACTED]");
        assert_eq!(value["nested"]["deeper"][0]["Authorization"], "[REDACTED]");
        assert!(!value.to_string().contains("secret"));
        assert!(!value.to_string().contains("Bearer value"));
    }

    #[test]
    fn sensitive_scalar_array_and_object_values_are_replaced_whole() {
        let value = redacted_json(
            r#"{"password":12,"token":["a",{"x":"b"}],"secret":{"nested":"c"},"ok":{"count":3}}"#,
        );
        assert_eq!(value["password"], "[REDACTED]");
        assert_eq!(value["token"], "[REDACTED]");
        assert_eq!(value["secret"], "[REDACTED]");
        assert_eq!(value["ok"]["count"], 3);
    }

    #[test]
    fn every_array_element_is_scanned_after_an_earlier_redaction() {
        let value = redacted_json(
            r#"[{"api_key":"first-secret"},{"Authorization":"second-secret"},{"safe":"kept"}]"#,
        );
        assert_eq!(value[0]["api_key"], "[REDACTED]");
        assert_eq!(value[1]["Authorization"], "[REDACTED]");
        assert_eq!(value[2]["safe"], "kept");
    }

    #[test]
    fn invite_token_is_explicitly_preserved() {
        let invite = "mesh-llm-invite-token-abc123def456ghi789jkl0";
        let value = redacted_json(&format!(r#"{{"invite_token":"{invite}","model":"safe"}}"#));
        assert_eq!(value["invite_token"], invite);
        assert_eq!(value["model"], "safe");
    }

    #[test]
    fn malformed_and_non_json_content_use_conservative_fallback() {
        let malformed = String::from_utf8(redact_artifact_bytes(
            br#"{"api_key":"secret-value","unterminated": "#,
        ))
        .expect("text fallback");
        assert!(!malformed.contains("secret-value"));

        let case_variant = String::from_utf8(redact_artifact_bytes(
            br#"{"API_KEY":"case-variant-secret","unterminated": "#,
        ))
        .expect("text fallback");
        assert_eq!(case_variant, "[REDACTED]");

        for malformed_escaped_key in [
            br#"{"api\u005fkey":"plain-opaque-value","unterminated": "#.as_slice(),
            br#"{"authoriz\u0061tion":"plain-opaque-value","unterminated": "#.as_slice(),
        ] {
            assert_eq!(
                redact_artifact_bytes(malformed_escaped_key),
                b"[REDACTED]",
                "malformed JSON-shaped input must fail closed even when a sensitive key is escaped"
            );
        }

        let free_form = String::from_utf8(redact_artifact_bytes(
            b"request failed: Bearer free-form-secret",
        ))
        .expect("text fallback");
        assert_eq!(free_form, "[REDACTED]");
    }

    #[test]
    fn invalid_utf8_and_binary_bytes_are_never_preserved() {
        let content = [0xff, 0x00, b's', b'e', b'c', b'r', b'e', b't'];
        assert_eq!(redact_artifact_bytes(&content), b"[REDACTED]");
    }

    #[test]
    fn safe_json_fields_and_arrays_survive() {
        let value = redacted_json(r#"{"id":"chatcmpl-safe","usage":[1,2],"ok":true}"#);
        assert_eq!(value["id"], "chatcmpl-safe");
        assert_eq!(value["usage"], serde_json::json!([1, 2]));
        assert_eq!(value["ok"], true);
    }

    #[test]
    fn bounded_safe_event_stream_is_not_clipped_by_log_presentation_limits() {
        let mut content = String::new();
        for index in 0..24 {
            content.push_str(&format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"chunk-{index}\"}}}}]}}\n\n"
            ));
        }
        content.push_str("data: [DONE]\n\n");
        assert!(content.len() > super::super::MAX_LOG_STRING_LEN);

        assert_eq!(
            redact_artifact_bytes(content.as_bytes()),
            content.as_bytes()
        );
    }
}
