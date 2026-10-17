//! The plugin's admission decision (allow or deny an exchange by its model),
//! independent of the wire protocol.
//!
//! Deny a chat-completions request whose `model` field starts with the blocked
//! prefix; allow everything else; fail safe (deny) on any body that cannot be
//! parsed, rather than silently passing it through.

pub const BLOCKED_MODEL_PREFIX: &str = "blocked-";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny { reason: String },
}

pub fn decide(body: &[u8]) -> Decision {
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(e) => return malformed(format!("body is not valid UTF-8: {e}")),
    };

    let parsed: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => return malformed(format!("body is not valid JSON: {e}")),
    };

    let model = match parsed.get("model").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => return malformed("body has no string \"model\" field".to_string()),
    };

    if is_blocked(model) {
        Decision::Deny {
            reason: format!(
                "blocked_model_prefix: {model:?} matches '{BLOCKED_MODEL_PREFIX}*' policy"
            ),
        }
    } else {
        Decision::Allow
    }
}

#[cfg(not(feature = "mutant-blocked-model-served"))]
fn is_blocked(model: &str) -> bool {
    model.starts_with(BLOCKED_MODEL_PREFIX)
}

// mutant-blocked-model-served: the deny condition never fires. A deliberately-broken
// build used only to prove the interop test catches a fail-open regression on
// the exact behavior the model blocklist exists to enforce.
#[cfg(feature = "mutant-blocked-model-served")]
fn is_blocked(_model: &str) -> bool {
    false
}

#[cfg(not(feature = "mutant-malformed-body-served"))]
fn malformed(detail: String) -> Decision {
    Decision::Deny {
        reason: format!("body_parse_failure: {detail}"),
    }
}

// mutant-malformed-body-served: a body the plugin cannot parse is treated as Allow
// instead of Deny. A deliberately-broken build used only to prove the interop
// test catches a fail-open regression on malformed-input handling.
#[cfg(feature = "mutant-malformed-body-served")]
fn malformed(_detail: String) -> Decision {
    Decision::Allow
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_unblocked_model() {
        assert_eq!(decide(br#"{"model":"gpt-neutral-1"}"#), Decision::Allow);
    }

    #[test]
    fn denies_blocked_prefix() {
        match decide(br#"{"model":"blocked-vendor-x"}"#) {
            Decision::Deny { reason } => assert!(reason.contains("blocked_model_prefix")),
            other => panic!("expected deny, got {other:?}"),
        }
    }

    #[test]
    fn fails_safe_on_malformed_json() {
        match decide(b"{not json") {
            Decision::Deny { reason } => assert!(reason.contains("body_parse_failure")),
            other => panic!("expected deny (fail-safe), got {other:?}"),
        }
    }

    #[test]
    fn fails_safe_on_missing_model_field() {
        match decide(br#"{"messages":[]}"#) {
            Decision::Deny { reason } => assert!(reason.contains("body_parse_failure")),
            other => panic!("expected deny (fail-safe), got {other:?}"),
        }
    }
}
