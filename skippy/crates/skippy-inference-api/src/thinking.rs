//! Client-facing summary of the reasoning controls a served model offers.
//!
//! The value is produced by Skippy's load-time, render-only probe of the
//! *selected* chat template (see `skippy_serving::thinking_probe`) and rides on
//! the model object served by `/v1/models`. It is a summary, not raw evidence:
//! the probe's prompt hashes, cases, and fingerprint stay in Skippy's
//! diagnostics and never appear on the wire.
//!
//! What it can say: whether a thinking toggle visibly changed the rendered
//! prompt, and which tested effort values visibly changed it. What it cannot
//! say: whether the model obeys those controls during generation — the probe
//! never runs a forward pass.

use serde::{Deserialize, Serialize};

/// Reasoning controls a client can offer for one served model.
///
/// Absent from a model object that was never probed: an unprobed control is
/// unknown, not off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ThinkingControls {
    /// Whether the selected template reacted to turning thinking on versus off,
    /// so a client can offer a thinking toggle.
    ///
    /// `false` also covers a probe that could not render one side of the
    /// comparison: a control this probe could not observe is not advertised.
    pub enabled: bool,
    /// Effort values that visibly changed the rendered prompt relative to a
    /// plain thinking-on render, in probe order.
    ///
    /// Empty when the template does not distinguish effort. Only `low`,
    /// `medium`, `high`, and `xhigh` are probed; other accepted values stay
    /// unreported.
    pub efforts: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_shape_is_the_client_summary_and_nothing_else() {
        let controls = ThinkingControls {
            enabled: true,
            efforts: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
        };
        let serialized = serde_json::to_value(&controls).expect("serialize thinking controls");
        assert_eq!(
            serialized,
            serde_json::json!({ "enabled": true, "efforts": ["low", "medium", "high"] })
        );
        let restored: ThinkingControls =
            serde_json::from_value(serialized).expect("deserialize thinking controls");
        assert_eq!(restored, controls);
    }

    #[test]
    fn a_model_without_controls_still_reports_the_summary_shape() {
        let serialized =
            serde_json::to_value(ThinkingControls::default()).expect("serialize thinking controls");
        assert_eq!(
            serialized,
            serde_json::json!({ "enabled": false, "efforts": [] })
        );
    }
}
