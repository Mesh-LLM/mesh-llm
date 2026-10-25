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

/// Observed effect of the on/off request on the selected template, not whether
/// the model is currently thinking or obeys the request during generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingToggle {
    Observed,
    NotObserved,
    #[default]
    Unknown,
}

/// Reasoning controls a client can offer for one served model.
///
/// Absent from a model object that was never probed: an unprobed control is
/// unknown, not off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ThinkingControls {
    /// Whether both renders succeeded and the on/off control changed the prompt.
    pub toggle: ThinkingToggle,
    /// Whether the native parser recognizes reasoning in the default render.
    /// This does not imply that reasoning can be disabled, or that it will occur.
    /// `None` means the renderer did not supply this evidence.
    pub reasoning_parser: Option<bool>,
    /// Effort values that visibly changed the rendered prompt relative to a
    /// plain thinking-on render, in probe order.
    ///
    /// Empty when the template does not distinguish effort. The probe tries
    /// every accepted effort value (`none`, `minimal`, `low`, `medium`, `high`,
    /// `xhigh`, `max`); values matching plain thinking-on stay unreported.
    pub efforts: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_shape_is_the_client_summary_and_nothing_else() {
        let controls = ThinkingControls {
            toggle: ThinkingToggle::Observed,
            reasoning_parser: Some(true),
            efforts: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
        };
        let serialized = serde_json::to_value(&controls).expect("serialize thinking controls");
        assert_eq!(
            serialized,
            serde_json::json!({ "toggle": "observed", "reasoning_parser": true, "efforts": ["low", "medium", "high"] })
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
            serde_json::json!({ "toggle": "unknown", "reasoning_parser": null, "efforts": [] })
        );
    }
}
