//! Wire shape for render-only reasoning-control observations.
//!
//! These types describe what a chat-template render showed, never what a model
//! did during generation. They ride on the model object so a client can see which
//! reasoning controls the selected template reacted to, and they are deliberately
//! explicit about the strength of that evidence: every report carries
//! [`ThinkingEvidence::RenderedPromptOnly`] and [`ControlObedience::Unknown`].
//!
//! The renderer that produces these lives in `skippy-serving`'s
//! `thinking_probe` module (see its module docs for the full detection limits).

use serde::{Deserialize, Serialize};

/// Identity the observations belong to.
///
/// A change to any field invalidates them: a different artifact, a different
/// selected template, or a different renderer may react differently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ThinkingControlsFingerprint {
    pub model_id: String,
    /// Artifact identity the runtime knows (source hash or package ref), when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    /// `embedded` for the artifact's own template, or `sha256:<hex>` of the
    /// template override the server selected.
    pub template: String,
    /// Native renderer identity, so a template-engine change re-keys the report.
    pub renderer: String,
}

/// How the observation was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEvidence {
    /// A chat template was rendered. No tokens were generated and no forward
    /// pass ran, so nothing here is evidence about generated behaviour.
    RenderedPromptOnly,
}

/// Whether the model follows the control during generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlObedience {
    /// The render probe cannot observe generation, so this is never claimed.
    Unknown,
}

/// What one tested control did to the rendered prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ThinkingRenderOutcome {
    Rendered {
        prompt_sha256: String,
        prompt_bytes: usize,
    },
    Error {
        message: String,
    },
}

/// One control input the probe tried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingControlCase {
    /// Stable name the differences refer to (`omitted`, `off`, `on`, or an effort value).
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    pub outcome: ThinkingRenderOutcome,
}

/// The observed relationship between two renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEffect {
    /// The tested input changed the rendered prompt. This is not evidence that
    /// the model obeyed the control.
    ChangesPrompt,
    /// The rendered prompt was unchanged. Inconclusive: another layer may apply
    /// the control outside the template.
    SamePrompt,
    /// At least one side of the comparison has no rendered prompt to compare.
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingControlDifference {
    pub left: String,
    pub right: String,
    pub effect: ThinkingEffect,
}

/// Render-only observations of the reasoning controls a served model reacted to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingControls {
    pub fingerprint: ThinkingControlsFingerprint,
    pub evidence: ThinkingEvidence,
    pub model_obeys_control: ControlObedience,
    /// What the probe did not exercise, in prose.
    pub untested: String,
    pub cases: Vec<ThinkingControlCase>,
    pub differences: Vec<ThinkingControlDifference>,
}

impl ThinkingControls {
    /// The name of the render with no reasoning control applied at all.
    pub const BASELINE_CASE: &'static str = "omitted";

    /// Yields `(tested control, effect)` for each tested input measured against
    /// the omitted-control render.
    pub fn observations(&self) -> impl Iterator<Item = (&str, ThinkingEffect)> {
        self.differences
            .iter()
            .filter(|difference| difference.right == Self::BASELINE_CASE)
            .map(|difference| (difference.left.as_str(), difference.effect))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> ThinkingControls {
        ThinkingControls {
            fingerprint: ThinkingControlsFingerprint {
                model_id: "org/model".to_string(),
                artifact: Some("abc".to_string()),
                template: "embedded".to_string(),
                renderer: "skippy-abi-0.1.66".to_string(),
            },
            evidence: ThinkingEvidence::RenderedPromptOnly,
            model_obeys_control: ControlObedience::Unknown,
            untested: "budget semantics".to_string(),
            cases: vec![ThinkingControlCase {
                name: "omitted".to_string(),
                enable_thinking: None,
                reasoning_effort: None,
                outcome: ThinkingRenderOutcome::Rendered {
                    prompt_sha256: "00".to_string(),
                    prompt_bytes: 1,
                },
            }],
            differences: vec![
                ThinkingControlDifference {
                    left: "off".to_string(),
                    right: ThinkingControls::BASELINE_CASE.to_string(),
                    effect: ThinkingEffect::ChangesPrompt,
                },
                ThinkingControlDifference {
                    left: "off".to_string(),
                    right: "on".to_string(),
                    effect: ThinkingEffect::ChangesPrompt,
                },
            ],
        }
    }

    #[test]
    fn observations_only_report_against_the_baseline() {
        let report = report();
        let observations = report.observations().collect::<Vec<_>>();
        assert_eq!(observations, vec![("off", ThinkingEffect::ChangesPrompt)]);
    }

    #[test]
    fn the_wire_shape_round_trips_with_the_caveats_attached() {
        let serialized = serde_json::to_value(report()).expect("serialize thinking controls");
        assert_eq!(serialized["evidence"], "rendered_prompt_only");
        assert_eq!(serialized["model_obeys_control"], "unknown");
        let restored: ThinkingControls =
            serde_json::from_value(serialized).expect("deserialize thinking controls");
        assert_eq!(restored, report());
    }
}
