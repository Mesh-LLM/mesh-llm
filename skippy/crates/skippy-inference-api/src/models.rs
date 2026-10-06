use std::{error::Error, fmt};

use serde::Serialize;

use crate::common::now_unix_secs;

/// Opaque OpenAI-facing model identifier.
///
/// The frontend deliberately does not parse this value. Mesh-style ids such as
/// `org/repo:Q4_K_M` should flow through unchanged so the backend or router can
/// resolve them against its own model registry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Result<Self, ModelIdError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ModelIdError);
        }
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelIdError;

impl fmt::Display for ModelIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("model id must not be empty")
    }
}

impl Error for ModelIdError {}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
    /// Render-only observations of the reasoning controls the selected chat
    /// template reacted to, when this model was probed.
    ///
    /// Absent for a model that was never probed: an unprobed control is unknown,
    /// not off. See [`crate::thinking`] for what these observations can claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<crate::thinking::ThinkingControls>,
}

impl ModelObject {
    pub fn new(id: impl Into<String>) -> Self {
        Self::try_new(id).expect("model id must not be empty")
    }

    pub fn try_new(id: impl Into<String>) -> Result<Self, ModelIdError> {
        let id = ModelId::new(id)?;
        Ok(Self {
            id: id.into_string(),
            object: "model",
            created: now_unix_secs(),
            owned_by: "skippy-runtime".to_string(),
            thinking: None,
        })
    }

    /// Attaches render-only reasoning-control observations to this model object.
    pub fn with_thinking(mut self, thinking: Option<crate::thinking::ThinkingControls>) -> Self {
        self.thinking = thinking;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelId, ModelObject};

    #[test]
    fn model_id_accepts_mesh_style_selector_refs_as_opaque_ids() {
        let id = ModelId::new("org/repo:Q4_K_M").unwrap();
        assert_eq!(id.as_str(), "org/repo:Q4_K_M");

        let object = ModelObject::try_new("org/repo:Q4_K_M").unwrap();
        assert_eq!(object.id, "org/repo:Q4_K_M");
    }

    #[test]
    fn model_id_rejects_empty_ids() {
        assert!(ModelId::new("").is_err());
        assert!(ModelId::new("   ").is_err());
        assert!(ModelObject::try_new("").is_err());
    }

    #[test]
    fn a_model_object_omits_thinking_until_the_probe_attaches_it() {
        let bare = serde_json::to_value(ModelObject::new("org/model:Q4_K_M")).unwrap();
        assert!(bare.get("thinking").is_none());

        let controls = crate::thinking::ThinkingControls {
            fingerprint: crate::thinking::ThinkingControlsFingerprint {
                model_id: "org/model:Q4_K_M".to_string(),
                artifact: None,
                template: "embedded".to_string(),
                renderer: "skippy-abi-0.1.66".to_string(),
            },
            evidence: crate::thinking::ThinkingEvidence::RenderedPromptOnly,
            model_obeys_control: crate::thinking::ControlObedience::Unknown,
            untested: "budget semantics".to_string(),
            cases: vec![],
            differences: vec![crate::thinking::ThinkingControlDifference {
                left: "off".to_string(),
                right: crate::thinking::ThinkingControls::BASELINE_CASE.to_string(),
                effect: crate::thinking::ThinkingEffect::ChangesPrompt,
            }],
        };
        let probed = serde_json::to_value(
            ModelObject::new("org/model:Q4_K_M").with_thinking(Some(controls)),
        )
        .unwrap();
        assert_eq!(probed["thinking"]["evidence"], "rendered_prompt_only");
        assert_eq!(probed["thinking"]["model_obeys_control"], "unknown");
        assert_eq!(probed["thinking"]["differences"][0]["left"], "off");
    }
}
