//! Mesh-scoped model discovery, independent of compatibility API projections.

use crate::mesh::{ModelRuntimeDescriptor, ModelWorkloadClass, ServedModelMetadata};
use crate::models::ModelCapabilities;
use serde::{Deserialize, Serialize};

/// A model's role in discovery. Modality does not determine whether it is an LLM.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshModelKind {
    Llm,
    #[default]
    Model,
    Virtual,
    #[serde(other)]
    Unknown,
}

/// Current observed availability; discovery is not a reservation for inference.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshModelAvailability {
    Available,
    Unavailable,
    #[default]
    #[serde(other)]
    Unknown,
}

/// Filters over advertisements in one mesh. All supplied filters combine with AND.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeshModelQuery {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_workload_filter"
    )]
    pub workload_class: Option<ModelWorkloadClass>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<MeshModelKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

fn deserialize_workload_filter<'de, D>(
    deserializer: D,
) -> Result<Option<ModelWorkloadClass>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    value
        .map(|value| match value.as_str() {
            "causal_generation" => Ok(ModelWorkloadClass::CausalGeneration),
            "embedding" => Ok(ModelWorkloadClass::Embedding),
            "rerank" => Ok(ModelWorkloadClass::Rerank),
            "encoder_decoder" => Ok(ModelWorkloadClass::EncoderDecoder),
            "speech_synthesis" => Ok(ModelWorkloadClass::SpeechSynthesis),
            "decision" => Ok(ModelWorkloadClass::Decision),
            "speech_recognition" => Ok(ModelWorkloadClass::SpeechRecognition),
            "document_ocr" => Ok(ModelWorkloadClass::DocumentOcr),
            "unknown" => Ok(ModelWorkloadClass::Unknown),
            _ => Err(serde::de::Error::custom("invalid workload_class filter")),
        })
        .transpose()
}

/// A bounded discovery result. Pass `next_cursor` unchanged to continue it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MeshModelPage {
    pub data: Vec<MeshModel>,
    pub next_cursor: Option<String>,
}

/// One public serving identity, retaining distinct observations from its providers.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MeshModel {
    pub id: String,
    pub display_name: String,
    pub kind: MeshModelKind,
    /// Absent when providers expose differing workload classes or no class is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload_class: Option<ModelWorkloadClass>,
    pub availability: MeshModelAvailability,
    /// Descriptive union: each operation is supported by at least one provider.
    pub operations: Vec<String>,
    /// Static facts common to all providers. Provider differences remain below.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ServedModelMetadata>,
    pub providers: Vec<MeshModelProvider>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub virtual_model: Option<serde_json::Value>,
}

impl MeshModel {
    /// Positive advertised LLM evidence, independent of legacy inference admission.
    pub fn is_llm(&self) -> bool {
        self.kind == MeshModelKind::Llm
            || (self.kind == MeshModelKind::Virtual
                && self.operations.iter().any(|op| op == "chat"))
    }

    /// Whether a provider can currently serve the LLM contract. An available
    /// OCR replica sharing the ID cannot make an unavailable LLM eligible.
    pub fn is_available_llm(&self) -> bool {
        self.is_llm()
            && self.providers.iter().any(|provider| {
                provider.available
                    && provider.workload_class == Some(ModelWorkloadClass::CausalGeneration)
            })
    }
}

/// Facts observed together for one provider and, for plugins, one endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MeshModelProvider {
    pub provider_id: String,
    pub local: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload_class: Option<ModelWorkloadClass>,
    pub available: bool,
    pub operations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<ModelCapabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ServedModelMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<ModelRuntimeDescriptor>,
    /// Controls probed on this provider; absence means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<serde_json::Value>,
    /// Descriptive advertised economics, never a binding quote or authorization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payment: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_rejects_unknown_filter_names_but_advertisements_remain_forward_compatible() {
        assert!(serde_json::from_str::<MeshModelQuery>(r#"{"workload_class":"typo"}"#).is_err());
        assert_eq!(
            serde_json::from_str::<MeshModelQuery>(r#"{"workload_class":"unknown"}"#)
                .unwrap()
                .workload_class,
            Some(ModelWorkloadClass::Unknown),
        );
        assert_eq!(
            serde_json::from_str::<ModelWorkloadClass>(r#""future_workload""#).unwrap(),
            ModelWorkloadClass::Unknown,
        );
    }
}
