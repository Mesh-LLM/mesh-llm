use serde_json::Value;
use skippy_inference_api::ReasoningEffort;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddedOpenAiRequestDefaults {
    /// Deployment/operator output limit. Package profile limits are resolved
    /// below this field and above the server fallback.
    pub max_tokens: Option<u32>,
    /// Publisher-reviewed profiles carried by model-package v2.
    #[serde(skip)]
    pub package_request_defaults: Option<skippy_package_format::GenerationRequestDefaults>,
    pub stop: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub logit_bias: Option<BTreeMap<String, Value>>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub typical_p: Option<f32>,
    pub top_nsigma: Option<f32>,
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    #[serde(with = "dry")]
    pub dry: Option<skippy_runtime::DrySamplingConfig>,
    #[serde(with = "xtc")]
    pub xtc: Option<skippy_runtime::XtcSamplingConfig>,
    pub mirostat_mode: Option<i32>,
    pub mirostat_entropy: Option<f32>,
    pub mirostat_learning_rate: Option<f32>,
    pub samplers: Option<Vec<String>>,
    pub sampler_sequence: Option<String>,
    pub ignore_eos: Option<bool>,
    pub reasoning_format: Option<EmbeddedReasoningFormat>,
    pub reasoning_enabled: Option<EmbeddedReasoningEnabled>,
    pub reasoning_budget: Option<EmbeddedReasoningBudget>,
    pub chat_template: Option<String>,
    pub jinja: Option<bool>,
    pub chat_template_kwargs: Option<Value>,
    pub skip_chat_parsing: Option<bool>,
    pub prefill_assistant: Option<Value>,
    pub system_prompt: Option<String>,
    pub grammar: Option<Value>,
    pub json_schema: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddedReasoningFormat {
    Auto,
    None,
    Deepseek,
    DeepseekLegacy,
    Hidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddedReasoningEnabled {
    Auto,
    Disabled,
    Enabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddedReasoningBudget {
    Auto,
    Unrestricted,
    Tokens(u32),
    Effort(ReasoningEffort),
}

impl EmbeddedOpenAiRequestDefaults {
    /// Validate operator defaults through the same parsers used for API requests.
    pub fn validate(&self) -> anyhow::Result<()> {
        let mut request: skippy_inference_api::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({"model": "validation", "messages": []}))?;
        crate::frontend::request::apply_chat_request_defaults(&mut request, self)
            .map_err(|error| anyhow::anyhow!("invalid request defaults: {error:?}"))?;
        crate::frontend::request::chat_sampling_config(&request, self)
            .map_err(|error| anyhow::anyhow!("invalid sampling defaults: {error:?}"))?;
        crate::frontend::request::chat_template_options(&request, self)
            .map_err(|error| anyhow::anyhow!("invalid chat defaults: {error:?}"))?;
        anyhow::ensure!(
            self.max_tokens != Some(0),
            "sampling max_tokens must be positive"
        );
        Ok(())
    }
}

mod dry {
    use serde::{Deserialize, Serialize};
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Settings {
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        sequence_breakers: Vec<String>,
    }
    pub fn serialize<S: serde::Serializer>(
        value: &Option<skippy_runtime::DrySamplingConfig>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .as_ref()
            .map(|value| Settings {
                multiplier: value.multiplier,
                base: value.base,
                allowed_length: value.allowed_length,
                penalty_last_n: value.penalty_last_n,
                sequence_breakers: value.sequence_breakers.clone(),
            })
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<skippy_runtime::DrySamplingConfig>, D::Error> {
        Ok(Option::<Settings>::deserialize(deserializer)?.map(
            |Settings {
                 multiplier,
                 base,
                 allowed_length,
                 penalty_last_n,
                 sequence_breakers,
             }| skippy_runtime::DrySamplingConfig {
                multiplier,
                base,
                allowed_length,
                penalty_last_n,
                sequence_breakers,
            },
        ))
    }
}

mod xtc {
    use serde::{Deserialize, Serialize};
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Settings {
        probability: f32,
        threshold: f32,
    }
    pub fn serialize<S: serde::Serializer>(
        value: &Option<skippy_runtime::XtcSamplingConfig>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .as_ref()
            .map(|value| Settings {
                probability: value.probability,
                threshold: value.threshold,
            })
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<skippy_runtime::XtcSamplingConfig>, D::Error> {
        Ok(Option::<Settings>::deserialize(deserializer)?.map(
            |Settings {
                 probability,
                 threshold,
             }| skippy_runtime::XtcSamplingConfig {
                probability,
                threshold,
            },
        ))
    }
}
