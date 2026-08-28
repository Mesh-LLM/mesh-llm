use anyhow::{Result, bail};
use openai_frontend::ReasoningEffort;
use skippy_server::{
    CONTEXT_BUDGET_MAX_TOKENS, EmbeddedReasoningBudget, EmbeddedReasoningEnabled,
    EmbeddedReasoningFormat,
};

use super::support::string_list_value;
use super::types::ResolvedRequestDefaultsConfig;
use crate::plugin::{
    ModelConfigDefaults, ModelConfigEntry, ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig,
};

pub(super) fn resolve_request_defaults(
    defaults: Option<&ModelConfigDefaults>,
    model_entry: Option<&ModelConfigEntry>,
    request_defaults: Option<&RequestDefaultsConfig>,
) -> Result<ResolvedRequestDefaultsConfig> {
    let model = model_entry.and_then(|entry| entry.request_defaults.as_ref());
    let global = defaults.and_then(|value| value.request_defaults.as_ref());

    reject_unsupported_request_defaults(request_defaults, "request_defaults")?;
    reject_unsupported_request_defaults(model, "models[].request_defaults")?;
    reject_unsupported_request_defaults(global, "defaults.request_defaults")?;

    Ok(ResolvedRequestDefaultsConfig {
        max_tokens: request_defaults
            .and_then(|value| value.max_tokens)
            .or_else(|| model.and_then(|value| value.max_tokens))
            .or_else(|| global.and_then(|value| value.max_tokens))
            .unwrap_or(CONTEXT_BUDGET_MAX_TOKENS),
        temperature: request_defaults
            .and_then(|value| value.temperature)
            .or_else(|| model.and_then(|value| value.temperature))
            .or_else(|| global.and_then(|value| value.temperature)),
        top_p: request_defaults
            .and_then(|value| value.top_p)
            .or_else(|| model.and_then(|value| value.top_p))
            .or_else(|| global.and_then(|value| value.top_p)),
        presence_penalty: request_defaults
            .and_then(|value| value.presence_penalty)
            .or_else(|| model.and_then(|value| value.presence_penalty))
            .or_else(|| global.and_then(|value| value.presence_penalty)),
        frequency_penalty: request_defaults
            .and_then(|value| value.frequency_penalty)
            .or_else(|| model.and_then(|value| value.frequency_penalty))
            .or_else(|| global.and_then(|value| value.frequency_penalty)),
        seed: request_defaults
            .and_then(|value| value.seed)
            .or_else(|| model.and_then(|value| value.seed))
            .or_else(|| global.and_then(|value| value.seed)),
        logit_bias: request_defaults
            .and_then(|value| value.logit_bias.clone())
            .or_else(|| model.and_then(|value| value.logit_bias.clone()))
            .or_else(|| global.and_then(|value| value.logit_bias.clone())),
        top_k: request_defaults
            .and_then(|value| value.top_k)
            .or_else(|| model.and_then(|value| value.top_k))
            .or_else(|| global.and_then(|value| value.top_k)),
        min_p: request_defaults
            .and_then(|value| value.min_p)
            .or_else(|| model.and_then(|value| value.min_p))
            .or_else(|| global.and_then(|value| value.min_p)),
        typical_p: request_defaults
            .and_then(|v| v.typical_p)
            .or_else(|| model.and_then(|v| v.typical_p))
            .or_else(|| global.and_then(|v| v.typical_p)),
        top_nsigma: request_defaults
            .and_then(|v| v.top_nsigma)
            .or_else(|| model.and_then(|v| v.top_nsigma))
            .or_else(|| global.and_then(|v| v.top_nsigma)),
        dynatemp_range: request_defaults
            .and_then(|v| v.dynatemp_range)
            .or_else(|| model.and_then(|v| v.dynatemp_range))
            .or_else(|| global.and_then(|v| v.dynatemp_range)),
        dynatemp_exponent: request_defaults
            .and_then(|v| v.dynatemp_exponent)
            .or_else(|| model.and_then(|v| v.dynatemp_exponent))
            .or_else(|| global.and_then(|v| v.dynatemp_exponent)),
        dry: request_defaults
            .and_then(|v| v.dry.clone())
            .or_else(|| model.and_then(|v| v.dry.clone()))
            .or_else(|| global.and_then(|v| v.dry.clone())),
        xtc: request_defaults
            .and_then(|v| v.xtc.clone())
            .or_else(|| model.and_then(|v| v.xtc.clone()))
            .or_else(|| global.and_then(|v| v.xtc.clone())),
        mirostat_mode: request_defaults
            .and_then(|v| v.mirostat_mode.clone())
            .or_else(|| model.and_then(|v| v.mirostat_mode.clone()))
            .or_else(|| global.and_then(|v| v.mirostat_mode.clone())),
        mirostat_entropy: request_defaults
            .and_then(|v| v.mirostat_entropy)
            .or_else(|| model.and_then(|v| v.mirostat_entropy))
            .or_else(|| global.and_then(|v| v.mirostat_entropy)),
        mirostat_learning_rate: request_defaults
            .and_then(|v| v.mirostat_learning_rate)
            .or_else(|| model.and_then(|v| v.mirostat_learning_rate))
            .or_else(|| global.and_then(|v| v.mirostat_learning_rate)),
        samplers: request_defaults
            .and_then(|v| v.samplers.clone())
            .or_else(|| model.and_then(|v| v.samplers.clone()))
            .or_else(|| global.and_then(|v| v.samplers.clone())),
        sampler_sequence: request_defaults
            .and_then(|v| v.sampler_sequence.clone())
            .or_else(|| model.and_then(|v| v.sampler_sequence.clone()))
            .or_else(|| global.and_then(|v| v.sampler_sequence.clone())),
        ignore_eos: request_defaults
            .and_then(|v| v.ignore_eos)
            .or_else(|| model.and_then(|v| v.ignore_eos))
            .or_else(|| global.and_then(|v| v.ignore_eos)),
        repeat_penalty: request_defaults
            .and_then(|value| value.repeat_penalty)
            .or_else(|| model.and_then(|value| value.repeat_penalty))
            .or_else(|| global.and_then(|value| value.repeat_penalty)),
        repeat_last_n: request_defaults
            .and_then(|value| value.repeat_last_n)
            .or_else(|| model.and_then(|value| value.repeat_last_n))
            .or_else(|| global.and_then(|value| value.repeat_last_n)),
        stop: request_defaults
            .and_then(|value| value.stop.as_ref())
            .or_else(|| model.and_then(|value| value.stop.as_ref()))
            .or_else(|| global.and_then(|value| value.stop.as_ref()))
            .map(string_list_value),
        reasoning_format: request_defaults
            .and_then(|value| value.reasoning_format.clone())
            .or_else(|| model.and_then(|value| value.reasoning_format.clone()))
            .or_else(|| global.and_then(|value| value.reasoning_format.clone())),
        reasoning_enabled: request_defaults
            .and_then(|value| value.reasoning_enabled.clone())
            .or_else(|| model.and_then(|value| value.reasoning_enabled.clone()))
            .or_else(|| global.and_then(|value| value.reasoning_enabled.clone())),
        reasoning_budget: request_defaults
            .and_then(|value| value.reasoning_budget.clone())
            .or_else(|| model.and_then(|value| value.reasoning_budget.clone()))
            .or_else(|| global.and_then(|value| value.reasoning_budget.clone())),
        chat_template: request_defaults
            .and_then(|v| v.chat_template.clone())
            .or_else(|| model.and_then(|v| v.chat_template.clone()))
            .or_else(|| global.and_then(|v| v.chat_template.clone())),
        chat_template_file: request_defaults
            .and_then(|v| v.chat_template_file.clone())
            .or_else(|| model.and_then(|v| v.chat_template_file.clone()))
            .or_else(|| global.and_then(|v| v.chat_template_file.clone())),
        jinja: request_defaults
            .and_then(|v| v.jinja)
            .or_else(|| model.and_then(|v| v.jinja))
            .or_else(|| global.and_then(|v| v.jinja)),
        chat_template_kwargs: request_defaults
            .and_then(|v| v.chat_template_kwargs.clone())
            .or_else(|| model.and_then(|v| v.chat_template_kwargs.clone()))
            .or_else(|| global.and_then(|v| v.chat_template_kwargs.clone())),
        skip_chat_parsing: request_defaults
            .and_then(|v| v.skip_chat_parsing)
            .or_else(|| model.and_then(|v| v.skip_chat_parsing))
            .or_else(|| global.and_then(|v| v.skip_chat_parsing)),
        prefill_assistant: request_defaults
            .and_then(|v| v.prefill_assistant.clone())
            .or_else(|| model.and_then(|v| v.prefill_assistant.clone()))
            .or_else(|| global.and_then(|v| v.prefill_assistant.clone())),
        system_prompt: request_defaults
            .and_then(|v| v.system_prompt.clone())
            .or_else(|| model.and_then(|v| v.system_prompt.clone()))
            .or_else(|| global.and_then(|v| v.system_prompt.clone())),
        grammar: request_defaults
            .and_then(|v| v.grammar.clone())
            .or_else(|| model.and_then(|v| v.grammar.clone()))
            .or_else(|| global.and_then(|v| v.grammar.clone())),
        json_schema: request_defaults
            .and_then(|v| v.json_schema.clone())
            .or_else(|| model.and_then(|v| v.json_schema.clone()))
            .or_else(|| global.and_then(|v| v.json_schema.clone())),
    })
}

pub(super) fn resolve_reasoning_format(value: &str) -> Option<EmbeddedReasoningFormat> {
    match value {
        "auto" => Some(EmbeddedReasoningFormat::Auto),
        "none" => Some(EmbeddedReasoningFormat::None),
        "deepseek" => Some(EmbeddedReasoningFormat::Deepseek),
        "deepseek-legacy" => Some(EmbeddedReasoningFormat::DeepseekLegacy),
        "hidden" => Some(EmbeddedReasoningFormat::Hidden),
        _ => None,
    }
}

pub(super) fn resolve_reasoning_budget(value: &ReasoningBudget) -> Option<EmbeddedReasoningBudget> {
    match value {
        ReasoningBudget::Integer(tokens) => Some(EmbeddedReasoningBudget::Tokens(*tokens)),
        ReasoningBudget::String(value) => match value.as_str() {
            "auto" => Some(EmbeddedReasoningBudget::Auto),
            "low" => Some(EmbeddedReasoningBudget::Effort(ReasoningEffort::Low)),
            "medium" => Some(EmbeddedReasoningBudget::Effort(ReasoningEffort::Medium)),
            "high" => Some(EmbeddedReasoningBudget::Effort(ReasoningEffort::High)),
            _ => None,
        },
    }
}

pub(super) fn resolve_reasoning_enabled(
    value: &ReasoningEnabled,
) -> Option<EmbeddedReasoningEnabled> {
    match value {
        ReasoningEnabled::Bool(true) => Some(EmbeddedReasoningEnabled::Enabled),
        ReasoningEnabled::Bool(false) => Some(EmbeddedReasoningEnabled::Disabled),
        ReasoningEnabled::String(value) => match value.as_str() {
            "auto" => Some(EmbeddedReasoningEnabled::Auto),
            "off" => Some(EmbeddedReasoningEnabled::Disabled),
            "on" => Some(EmbeddedReasoningEnabled::Enabled),
            _ => None,
        },
    }
}

pub(super) fn resolve_request_seed(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| anyhow::anyhow!("request_defaults.seed must be greater than or equal to 0"))
}

pub(super) fn resolve_request_top_k(value: i64) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| anyhow::anyhow!("request_defaults.top_k exceeds supported i32 range"))
}

pub(super) fn resolve_request_repeat_last_n(value: i64) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| anyhow::anyhow!("request_defaults.repeat_last_n exceeds supported i32 range"))
}

pub(super) fn resolve_request_logit_bias(
    value: &toml::Value,
) -> Result<std::collections::BTreeMap<String, serde_json::Value>> {
    let json = serde_json::to_value(value).map_err(|error| {
        anyhow::anyhow!("request_defaults.logit_bias could not be converted to JSON: {error}")
    })?;
    serde_json::from_value::<std::collections::BTreeMap<String, serde_json::Value>>(json).map_err(
        |_| anyhow::anyhow!("request_defaults.logit_bias must be an object keyed by token id"),
    )
}

fn reject_unsupported_request_defaults(
    config: Option<&RequestDefaultsConfig>,
    base_path: &str,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };

    for (field, present) in [
        ("adaptive", config.adaptive.is_some()),
        ("backend_sampling", config.backend_sampling.is_some()),
        ("logprobs", config.logprobs.is_some()),
    ] {
        if present {
            bail!(
                "{base_path}.{field} is accepted by config schema but not supported by the skippy OpenAI frontend/runtime"
            );
        }
    }

    Ok(())
}
