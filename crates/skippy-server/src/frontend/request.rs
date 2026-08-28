use crate::frontend::EmbeddedOpenAiRequestDefaults;
use crate::frontend::EmbeddedReasoningBudget;
use crate::frontend::EmbeddedReasoningEnabled;
use crate::frontend::EmbeddedReasoningFormat;
use base64::Engine;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::ChatMessage;
use openai_frontend::CompletionRequest;
use openai_frontend::MessageContent;
use openai_frontend::MessageContentPart;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::Value;
use skippy_protocol::binary::MAX_STAGE_DRY_SEQUENCE_BREAKERS;
use skippy_protocol::binary::MAX_STAGE_LOGIT_BIAS;
use skippy_protocol::binary::MAX_STAGE_SAMPLERS;
use skippy_protocol::binary::StageLogitBias as WireLogitBias;
use skippy_protocol::binary::StageSamplingConfig as WireSamplingConfig;
use skippy_runtime::ChatReasoningFormat;
use skippy_runtime::ChatTemplateOptions;
use skippy_runtime::DrySamplingConfig;
use skippy_runtime::LogitBias as RuntimeLogitBias;
use skippy_runtime::MAX_DRY_SEQUENCE_BREAKER_BYTES;
use skippy_runtime::MAX_LOGIT_BIAS;
use skippy_runtime::MediaInput;
use skippy_runtime::SamplingConfig;
use skippy_runtime::XtcSamplingConfig;

const MAX_NATIVE_PARSER_INPUT_BYTES: usize = 1024 * 1024;

struct SharedRequestFields<'a> {
    presence_penalty: &'a mut Option<f32>,
    frequency_penalty: &'a mut Option<f32>,
    seed: &'a mut Option<u64>,
    logit_bias: &'a mut Option<std::collections::BTreeMap<String, serde_json::Value>>,
    temperature: &'a mut Option<f32>,
    top_p: &'a mut Option<f32>,
    stop: &'a mut Option<openai_frontend::StopSequence>,
    extra: &'a mut std::collections::BTreeMap<String, serde_json::Value>,
}

pub(super) fn apply_chat_request_defaults(
    request: &mut ChatCompletionRequest,
    defaults: &EmbeddedOpenAiRequestDefaults,
) -> OpenAiResult<()> {
    apply_shared_request_defaults(
        SharedRequestFields {
            presence_penalty: &mut request.presence_penalty,
            frequency_penalty: &mut request.frequency_penalty,
            seed: &mut request.seed,
            logit_bias: &mut request.logit_bias,
            temperature: &mut request.temperature,
            top_p: &mut request.top_p,
            stop: &mut request.stop,
            extra: &mut request.extra,
        },
        defaults,
    );
    apply_chat_only_request_defaults(request, defaults)
}

fn apply_chat_only_request_defaults(
    request: &mut ChatCompletionRequest,
    defaults: &EmbeddedOpenAiRequestDefaults,
) -> OpenAiResult<()> {
    for (name, value) in [
        (
            "chat_template",
            defaults.chat_template.clone().map(Value::from),
        ),
        ("jinja", defaults.jinja.map(Value::from)),
        (
            "chat_template_kwargs",
            defaults.chat_template_kwargs.clone(),
        ),
        (
            "skip_chat_parsing",
            defaults.skip_chat_parsing.map(Value::from),
        ),
        ("prefill_assistant", defaults.prefill_assistant.clone()),
        (
            "system_prompt",
            defaults.system_prompt.clone().map(Value::from),
        ),
        ("grammar", defaults.grammar.clone()),
        ("json_schema", defaults.json_schema.clone()),
        (
            "reasoning_format",
            defaults.reasoning_format.map(|value| {
                Value::from(match value {
                    EmbeddedReasoningFormat::Auto => "auto",
                    EmbeddedReasoningFormat::None => "none",
                    EmbeddedReasoningFormat::Deepseek => "deepseek",
                    EmbeddedReasoningFormat::DeepseekLegacy => "deepseek-legacy",
                    EmbeddedReasoningFormat::Hidden => "hidden",
                })
            }),
        ),
    ] {
        if extra_value_is_omitted(&request.extra, name)
            && let Some(value) = value
        {
            request.extra.insert(name.to_string(), value);
        }
    }
    if let Some(system_prompt) = optional_string_extra(&request.extra, "system_prompt")?
        && !request
            .messages
            .iter()
            .any(|message| message.role == "system")
    {
        request.messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: Some(openai_frontend::MessageContent::Text(system_prompt)),
                extra: std::collections::BTreeMap::new(),
            },
        );
    }
    if let Some(prefill) = request
        .extra
        .get("prefill_assistant")
        .filter(|value| !value.is_null())
    {
        request.messages.push(prefill_assistant_message(prefill)?);
    }
    Ok(())
}

fn prefill_assistant_message(value: &Value) -> OpenAiResult<ChatMessage> {
    if let Some(content) = value.as_str() {
        return Ok(ChatMessage {
            role: "assistant".to_string(),
            content: Some(openai_frontend::MessageContent::Text(content.to_string())),
            extra: std::collections::BTreeMap::new(),
        });
    }
    let message = serde_json::from_value::<ChatMessage>(value.clone()).map_err(|_| {
        OpenAiError::invalid_request("prefill_assistant must be a string or chat message object")
    })?;
    if message.role != "assistant" {
        return Err(OpenAiError::invalid_request(
            "prefill_assistant message role must be assistant",
        ));
    }
    Ok(message)
}

pub(super) fn apply_completion_request_defaults(
    request: &mut CompletionRequest,
    defaults: &EmbeddedOpenAiRequestDefaults,
) {
    apply_shared_request_defaults(
        SharedRequestFields {
            presence_penalty: &mut request.presence_penalty,
            frequency_penalty: &mut request.frequency_penalty,
            seed: &mut request.seed,
            logit_bias: &mut request.logit_bias,
            temperature: &mut request.temperature,
            top_p: &mut request.top_p,
            stop: &mut request.stop,
            extra: &mut request.extra,
        },
        defaults,
    );
}

pub(super) fn message_content_to_generation_text(
    content: &MessageContent,
    marker: &str,
    media: &mut Vec<MediaInput>,
) -> OpenAiResult<String> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut chunks = Vec::new();
            for part in parts {
                if part.content_type == "text" {
                    if let Some(text) = part.text.as_deref() {
                        chunks.push(text.to_string());
                    }
                    continue;
                }
                if let Some(bytes) = media_bytes_from_part(part)? {
                    media.push(MediaInput { bytes });
                    chunks.push(marker.to_string());
                }
            }
            Ok(chunks.join("\n"))
        }
        MessageContent::Other(_) => Ok(String::new()),
    }
}

pub(super) fn media_bytes_from_part(part: &MessageContentPart) -> OpenAiResult<Option<Vec<u8>>> {
    let is_media = matches!(
        part.content_type.as_str(),
        "image_url" | "input_image" | "image" | "input_audio" | "audio" | "audio_url"
    );
    if !is_media {
        return Ok(None);
    }
    if let Some(url) = media_url(part) {
        return decode_media_url(&url).map(Some);
    }
    if let Some(data) = media_data(part) {
        return decode_base64_payload(&data).map(Some);
    }
    Err(OpenAiError::invalid_request(format!(
        "media content block '{}' is missing url or data",
        part.content_type
    )))
}

pub(super) fn media_url(part: &MessageContentPart) -> Option<String> {
    part.media_url()
}

pub(super) fn media_data(part: &MessageContentPart) -> Option<String> {
    part.media_data()
}

pub(super) fn decode_media_url(url: &str) -> OpenAiResult<Vec<u8>> {
    match url.split_once(',') {
        Some((prefix, payload)) if prefix.starts_with("data:") && prefix.contains(";base64") => {
            return decode_base64_payload(payload);
        }
        _ => {}
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(OpenAiError::unsupported(
            "remote multimodal URLs must be fetched by mesh before reaching skippy",
        ));
    }
    decode_base64_payload(url)
}

pub(super) fn decode_base64_payload(payload: &str) -> OpenAiResult<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload.as_bytes()))
        .map_err(|error| OpenAiError::invalid_request(format!("invalid media base64: {error}")))
}

fn apply_shared_request_defaults(
    fields: SharedRequestFields<'_>,
    defaults: &EmbeddedOpenAiRequestDefaults,
) {
    let SharedRequestFields {
        presence_penalty,
        frequency_penalty,
        seed,
        logit_bias,
        temperature,
        top_p,
        stop,
        extra,
    } = fields;
    if presence_penalty.is_none() {
        *presence_penalty = defaults.presence_penalty;
    }
    if frequency_penalty.is_none() {
        *frequency_penalty = defaults.frequency_penalty;
    }
    if seed.is_none() {
        *seed = defaults.seed;
    }
    if logit_bias.is_none() {
        *logit_bias = defaults.logit_bias.clone();
    }
    if temperature.is_none() {
        *temperature = defaults.temperature;
    }
    if top_p.is_none() {
        *top_p = defaults.top_p;
    }
    if stop.is_none() {
        *stop = defaults
            .stop
            .as_ref()
            .map(|values| openai_frontend::StopSequence::from_values(values.clone()));
    }
    if let (true, Some(value)) = (extra_value_is_omitted(extra, "top_k"), defaults.top_k) {
        extra.insert("top_k".to_string(), serde_json::json!(value));
    }
    if let (true, Some(value)) = (extra_value_is_omitted(extra, "min_p"), defaults.min_p) {
        extra.insert("min_p".to_string(), serde_json::json!(value));
    }
    if let (true, Some(value)) = (
        extra_value_is_omitted(extra, "repeat_penalty")
            && extra_value_is_omitted(extra, "repetition_penalty"),
        defaults.repeat_penalty,
    ) {
        extra.insert("repeat_penalty".to_string(), serde_json::json!(value));
    }
    if let (true, Some(value)) = (
        extra_value_is_omitted(extra, "repeat_last_n"),
        defaults.repeat_last_n,
    ) {
        extra.insert("repeat_last_n".to_string(), serde_json::json!(value));
    }
    for (name, value) in [
        ("typical_p", defaults.typical_p.map(Value::from)),
        ("top_nsigma", defaults.top_nsigma.map(Value::from)),
        ("dynatemp_range", defaults.dynatemp_range.map(Value::from)),
        (
            "dynatemp_exponent",
            defaults.dynatemp_exponent.map(Value::from),
        ),
        ("mirostat_mode", defaults.mirostat_mode.map(Value::from)),
        (
            "mirostat_entropy",
            defaults.mirostat_entropy.map(Value::from),
        ),
        (
            "mirostat_learning_rate",
            defaults.mirostat_learning_rate.map(Value::from),
        ),
        ("ignore_eos", defaults.ignore_eos.map(Value::from)),
    ] {
        if extra_value_is_omitted(extra, name)
            && let Some(value) = value
        {
            extra.insert(name.to_string(), value);
        }
    }
    if extra_value_is_omitted(extra, "dry")
        && let Some(dry) = defaults.dry.as_ref()
    {
        extra.insert(
            "dry".to_string(),
            serde_json::json!({
                "multiplier": dry.multiplier,
                "base": dry.base,
                "allowed_length": dry.allowed_length,
                "penalty_last_n": dry.penalty_last_n,
                "sequence_breakers": dry.sequence_breakers,
            }),
        );
    }
    if extra_value_is_omitted(extra, "xtc")
        && let Some(xtc) = defaults.xtc.as_ref()
    {
        extra.insert(
            "xtc".to_string(),
            serde_json::json!({"probability": xtc.probability, "threshold": xtc.threshold}),
        );
    }
    if extra_value_is_omitted(extra, "samplers")
        && let Some(samplers) = defaults.samplers.as_ref()
    {
        extra.insert("samplers".to_string(), serde_json::json!(samplers));
    }
    if extra_value_is_omitted(extra, "sampler_sequence")
        && let Some(sequence) = defaults.sampler_sequence.as_ref()
    {
        extra.insert(
            "sampler_sequence".to_string(),
            Value::from(sequence.clone()),
        );
    }
}

fn extra_value_is_omitted(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
    field: &str,
) -> bool {
    extra.get(field).is_none_or(Value::is_null)
}

pub(super) fn chat_sampling_config(
    request: &ChatCompletionRequest,
) -> OpenAiResult<SamplingConfig> {
    sampling_config(
        request.temperature,
        request.top_p,
        request.presence_penalty,
        request.frequency_penalty,
        request.seed,
        request.logit_bias.as_ref(),
        &request.extra,
    )
}

pub(super) fn completion_sampling_config(
    request: &CompletionRequest,
) -> OpenAiResult<SamplingConfig> {
    sampling_config(
        request.temperature,
        request.top_p,
        request.presence_penalty,
        request.frequency_penalty,
        request.seed,
        request.logit_bias.as_ref(),
        &request.extra,
    )
}

pub(super) fn chat_template_options(
    request: &ChatCompletionRequest,
    defaults: &EmbeddedOpenAiRequestDefaults,
) -> OpenAiResult<ChatTemplateOptions> {
    let reasoning = openai_frontend::normalize_reasoning_template_options(
        request.reasoning.as_ref(),
        request.reasoning_effort,
        &request.extra,
    )?;
    Ok(ChatTemplateOptions {
        add_assistant: request
            .extra
            .get("prefill_assistant")
            .is_none_or(Value::is_null),
        reasoning_format: Some(request_reasoning_format(request, defaults)?),
        enable_thinking: reasoning
            .enable_thinking
            .or_else(|| default_reasoning_enabled(defaults.reasoning_enabled))
            .or_else(|| default_reasoning_budget_enabled(defaults.reasoning_budget)),
        chat_template_kwargs: merged_chat_template_kwargs(
            defaults,
            &reasoning.chat_template_kwargs,
        )?
        .map(|kwargs| serialize_bounded_native_parser_json("chat_template_kwargs", &kwargs))
        .transpose()
        .map_err(|error| OpenAiError::invalid_request(error.to_string()))?,
        chat_template: bounded_optional_string_extra(&request.extra, "chat_template")?,
        use_jinja: optional_bool_extra(&request.extra, "jinja")?.unwrap_or(true),
        grammar: structured_output_string(request, "grammar")?,
        json_schema: structured_output_json(request, "json_schema")?,
        skip_chat_parsing: optional_bool_extra(&request.extra, "skip_chat_parsing")?
            .unwrap_or(false),
    })
}

fn merged_chat_template_kwargs(
    defaults: &EmbeddedOpenAiRequestDefaults,
    request: &std::collections::BTreeMap<String, Value>,
) -> OpenAiResult<Option<std::collections::BTreeMap<String, Value>>> {
    let mut merged = defaults
        .chat_template_kwargs
        .as_ref()
        .map(|value| {
            value.as_object().cloned().ok_or_else(|| {
                OpenAiError::invalid_request("chat_template_kwargs must be an object")
            })
        })
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    if let Some(budget) = defaults.reasoning_budget {
        match budget {
            EmbeddedReasoningBudget::Tokens(tokens) => {
                merged.insert("reasoning_budget".to_string(), Value::from(tokens));
                merged.insert("thinking_budget".to_string(), Value::from(tokens));
            }
            EmbeddedReasoningBudget::Effort(effort) => {
                merged.insert(
                    "reasoning_effort".to_string(),
                    Value::from(match effort {
                        openai_frontend::ReasoningEffort::None => "none",
                        openai_frontend::ReasoningEffort::Minimal => "minimal",
                        openai_frontend::ReasoningEffort::Low => "low",
                        openai_frontend::ReasoningEffort::Medium => "medium",
                        openai_frontend::ReasoningEffort::High => "high",
                        openai_frontend::ReasoningEffort::Xhigh => "xhigh",
                        openai_frontend::ReasoningEffort::Max => "max",
                    }),
                );
            }
            EmbeddedReasoningBudget::Auto => {}
        }
    }
    merged.extend(request.clone());
    Ok((!merged.is_empty()).then_some(merged))
}

fn request_reasoning_format(
    request: &ChatCompletionRequest,
    defaults: &EmbeddedOpenAiRequestDefaults,
) -> OpenAiResult<ChatReasoningFormat> {
    let value = optional_string_extra(&request.extra, "reasoning_format")?;
    match value.as_deref() {
        Some("auto") => Ok(ChatReasoningFormat::Auto),
        None => Ok(chat_reasoning_format(defaults.reasoning_format)),
        Some("none") => Ok(ChatReasoningFormat::None),
        Some("deepseek") => Ok(ChatReasoningFormat::Deepseek),
        Some("deepseek-legacy") => Ok(ChatReasoningFormat::DeepseekLegacy),
        Some("hidden") => Ok(ChatReasoningFormat::Hidden),
        Some(_) => Err(OpenAiError::invalid_request(
            "reasoning_format must be auto, none, deepseek, deepseek-legacy, or hidden",
        )),
    }
}

fn optional_string_extra(
    extra: &std::collections::BTreeMap<String, Value>,
    name: &str,
) -> OpenAiResult<Option<String>> {
    extra
        .get(name)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| OpenAiError::invalid_request(format!("{name} must be a string")))
        })
        .transpose()
}

fn bounded_optional_string_extra(
    extra: &std::collections::BTreeMap<String, Value>,
    name: &str,
) -> OpenAiResult<Option<String>> {
    let value = optional_string_extra(extra, name)?;
    if value
        .as_ref()
        .is_some_and(|value| value.len() > MAX_NATIVE_PARSER_INPUT_BYTES)
    {
        return Err(OpenAiError::invalid_request(format!(
            "{name} exceeds the {MAX_NATIVE_PARSER_INPUT_BYTES}-byte limit"
        )));
    }
    Ok(value)
}

fn optional_bool_extra(
    extra: &std::collections::BTreeMap<String, Value>,
    name: &str,
) -> OpenAiResult<Option<bool>> {
    extra
        .get(name)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| OpenAiError::invalid_request(format!("{name} must be a boolean")))
        })
        .transpose()
}

fn structured_output_string(
    request: &ChatCompletionRequest,
    name: &str,
) -> OpenAiResult<Option<String>> {
    let value = bounded_optional_string_extra(&request.extra, name)?;
    if value.is_some()
        && request
            .extra
            .get("json_schema")
            .is_some_and(|value| !value.is_null())
    {
        return Err(OpenAiError::invalid_request(
            "grammar and json_schema cannot both be set",
        ));
    }
    Ok(value)
}

fn structured_output_json(
    request: &ChatCompletionRequest,
    name: &str,
) -> OpenAiResult<Option<String>> {
    request
        .extra
        .get(name)
        .filter(|value| !value.is_null())
        .map(|value| {
            if !value.is_object() {
                return Err(OpenAiError::invalid_request(
                    "json_schema must be an object",
                ));
            }
            serialize_bounded_native_parser_json("json_schema", value)
        })
        .transpose()
}

fn serialize_bounded_native_parser_json(
    name: &str,
    value: &impl serde::Serialize,
) -> OpenAiResult<String> {
    let serialized = serde_json::to_string(value)
        .map_err(|error| OpenAiError::invalid_request(format!("serialize {name}: {error}")))?;
    if serialized.len() > MAX_NATIVE_PARSER_INPUT_BYTES {
        return Err(OpenAiError::invalid_request(format!(
            "{name} exceeds the {MAX_NATIVE_PARSER_INPUT_BYTES}-byte limit"
        )));
    }
    Ok(serialized)
}

fn default_reasoning_enabled(value: Option<EmbeddedReasoningEnabled>) -> Option<bool> {
    match value {
        Some(EmbeddedReasoningEnabled::Disabled) => Some(false),
        Some(EmbeddedReasoningEnabled::Enabled) => Some(true),
        Some(EmbeddedReasoningEnabled::Auto) | None => None,
    }
}

fn default_reasoning_budget_enabled(value: Option<EmbeddedReasoningBudget>) -> Option<bool> {
    match value {
        Some(EmbeddedReasoningBudget::Tokens(0)) => Some(false),
        Some(EmbeddedReasoningBudget::Tokens(_)) => Some(true),
        Some(EmbeddedReasoningBudget::Effort(openai_frontend::ReasoningEffort::None)) => {
            Some(false)
        }
        Some(EmbeddedReasoningBudget::Effort(_)) => Some(true),
        Some(EmbeddedReasoningBudget::Auto) | None => None,
    }
}

fn chat_reasoning_format(value: Option<EmbeddedReasoningFormat>) -> ChatReasoningFormat {
    match value.unwrap_or(EmbeddedReasoningFormat::Auto) {
        EmbeddedReasoningFormat::Auto => ChatReasoningFormat::Auto,
        EmbeddedReasoningFormat::None => ChatReasoningFormat::None,
        EmbeddedReasoningFormat::Deepseek => ChatReasoningFormat::Deepseek,
        EmbeddedReasoningFormat::DeepseekLegacy => ChatReasoningFormat::DeepseekLegacy,
        EmbeddedReasoningFormat::Hidden => ChatReasoningFormat::Hidden,
    }
}

pub(super) fn ensure_chat_runtime_features_supported(
    request: &ChatCompletionRequest,
) -> OpenAiResult<()> {
    if request.logprobs.unwrap_or(false) || request.top_logprobs.is_some() {
        return Err(OpenAiError::unsupported(
            "chat logprobs are parsed by openai-frontend but not yet implemented by skippy runtime",
        ));
    }
    Ok(())
}

pub(super) fn ensure_completion_runtime_features_supported(
    request: &CompletionRequest,
) -> OpenAiResult<()> {
    if request.logprobs.is_some() {
        return Err(OpenAiError::unsupported(
            "completion logprobs are parsed by openai-frontend but not yet implemented by skippy runtime",
        ));
    }
    Ok(())
}

pub(super) fn has_requested_tools(value: &Value) -> bool {
    !matches!(value, Value::Array(items) if items.is_empty())
}

pub(super) fn ensure_extra_generation_fields_absent(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
) -> OpenAiResult<()> {
    const UNSUPPORTED_FIELDS: &[&str] = &["adaptive"];

    for field in UNSUPPORTED_FIELDS {
        if extra.get(*field).is_some_and(|value| !value.is_null()) {
            return Err(OpenAiError::unsupported(format!(
                "{field} is parsed but not yet implemented"
            )));
        }
    }
    Ok(())
}

pub(super) fn sampling_config(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    logit_bias: Option<&std::collections::BTreeMap<String, serde_json::Value>>,
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
) -> OpenAiResult<SamplingConfig> {
    ensure_extra_generation_fields_absent(extra)?;
    let temperature = temperature.unwrap_or(0.8);
    let top_p = top_p.unwrap_or(0.95);
    let presence_penalty = presence_penalty.unwrap_or(0.0);
    let frequency_penalty = frequency_penalty.unwrap_or(0.0);
    let top_k = optional_i32_extra(extra, "top_k")?.unwrap_or(40);
    let min_p = optional_f32_extra(extra, "min_p")?.unwrap_or(0.05);
    let repeat_penalty = optional_f32_extra(extra, "repeat_penalty")?
        .or(optional_f32_extra(extra, "repetition_penalty")?)
        .unwrap_or(1.0);
    let penalty_last_n = optional_i32_extra(extra, "repeat_last_n")?.unwrap_or(-1);
    let typical_p = optional_f32_extra(extra, "typical_p")?.unwrap_or(1.0);
    let top_nsigma = optional_f32_extra(extra, "top_nsigma")?.unwrap_or(-1.0);
    let dynatemp_range = optional_f32_extra(extra, "dynatemp_range")?.unwrap_or(0.0);
    let dynatemp_exponent = optional_f32_extra(extra, "dynatemp_exponent")?.unwrap_or(1.0);
    let dry = parse_dry_sampling(extra.get("dry"))?;
    let xtc = parse_xtc_sampling(extra.get("xtc"))?;
    let mirostat_mode = optional_i32_extra(extra, "mirostat_mode")?.unwrap_or(0);
    let mirostat_entropy = optional_f32_extra(extra, "mirostat_entropy")?.unwrap_or(5.0);
    let mirostat_learning_rate =
        optional_f32_extra(extra, "mirostat_learning_rate")?.unwrap_or(0.1);
    let samplers = parse_sampler_order(extra)?;
    let ignore_eos = optional_bool_extra(extra, "ignore_eos")?.unwrap_or(false);
    validate_sampling_range("temperature", temperature, 0.0..=100.0)?;
    validate_sampling_range("top_p", top_p, 0.0..=1.0)?;
    validate_sampling_range("presence_penalty", presence_penalty, -2.0..=2.0)?;
    validate_sampling_range("frequency_penalty", frequency_penalty, -2.0..=2.0)?;
    validate_sampling_range("min_p", min_p, 0.0..=1.0)?;
    validate_sampling_range("repeat_penalty", repeat_penalty, 0.0..=100.0)?;
    validate_sampling_range("typical_p", typical_p, 0.0..=1.0)?;
    if top_k < 0 {
        return Err(OpenAiError::invalid_request(
            "top_k must be greater than or equal to zero",
        ));
    }
    if penalty_last_n < -1 {
        return Err(OpenAiError::invalid_request(
            "repeat_last_n must be greater than or equal to -1",
        ));
    }
    let seed = match seed {
        Some(seed) => u32::try_from(seed)
            .map_err(|_| OpenAiError::invalid_request("seed exceeds u32 range"))?,
        None => 0,
    };
    let logit_bias = parse_logit_bias(logit_bias)?;
    let defaults = SamplingConfig::default();
    let enabled = seed != 0
        || temperature <= 0.0
        || (temperature - 1.0).abs() > f32::EPSILON
        || (top_p - 1.0).abs() > f32::EPSILON
        || top_k > 0
        || min_p > 0.0
        || presence_penalty.abs() > f32::EPSILON
        || frequency_penalty.abs() > f32::EPSILON
        || (repeat_penalty - 1.0).abs() > f32::EPSILON
        || penalty_last_n != -1
        || !logit_bias.is_empty()
        || (typical_p - defaults.typical_p).abs() > f32::EPSILON
        || (top_nsigma - defaults.top_nsigma).abs() > f32::EPSILON
        || dynatemp_range.abs() > f32::EPSILON
        || (dynatemp_exponent - defaults.dynatemp_exponent).abs() > f32::EPSILON
        || dry != defaults.dry
        || xtc != defaults.xtc
        || mirostat_mode != defaults.mirostat_mode
        || (mirostat_entropy - defaults.mirostat_entropy).abs() > f32::EPSILON
        || (mirostat_learning_rate - defaults.mirostat_learning_rate).abs() > f32::EPSILON
        || samplers != defaults.samplers
        || ignore_eos;
    Ok(SamplingConfig {
        enabled,
        seed,
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        frequency_penalty,
        repeat_penalty,
        penalty_last_n,
        logit_bias,
        typical_p,
        top_nsigma,
        dynatemp_range,
        dynatemp_exponent,
        dry,
        xtc,
        mirostat_mode,
        mirostat_entropy,
        mirostat_learning_rate,
        samplers,
        ignore_eos,
    })
}

fn parse_dry_sampling(value: Option<&Value>) -> OpenAiResult<DrySamplingConfig> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(SamplingConfig::default().dry);
    };
    let object = value
        .as_object()
        .ok_or_else(|| OpenAiError::invalid_request("dry must be an object"))?;
    ensure_allowed_object_keys(
        object,
        "dry",
        &[
            "multiplier",
            "base",
            "allowed_length",
            "penalty_last_n",
            "sequence_breakers",
        ],
    )?;
    let defaults = SamplingConfig::default().dry;
    let sequence_breakers = match object.get("sequence_breakers") {
        None | Some(Value::Null) => defaults.sequence_breakers,
        Some(Value::Array(values)) if values.len() <= MAX_STAGE_DRY_SEQUENCE_BREAKERS => values
            .iter()
            .map(|value| {
                let value = value.as_str().ok_or_else(|| {
                    OpenAiError::invalid_request("dry.sequence_breakers must contain strings")
                })?;
                if value.len() >= MAX_DRY_SEQUENCE_BREAKER_BYTES {
                    return Err(OpenAiError::invalid_request(
                        "dry.sequence_breakers entry exceeds maximum length",
                    ));
                }
                Ok(value.to_string())
            })
            .collect::<OpenAiResult<Vec<_>>>()?,
        Some(Value::Array(_)) => {
            return Err(OpenAiError::invalid_request(
                "dry.sequence_breakers contains too many entries",
            ));
        }
        Some(_) => {
            return Err(OpenAiError::invalid_request(
                "dry.sequence_breakers must be an array",
            ));
        }
    };
    Ok(DrySamplingConfig {
        multiplier: optional_object_f32(object, "dry.multiplier", "multiplier")?
            .unwrap_or(defaults.multiplier),
        base: optional_object_f32(object, "dry.base", "base")?.unwrap_or(defaults.base),
        allowed_length: optional_object_i32(object, "dry.allowed_length", "allowed_length")?
            .unwrap_or(defaults.allowed_length),
        penalty_last_n: optional_object_i32(object, "dry.penalty_last_n", "penalty_last_n")?
            .unwrap_or(defaults.penalty_last_n),
        sequence_breakers,
    })
}

fn parse_xtc_sampling(value: Option<&Value>) -> OpenAiResult<XtcSamplingConfig> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(SamplingConfig::default().xtc);
    };
    let object = value
        .as_object()
        .ok_or_else(|| OpenAiError::invalid_request("xtc must be an object"))?;
    ensure_allowed_object_keys(object, "xtc", &["probability", "threshold"])?;
    let defaults = SamplingConfig::default().xtc;
    Ok(XtcSamplingConfig {
        probability: optional_object_f32(object, "xtc.probability", "probability")?
            .unwrap_or(defaults.probability),
        threshold: optional_object_f32(object, "xtc.threshold", "threshold")?
            .unwrap_or(defaults.threshold),
    })
}

fn optional_object_f32(
    object: &serde_json::Map<String, Value>,
    name: &str,
    key: &str,
) -> OpenAiResult<Option<f32>> {
    object
        .get(key)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{name} must be a number")))
        })
        .transpose()
}

fn optional_object_i32(
    object: &serde_json::Map<String, Value>,
    name: &str,
    key: &str,
) -> OpenAiResult<Option<i32>> {
    object
        .get(key)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{name} must be an integer")))
        })
        .transpose()
}

fn ensure_allowed_object_keys(
    object: &serde_json::Map<String, Value>,
    name: &str,
    allowed: &[&str],
) -> OpenAiResult<()> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(OpenAiError::invalid_request(format!(
            "{name} contains unknown field {key}"
        )));
    }
    Ok(())
}

fn parse_sampler_order(
    extra: &std::collections::BTreeMap<String, Value>,
) -> OpenAiResult<Vec<String>> {
    if let Some(value) = extra.get("samplers").filter(|value| !value.is_null()) {
        let values = value
            .as_array()
            .ok_or_else(|| OpenAiError::invalid_request("samplers must be an array"))?;
        if values.len() > MAX_STAGE_SAMPLERS {
            return Err(OpenAiError::invalid_request(
                "samplers contains too many entries",
            ));
        }
        return values
            .iter()
            .map(|value| {
                let sampler = value
                    .as_str()
                    .ok_or_else(|| OpenAiError::invalid_request("samplers must contain strings"))?;
                canonical_sampler_name(sampler).map(str::to_string)
            })
            .collect();
    }
    if let Some(value) = extra
        .get("sampler_sequence")
        .filter(|value| !value.is_null())
    {
        let sequence = value
            .as_str()
            .ok_or_else(|| OpenAiError::invalid_request("sampler_sequence must be a string"))?;
        let samplers = sequence
            .chars()
            .filter(|value| !value.is_whitespace())
            .map(|value| match value {
                'e' => Ok("penalties"),
                'd' => Ok("dry"),
                's' => Ok("top_n_sigma"),
                'k' => Ok("top_k"),
                'y' => Ok("typical_p"),
                'p' => Ok("top_p"),
                'm' => Ok("min_p"),
                'x' => Ok("xtc"),
                't' => Ok("temperature"),
                _ => Err(OpenAiError::invalid_request(
                    "sampler_sequence contains an unsupported sampler code",
                )),
            })
            .map(|result| result.map(str::to_string))
            .collect::<OpenAiResult<Vec<_>>>()?;
        if samplers.len() > MAX_STAGE_SAMPLERS {
            return Err(OpenAiError::invalid_request(
                "sampler_sequence contains too many entries",
            ));
        }
        return Ok(samplers);
    }
    Ok(SamplingConfig::default().samplers)
}

fn canonical_sampler_name(value: &str) -> OpenAiResult<&'static str> {
    match value {
        "penalties" => Ok("penalties"),
        "dry" => Ok("dry"),
        "top_n_sigma" => Ok("top_n_sigma"),
        "top_k" => Ok("top_k"),
        "typical_p" | "typ_p" => Ok("typical_p"),
        "top_p" => Ok("top_p"),
        "min_p" => Ok("min_p"),
        "xtc" => Ok("xtc"),
        "temperature" | "temp" => Ok("temperature"),
        _ => Err(OpenAiError::invalid_request(
            "samplers contains an unsupported sampler name",
        )),
    }
}

pub(super) fn parse_logit_bias(
    logit_bias: Option<&std::collections::BTreeMap<String, serde_json::Value>>,
) -> OpenAiResult<Vec<RuntimeLogitBias>> {
    let Some(logit_bias) = logit_bias else {
        return Ok(Vec::new());
    };
    if logit_bias.len() > MAX_LOGIT_BIAS {
        return Err(OpenAiError::invalid_request(format!(
            "logit_bias supports at most {MAX_LOGIT_BIAS} entries"
        )));
    }
    let mut parsed = Vec::with_capacity(logit_bias.len());
    for (token_id, bias) in logit_bias {
        let token_id = token_id
            .parse::<i32>()
            .map_err(|_| OpenAiError::invalid_request("logit_bias token IDs must be integers"))?;
        if token_id < 0 {
            return Err(OpenAiError::invalid_request(
                "logit_bias token IDs must be greater than or equal to zero",
            ));
        }
        let bias = serde_json::from_value::<f32>(bias.clone())
            .map_err(|_| OpenAiError::invalid_request("logit_bias values must be numbers"))?;
        validate_sampling_range("logit_bias", bias, -100.0..=100.0)?;
        parsed.push(RuntimeLogitBias { token_id, bias });
    }
    Ok(parsed)
}

pub(super) fn validate_sampling_range(
    name: &str,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
) -> OpenAiResult<()> {
    if !value.is_finite() || !range.contains(&value) {
        return Err(OpenAiError::invalid_request(format!(
            "{name} is outside the supported range"
        )));
    }
    Ok(())
}

pub(super) fn optional_f32_extra(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
    field: &str,
) -> OpenAiResult<Option<f32>> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<f32>(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{field} must be a number")))
        })
        .transpose()
}

pub(super) fn optional_i32_extra(
    extra: &std::collections::BTreeMap<String, serde_json::Value>,
    field: &str,
) -> OpenAiResult<Option<i32>> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<i32>(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{field} must be an integer")))
        })
        .transpose()
}

pub(super) fn wire_sampling_config(sampling: &SamplingConfig) -> Option<WireSamplingConfig> {
    if !sampling.enabled {
        return None;
    }
    let mut wire = WireSamplingConfig {
        flags: u32::from(sampling.enabled),
        seed: sampling.seed,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        repeat_penalty: sampling.repeat_penalty,
        penalty_last_n: sampling.penalty_last_n,
        typical_p: sampling.typical_p,
        top_nsigma: sampling.top_nsigma,
        dynatemp_range: sampling.dynatemp_range,
        dynatemp_exponent: sampling.dynatemp_exponent,
        dry_multiplier: sampling.dry.multiplier,
        dry_base: sampling.dry.base,
        dry_allowed_length: sampling.dry.allowed_length,
        dry_penalty_last_n: sampling.dry.penalty_last_n,
        dry_sequence_breakers: sampling.dry.sequence_breakers.clone(),
        xtc_probability: sampling.xtc.probability,
        xtc_threshold: sampling.xtc.threshold,
        mirostat_mode: sampling.mirostat_mode,
        mirostat_entropy: sampling.mirostat_entropy,
        mirostat_learning_rate: sampling.mirostat_learning_rate,
        samplers: sampling.samplers.clone(),
        ignore_eos: sampling.ignore_eos,
        ..WireSamplingConfig::default()
    };
    wire.logit_bias = sampling
        .logit_bias
        .iter()
        .take(MAX_STAGE_LOGIT_BIAS)
        .map(|source| WireLogitBias {
            token_id: source.token_id,
            bias: source.bias,
        })
        .collect();
    Some(wire)
}
