use super::support::*;
use super::*;

#[test]
fn chat_runtime_feature_guard_allows_noop_parity_fields() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [],
        "tool_choice": null,
        "parallel_tool_calls": false,
        "response_format": {"type": "text"}
    }))
    .unwrap();

    ensure_chat_runtime_features_supported(&request).unwrap();
}

#[test]
fn chat_runtime_feature_guard_rejects_logprobs() {
    for payload in [
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": false,
            "top_logprobs": 1
        }),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
        let error = ensure_chat_runtime_features_supported(&request).unwrap_err();
        assert_eq!(
            unsupported_code(error),
            Some("unsupported_model_feature".to_string())
        );
    }
}

#[test]
fn completion_runtime_feature_guard_rejects_logprobs() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "prompt": "hi",
        "logprobs": 2
    }))
    .unwrap();

    let error = ensure_completion_runtime_features_supported(&request).unwrap_err();
    assert_eq!(
        unsupported_code(error),
        Some("unsupported_model_feature".to_string())
    );
}

#[test]
fn default_sampling_controls_are_allowed() {
    // When no sampling params are specified, the server applies its own
    // defaults (temp=0.8, top_k=40, top_p=0.95, min_p=0.05) which enable
    // the sampling chain automatically.
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.temperature, 0.8);
    assert_eq!(sampling.top_p, 0.95);
    assert_eq!(sampling.top_k, 40);
    assert_eq!(sampling.min_p, 0.05);
}

#[test]
fn non_default_sampling_controls_are_enabled() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "prompt": "hello",
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42
    }))
    .unwrap();

    let sampling = completion_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.seed, 42);
    assert_eq!(sampling.temperature, 0.7);
}

#[test]
fn typed_sampling_penalties_are_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "presence_penalty": 1.0
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.presence_penalty, 1.0);
}

#[test]
fn extra_sampling_fields_are_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "top_k": 40
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.top_k, 40);
}

#[test]
fn request_defaults_fill_omitted_chat_fields_only() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    apply_chat_request_defaults(&mut request, &test_request_defaults());

    let sampling = chat_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.2));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.presence_penalty, Some(1.25));
    assert_eq!(request.frequency_penalty, Some(0.5));
    assert_eq!(request.seed, Some(77));
    assert_eq!(request.logit_bias, test_request_defaults().logit_bias);
    assert_eq!(
        request.stop,
        Some(openai_frontend::StopSequence::One("</stop>".to_string()))
    );
    assert_eq!(sampling.temperature, 0.2);
    assert_eq!(sampling.top_p, 0.9);
    assert_eq!(sampling.presence_penalty, 1.25);
    assert_eq!(sampling.frequency_penalty, 0.5);
    assert_eq!(sampling.seed, 77);
    assert_eq!(sampling.top_k, 12);
    assert_eq!(sampling.min_p, 0.1);
    assert_eq!(sampling.repeat_penalty, 1.2);
    assert_eq!(sampling.penalty_last_n, 64);
    assert_eq!(sampling.logit_bias.len(), 2);
    let template_options = chat_template_options(&request, &test_request_defaults()).unwrap();
    assert_eq!(template_options.enable_thinking, None);
    assert_eq!(
        template_options.reasoning_format,
        Some(ChatReasoningFormat::Hidden)
    );
    assert_eq!(request.reasoning, None);
    assert_eq!(
        GenerationTokenLimit::from_request(request.effective_max_tokens(), 64),
        GenerationTokenLimit::Default(64)
    );
}

#[test]
fn request_defaults_fill_omitted_completion_fields_and_nulls() {
    let mut request: CompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "prompt": "hello",
        "top_k": null,
        "repeat_last_n": null,
        "min_p": null
    }))
    .unwrap();

    apply_completion_request_defaults(&mut request, &test_request_defaults());

    let sampling = completion_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.2));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.presence_penalty, Some(1.25));
    assert_eq!(request.frequency_penalty, Some(0.5));
    assert_eq!(request.seed, Some(77));
    assert_eq!(request.logit_bias, test_request_defaults().logit_bias);
    assert_eq!(
        request.stop,
        Some(openai_frontend::StopSequence::One("</stop>".to_string()))
    );
    assert_eq!(sampling.seed, 77);
    assert_eq!(sampling.presence_penalty, 1.25);
    assert_eq!(sampling.frequency_penalty, 0.5);
    assert_eq!(sampling.top_k, 12);
    assert_eq!(sampling.min_p, 0.1);
    assert_eq!(sampling.repeat_penalty, 1.2);
    assert_eq!(sampling.penalty_last_n, 64);
    assert_eq!(sampling.logit_bias.len(), 2);
    assert_eq!(
        GenerationTokenLimit::from_request(request.max_tokens, 48),
        GenerationTokenLimit::Default(48)
    );
}

#[test]
fn explicit_chat_request_values_override_request_defaults() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "temperature": 0.8,
        "top_p": 0.7,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 9,
        "logit_bias": {"7": 1.0},
        "stop": ["USER"],
        "repetition_penalty": 1.8,
        "repeat_last_n": 24,
        "reasoning": {"enabled": false}
    }))
    .unwrap();

    apply_chat_request_defaults(&mut request, &test_request_defaults());

    let sampling = chat_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.8));
    assert_eq!(request.top_p, Some(0.7));
    assert_eq!(request.presence_penalty, Some(0.1));
    assert_eq!(request.frequency_penalty, Some(0.2));
    assert_eq!(request.seed, Some(9));
    assert_eq!(request.effective_max_tokens(), Some(32));
    assert_eq!(
        request.stop,
        Some(openai_frontend::StopSequence::Many(vec![
            "USER".to_string()
        ]))
    );
    assert_eq!(sampling.top_p, 0.7);
    assert_eq!(sampling.presence_penalty, 0.1);
    assert_eq!(sampling.frequency_penalty, 0.2);
    assert_eq!(sampling.seed, 9);
    assert_eq!(sampling.repeat_penalty, 1.8);
    assert_eq!(sampling.penalty_last_n, 24);
    assert_eq!(sampling.logit_bias.len(), 1);
    let template_options = chat_template_options(&request, &test_request_defaults()).unwrap();
    assert_eq!(template_options.enable_thinking, Some(false));
    assert_eq!(
        template_options.reasoning_format,
        Some(ChatReasoningFormat::Hidden)
    );
    assert_eq!(
        GenerationTokenLimit::from_request(request.effective_max_tokens(), 64),
        GenerationTokenLimit::Explicit(32)
    );
}

#[test]
fn explicit_completion_request_values_override_request_defaults() {
    let mut request: CompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "prompt": "hello",
        "max_tokens": 12,
        "temperature": 0.6,
        "top_p": 0.4,
        "presence_penalty": 0.25,
        "frequency_penalty": 0.75,
        "seed": 12,
        "logit_bias": {"8": -3.0},
        "stop": ["DONE"],
        "repeat_penalty": 1.4,
        "repeat_last_n": 16,
        "reasoning": {"enabled": false}
    }))
    .unwrap();

    apply_completion_request_defaults(&mut request, &test_request_defaults());

    let sampling = completion_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.6));
    assert_eq!(request.top_p, Some(0.4));
    assert_eq!(request.presence_penalty, Some(0.25));
    assert_eq!(request.frequency_penalty, Some(0.75));
    assert_eq!(request.seed, Some(12));
    assert_eq!(request.max_tokens, Some(12));
    assert_eq!(sampling.repeat_penalty, 1.4);
    assert_eq!(sampling.penalty_last_n, 16);
    assert_eq!(sampling.logit_bias.len(), 1);
    assert_eq!(
        GenerationTokenLimit::from_request(request.max_tokens, 48),
        GenerationTokenLimit::Explicit(12)
    );
}

#[test]
fn request_defaults_do_not_make_logprobs_executable() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}}
        },
        "logprobs": true,
        "top_logprobs": 2
    }))
    .unwrap();

    apply_chat_request_defaults(&mut request, &test_request_defaults());

    let error = ensure_chat_runtime_features_supported(&request).unwrap_err();
    assert_eq!(
        unsupported_code(error),
        Some("unsupported_model_feature".to_string())
    );
}

#[test]
fn canonical_reasoning_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": false}
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn chat_template_options_default_to_hidden_reasoning_parser() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    let options = chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
        .expect("template options");

    assert_eq!(options.enable_thinking, None);
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn request_default_reasoning_format_controls_chat_parser_mode() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    let defaults = EmbeddedOpenAiRequestDefaults {
        reasoning_format: Some(EmbeddedReasoningFormat::None),
        ..EmbeddedOpenAiRequestDefaults::default()
    };

    let options = chat_template_options(&request, &defaults).expect("template options");

    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::None));
    assert!(!chat_output_parser_required(&request, &options));
}

#[test]
fn reasoning_effort_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "none"}
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn top_level_reasoning_effort_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "none"
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn provider_enable_thinking_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": false},
        "enable_thinking": true
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(true));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn chat_template_kwargs_enable_thinking_overrides_template() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": false}
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn thinking_boolean_aliases_override_chat_template_thinking() {
    for field in openai_frontend::THINKING_BOOLEAN_ALIASES {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}],
            (*field): false
        }))
        .unwrap();
        assert_eq!(
            chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
                .unwrap()
                .enable_thinking,
            Some(false),
            "top-level alias {field}"
        );

        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {(*field): false}
        }))
        .unwrap();
        assert_eq!(
            chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
                .unwrap()
                .enable_thinking,
            Some(false),
            "chat_template_kwargs alias {field}"
        );
    }
}

#[test]
fn reasoning_budget_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"max_tokens": 1024}
    }))
    .unwrap();
    assert_eq!(
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
            .unwrap()
            .enable_thinking,
        Some(true)
    );

    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": true},
        "thinking_budget": 0
    }))
    .unwrap();
    assert_eq!(
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
            .unwrap()
            .enable_thinking,
        Some(false)
    );
}

#[test]
fn logit_bias_is_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "logit_bias": {"123": -50.0, "456": 12.5}
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.logit_bias.len(), 2);
    assert_eq!(sampling.logit_bias[0].token_id, 123);
    assert_eq!(sampling.logit_bias[0].bias, -50.0);
    assert_eq!(sampling.logit_bias[1].token_id, 456);
    assert_eq!(sampling.logit_bias[1].bias, 12.5);
}

#[test]
fn invalid_logit_bias_returns_openai_error() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "logit_bias": {"not-a-token": 1.0}
    }))
    .unwrap();

    let error = chat_sampling_config(&request).unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
}

#[test]
fn unsupported_extra_generation_fields_return_openai_error() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "typical_p": 0.5
    }))
    .unwrap();

    let error = chat_sampling_config(&request).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
}

#[test]
fn min_p_is_accepted_and_forwarded() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "min_p": 0.1
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.min_p, 0.1);
}
