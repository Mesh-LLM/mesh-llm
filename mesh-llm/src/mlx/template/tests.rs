use super::*;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct HfTemplateFixture {
    repo: String,
    source_file: String,
    expect_hf_render: bool,
    family: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    pad_token: Option<String>,
    unk_token: Option<String>,
    template: String,
}

fn hf_template_corpus() -> Vec<HfTemplateFixture> {
    serde_json::from_str(include_str!("../testdata/hf_template_corpus.json"))
        .expect("valid HF template corpus")
}

fn fixture_config(family: &str) -> Value {
    match family {
        "llama" => json!({"model_type":"llama","architectures":["LlamaForCausalLM"]}),
        "qwen" | "qwen3" | "qwen3_coder_next" | "deepseek_qwen3" => {
            json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]})
        }
        "qwen3_coder_30b" => json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}),
        "gemma3" => {
            json!({"model_type":"gemma3","architectures":["Gemma3ForConditionalGeneration"]})
        }
        "mistral" => json!({"model_type":"mistral","architectures":["MistralForCausalLM"]}),
        "lfm2" => json!({"model_type":"lfm2","architectures":["LlamaForCausalLM"]}),
        "devstral" => json!({"model_type":"mistral","architectures":["MistralForCausalLM"]}),
        "glm4v" => json!({"model_type":"glm","architectures":["GlmForCausalLM"]}),
        "kimi" => json!({"model_type":"kimi","architectures":["KimiForCausalLM"]}),
        "gpt_oss" => json!({"model_type":"gpt_oss","architectures":["GptOssForCausalLM"]}),
        other => panic!("unknown fixture family: {other}"),
    }
}

fn fixture_request(family: &str) -> Value {
    match family {
        "llama" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ],
            "add_generation_prompt": true
        }),
        "qwen" => json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}],
            "add_generation_prompt": true
        }),
        "gemma3" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": [
                    {"type": "text", "text": "look "},
                    {"type": "image"},
                    {"type": "text", "text": "here"}
                ]}
            ],
            "add_generation_prompt": true
        }),
        "mistral" => json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"}
            ],
            "add_generation_prompt": true
        }),
        "lfm2" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think>\ninternal\n</think>\nhi"},
                {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image"}]}
            ],
            "keep_past_thinking": false,
            "add_generation_prompt": true
        }),
        "deepseek_qwen3" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ],
            "add_generation_prompt": true
        }),
        "qwen3" | "qwen3_coder_next" | "qwen3_coder_30b" => json!({
            "messages": [{"role": "user", "content": "hello"}],
            "add_generation_prompt": true
        }),
        "devstral" => json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ],
            "add_generation_prompt": true
        }),
        "glm4v" => json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}],
            "add_generation_prompt": true
        }),
        "kimi" => json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url"}]}],
            "add_generation_prompt": true
        }),
        "gpt_oss" => json!({
            "messages": [{"role": "user", "content": "hello"}],
            "builtin_tools": ["browser", "python"],
            "reasoning_effort": "medium",
            "add_generation_prompt": true
        }),
        other => panic!("unknown fixture family: {other}"),
    }
}

fn write_hf_fixture_dir(fixture: &HfTemplateFixture) -> std::path::PathBuf {
    let slug = fixture
        .repo
        .replace('/', "-")
        .replace('.', "-")
        .replace('_', "-");
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-hf-template-corpus-{}-{}",
        slug,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    match fixture.source_file.as_str() {
        "chat_template.jinja" => {
            std::fs::write(root.join("chat_template.jinja"), &fixture.template).unwrap();
        }
        "chat_template.json" => {
            std::fs::write(
                root.join("chat_template.json"),
                serde_json::to_string(&fixture.template).unwrap(),
            )
            .unwrap();
        }
        "tokenizer_config.json" => {
            std::fs::write(
                root.join("tokenizer_config.json"),
                serde_json::json!({
                    "chat_template": fixture.template,
                    "bos_token": fixture.bos_token,
                    "eos_token": fixture.eos_token,
                    "pad_token": fixture.pad_token,
                    "unk_token": fixture.unk_token,
                })
                .to_string(),
            )
            .unwrap();
            return root;
        }
        other => panic!("unknown template source: {other}"),
    }

    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": fixture.bos_token,
            "eos_token": fixture.eos_token,
            "pad_token": fixture.pad_token,
            "unk_token": fixture.unk_token,
        })
        .to_string(),
    )
    .unwrap();

    root
}

#[test]
fn normalizes_python_get_calls() {
    let template = "{{ msg.get('content') }} {{ msg.get(\"role\", \"user\") }}";
    let normalized = normalize_hf_template(template);
    assert_eq!(
        normalized,
        "{{ msg[\"content\"]|default(none) }} {{ msg[\"role\"]|default(\"user\") }}"
    );
}

#[test]
fn normalizes_tojson_keyword_arguments() {
    let template = "{{ tools | tojson(separators=(',', ':'), ensure_ascii=False) }}";
    let normalized = normalize_hf_template(template);
    assert_eq!(normalized, "{{ tools | tojson }}");
}

#[test]
fn prefers_chat_template_jinja_over_tokenizer_config() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-jinja-precedence-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("chat_template.jinja"), "{{ '<jinja-template>' }}").unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{{ '<json-template>' }}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert_eq!(prompt, "<jinja-template>");
}

#[test]
fn falls_back_when_template_uses_unsupported_python_method() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-unsupported-method-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{% if messages[0].content.removeprefix('h') %}<bad>{% endif %}",
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("hello"));
}

#[test]
fn kimi_template_compiles_after_normalization() {
    let fixture = hf_template_corpus()
        .into_iter()
        .find(|fixture| fixture.repo == "mlx-community/Kimi-K2.5")
        .expect("kimi fixture exists");
    let normalized = normalize_hf_template(&fixture.template);
    validate_hf_template(&normalized).expect("normalized Kimi template should compile");
}

#[test]
fn detects_chatml_from_tokenizer_config() {
    let root =
        std::env::temp_dir().join(format!("mesh-llm-template-chatml-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    match template {
        PromptTemplate::HuggingFace { fallback, .. } => {
            assert_eq!(
                *fallback,
                PromptTemplate::ChatMl {
                    default_system_prompt: Some("You are a helpful assistant.".to_string())
                }
            );
        }
        other => panic!("expected huggingface template, got {other:?}"),
    }
}

#[test]
fn renders_llama3_prompt_from_hf_template() {
    let root =
        std::env::temp_dir().join(format!("mesh-llm-template-llama3-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<|begin_of_text|>",
            "chat_template": "{{- bos_token }}{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"llama"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    assert!(prompt.starts_with("<|begin_of_text|>"));
    assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"));
    assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
}

#[test]
fn renders_qwen_tools_template_with_minijinja() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-qwen-tools-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "chat_template": "{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{{- messages[0]['content'] }}{%- else %}{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}{%- endif %}{{- '\\n\\n# Tools\\n\\n<tools>' }}{%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}{{- '\\n</tools><|im_end|>\\n' }}{%- endif %}{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Run a command"}}]
        }))
        .unwrap();

    assert!(prompt.contains("# Tools"));
    assert!(prompt.contains("\"name\":\"run\""));
    assert!(prompt.contains("<|im_start|>assistant\n"));
}

#[test]
fn qwen_prompt_parity_fixture_matches_expected_output() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-qwen-fixture-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{%- if messages[0]['role'] != 'system' -%}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{%- endif -%}{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

    assert_eq!(
        prompt,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nhello<|im_end|><|im_start|>assistant"
    );
}

#[test]
fn llama3_prompt_parity_fixture_matches_expected_output() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-llama3-fixture-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<|begin_of_text|>",
            "chat_template": "{{- bos_token }}{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"llama"}));
    let prompt = template
        .render_request(&json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"}
            ]
        }))
        .unwrap();

    assert_eq!(
        prompt,
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    );
}

#[test]
fn gemma3_prompt_parity_fixture_matches_expected_output() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-template-gemma3-fixture-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<bos>",
            "chat_template": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- endif -%}\n    {{ '<end_of_turn>\\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\\n'}}\n{%- endif -%}\n"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"gemma3"}));
    let prompt = template
        .render_request(&json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": [
                    {"type": "text", "text": "look "},
                    {"type": "image"},
                    {"type": "text", "text": "here"}
                ]}
            ]
        }))
        .unwrap();

    assert_eq!(
        prompt,
        "<bos><start_of_turn>user\nBe concise.\n\nhello<end_of_turn>\n<start_of_turn>model\nhi<end_of_turn>\n<start_of_turn>user\nlook<start_of_image>here<end_of_turn>\n<start_of_turn>model\n"
    );
}

#[test]
fn heuristic_fallback_uses_gemma3_for_gemma_models() {
    let template = PromptTemplate::detect(
        Path::new("/tmp/does-not-need-to-exist"),
        &serde_json::json!({"model_type":"gemma3","architectures":["Gemma3ForConditionalGeneration"]}),
    );
    assert_eq!(template, PromptTemplate::Gemma3);
}

#[test]
fn renders_when_template_uses_strftime_now() {
    let root =
        std::env::temp_dir().join(format!("mesh-llm-template-fallback-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::json!({
            "chat_template": "{{ strftime_now('%Y-%m-%d') }}"
        })
        .to_string(),
    )
    .unwrap();

    let template = PromptTemplate::detect(&root, &serde_json::json!({"model_type":"qwen2"}));
    let prompt = template
        .render_request(&json!({
            "messages": [{"role": "user", "content": "hello world"}]
        }))
        .unwrap();

    assert_eq!(prompt.len(), 10);
    assert_eq!(prompt.chars().filter(|c| *c == '-').count(), 2);
}

#[test]
fn real_hf_template_corpus_behaves_as_expected() {
    for fixture in hf_template_corpus() {
        let root = write_hf_fixture_dir(&fixture);
        let config = fixture_config(&fixture.family);
        let req = fixture_request(&fixture.family);
        let special_tokens = read_special_tokens(&root);
        let normalized = normalize_hf_template(&fixture.template);

        if fixture.expect_hf_render {
            validate_hf_template(&normalized)
                .unwrap_or_else(|err| panic!("{} should compile: {err}", fixture.repo));
            let prompt = render_hf_template(&normalized, &special_tokens, &req)
                .unwrap_or_else(|err| panic!("{} should render via HF path: {err}", fixture.repo));
            assert!(
                !prompt.trim().is_empty(),
                "{} rendered an empty prompt",
                fixture.repo
            );
        } else {
            if validate_hf_template(&normalized).is_ok() {
                render_hf_template(&normalized, &special_tokens, &req)
                    .expect_err("fixture should still require fallback");
            }

            let prompt = PromptTemplate::detect(&root, &config)
                .render_request(&req)
                .unwrap_or_else(|render_err| {
                    panic!("{} should render via fallback: {render_err}", fixture.repo)
                });
            assert!(
                !prompt.trim().is_empty(),
                "{} fallback rendered an empty prompt",
                fixture.repo
            );
        }
    }
}
