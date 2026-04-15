use super::attention::apply_rope;
use super::config::{
    attention_window_size_for_layer, kv_shared_source_for_layer, normalized_model_config_json,
};
use super::family::{
    ensure_supported_mlx_model, model_architecture, uses_traditional_rope, ModelArchitecture,
};
use super::*;
use serde_json::Value;
use serial_test::serial;
use std::collections::HashMap;

#[test]
fn mlx_model_dir_accepts_directory_and_known_files() {
    let root = std::env::temp_dir().join(format!("mesh-llm-mlx-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), "{}").unwrap();
    std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(root.join("model.safetensors"), b"12345678").unwrap();

    assert_eq!(mlx_model_dir(&root), Some(root.as_path()));
    assert_eq!(
        mlx_model_dir(&root.join("config.json")),
        Some(root.as_path())
    );
    assert_eq!(
        mlx_model_dir(&root.join("model.safetensors")),
        Some(root.as_path())
    );

    std::fs::remove_file(root.join("model.safetensors")).unwrap();
    std::fs::write(root.join("model-00001-of-00002.safetensors"), b"12345678").unwrap();
    std::fs::write(root.join("model-00002-of-00002.safetensors"), b"12345678").unwrap();
    assert_eq!(
        mlx_model_dir(&root.join("model-00001-of-00002.safetensors")),
        Some(root.as_path())
    );
}

#[test]
fn config_supports_known_mlx_architectures() {
    let deepseek: Value = serde_json::json!({
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"]
    });
    let kimi: Value = serde_json::json!({
        "model_type": "kimi_k2",
        "architectures": ["DeepseekV3ForCausalLM"]
    });
    let glm4: Value = serde_json::json!({
        "model_type": "glm4",
        "architectures": ["Glm4ForCausalLM"]
    });
    let lfm2: Value = serde_json::json!({
        "model_type": "lfm2",
        "architectures": ["Lfm2ForCausalLM"]
    });
    let qwen: Value = serde_json::json!({
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"]
    });
    let phi3: Value = serde_json::json!({
        "model_type": "phi3",
        "architectures": ["Phi3ForCausalLM"]
    });
    let gpt_oss: Value = serde_json::json!({
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"]
    });
    let kimi_linear: Value = serde_json::json!({
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"]
    });
    let olmo2: Value = serde_json::json!({
        "model_type": "olmo2",
        "architectures": ["Olmo2ForCausalLM"]
    });
    let olmo: Value = serde_json::json!({
        "model_type": "olmo",
        "architectures": ["OlmoForCausalLM"]
    });
    let llama: Value = serde_json::json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    });
    let mistral: Value = serde_json::json!({
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"]
    });
    let gemma2: Value = serde_json::json!({
        "model_type": "gemma2",
        "architectures": ["Gemma2ForCausalLM"]
    });
    let gemma3: Value = serde_json::json!({
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"]
    });
    let gemma4: Value = serde_json::json!({
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"model_type": "gemma4_text"}
    });

    assert!(config_supports_mlx(&deepseek));
    assert!(config_supports_mlx(&kimi));
    assert!(config_supports_mlx(&glm4));
    assert!(config_supports_mlx(&lfm2));
    assert!(config_supports_mlx(&phi3));
    assert!(config_supports_mlx(&qwen));
    assert!(config_supports_mlx(&gpt_oss));
    assert!(config_supports_mlx(&kimi_linear));
    assert!(config_supports_mlx(&olmo));
    assert!(config_supports_mlx(&olmo2));
    assert!(config_supports_mlx(&llama));
    assert!(config_supports_mlx(&mistral));
    assert!(config_supports_mlx(&gemma2));
    assert!(config_supports_mlx(&gemma3));
    assert!(config_supports_mlx(&gemma4));
}

#[test]
fn config_rejects_other_reasoning_families_for_runtime_loading() {
    let glm: Value = serde_json::json!({
        "model_type": "glm",
        "architectures": ["GlmForCausalLM"]
    });
    let lfm2: Value = serde_json::json!({
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"]
    });

    assert!(!config_supports_mlx(&glm));
    assert!(!config_supports_mlx(&lfm2));
}

#[test]
fn phi3_tokenizer_patch_preserves_role_marker_whitespace() {
    let config = serde_json::json!({"model_type": "phi3"});
    let mut tokenizer = serde_json::json!({
        "added_tokens": [
            {"content":"<|user|>","rstrip":true},
            {"content":"<|assistant|>","rstrip":true},
            {"content":"<|end|>","rstrip":true},
            {"content":"<|endoftext|>","rstrip":true},
            {"content":"<irrelevant>","rstrip":true}
        ]
    });

    patch_phi3_special_token_whitespace(&mut tokenizer, &config);

    let added = tokenizer["added_tokens"].as_array().unwrap();
    assert_eq!(added[0]["rstrip"], Value::Bool(false));
    assert_eq!(added[1]["rstrip"], Value::Bool(false));
    assert_eq!(added[2]["rstrip"], Value::Bool(false));
    assert_eq!(added[3]["rstrip"], Value::Bool(true));
    assert_eq!(added[4]["rstrip"], Value::Bool(true));
}

#[test]
fn model_config_honors_explicit_head_dim() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": false,
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": 151645
    }))
    .unwrap();

    let derived = config.hidden_size / config.num_attention_heads;
    assert_eq!(derived, 64);
    assert_eq!(config.head_dim, Some(128));
    assert_eq!(
        config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / config.num_attention_heads),
        128
    );
}

#[test]
fn mistral_is_accepted_as_llama_like_mlx_architecture() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-mlx-mistral-supported-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let config = serde_json::json!({
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"]
    });

    ensure_supported_mlx_model(&root, &config).unwrap();
}

#[test]
fn olmo_is_accepted_as_mlx_architecture() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-mlx-olmo-supported-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let config = serde_json::json!({
        "model_type": "olmo",
        "architectures": ["OlmoForCausalLM"]
    });

    ensure_supported_mlx_model(&root, &config).unwrap();
}

#[test]
fn mistral_uses_traditional_rope() {
    let config = serde_json::json!({
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"]
    });
    let explicit = serde_json::json!({
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"],
        "rope_traditional": true
    });
    let llama = serde_json::json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    });

    assert!(!uses_traditional_rope(&config));
    assert!(uses_traditional_rope(&explicit));
    assert!(!uses_traditional_rope(&llama));
}

#[test]
fn unsupported_architecture_error_mentions_model_type() {
    let root =
        std::env::temp_dir().join(format!("mesh-llm-mlx-unsupported-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let config = serde_json::json!({
        "model_type": "starcoder2",
        "architectures": ["Starcoder2ForCausalLM"]
    });

    let err = ensure_supported_mlx_model(&root, &config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("unsupported MLX model architecture"));
    assert!(err.contains("model_type=starcoder2"));
    assert!(err.contains("Starcoder2ForCausalLM"));
}

#[test]
fn unsupported_reasoning_family_errors_are_explicit() {
    let root = std::env::temp_dir().join(format!(
        "mesh-llm-mlx-unsupported-reasoning-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    for config in [
        serde_json::json!({
            "model_type": "glm",
            "architectures": ["GlmForCausalLM"]
        }),
        serde_json::json!({
            "model_type": "lfm2_moe",
            "architectures": ["Lfm2MoeForCausalLM"]
        }),
    ] {
        let err = ensure_supported_mlx_model(&root, &config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported MLX model architecture"));
        assert!(err.contains("model_type="));
        assert!(err.contains("architectures="));
    }
}

#[test]
fn effective_text_config_extracts_gemma3_text_config() {
    let config = serde_json::json!({
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "quantization": {"group_size": 64, "bits": 4},
        "eos_token_id": [1, 106],
        "tie_word_embeddings": null,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000,
        "max_position_embeddings": 32768,
        "hidden_activation": "gelu_pytorch_tanh",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 1152,
            "num_hidden_layers": 26,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 262144
        }
    });

    let effective = effective_text_config_json(&config);
    let parsed: ModelConfig = serde_json::from_value(effective).unwrap();
    assert_eq!(parsed.hidden_size, 1152);
    assert_eq!(parsed.head_dim, Some(256));
    assert_eq!(parsed.query_pre_attn_scalar, Some(256.0));
    assert_eq!(
        parsed.hidden_activation.as_deref(),
        Some("gelu_pytorch_tanh")
    );
    assert!(!parsed.tie_word_embeddings);
    assert_eq!(parsed.eos_token_id, vec![1, 106]);
}

#[test]
fn normalized_gemma3_config_injects_hybrid_attention_defaults() {
    let raw = serde_json::json!({
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "quantization": {"group_size": 64, "bits": 4},
        "eos_token_id": [1, 106],
        "tie_word_embeddings": false,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0,
        "sliding_window": 512,
        "sliding_window_pattern": 3,
        "max_position_embeddings": 32768,
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 1152,
            "num_hidden_layers": 8,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 262144,
            "layer_types": null,
            "rope_parameters": null,
            "use_bidirectional_attention": null
        }
    });

    let normalized = normalized_model_config_json(&raw);
    let parsed: ModelConfig = serde_json::from_value(normalized.clone()).unwrap();

    assert_eq!(
        normalized
            .get("layer_types")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ]
    );
    assert_eq!(
        parsed
            .rope_parameters
            .as_ref()
            .and_then(|params| params.get("sliding_attention"))
            .and_then(|params| params.rope_theta),
        Some(10_000.0)
    );
    assert_eq!(
        parsed
            .rope_parameters
            .as_ref()
            .and_then(|params| params.get("full_attention"))
            .and_then(|params| params.rope_theta),
        Some(1_000_000.0)
    );
    assert_eq!(
        normalized
            .get("use_bidirectional_attention")
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        parsed.layer_types.as_ref().map(Vec::len),
        Some(parsed.num_hidden_layers as usize)
    );
}

#[test]
fn model_architecture_detects_gemma3_from_text_config() {
    let config = serde_json::json!({
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "text_config": {"model_type": "gemma3_text"}
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::Gemma3);
}

#[test]
fn model_architecture_detects_gemma2() {
    let config = serde_json::json!({
        "model_type": "gemma2",
        "architectures": ["Gemma2ForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::Gemma2);
}

#[test]
fn model_architecture_detects_glm4() {
    let config = serde_json::json!({
        "model_type": "glm4",
        "architectures": ["Glm4ForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::Glm4);
}

#[test]
fn model_architecture_detects_lfm2() {
    let config = serde_json::json!({
        "model_type": "lfm2",
        "architectures": ["Lfm2ForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::Lfm2);
}

#[test]
fn model_architecture_detects_olmo() {
    let config = serde_json::json!({
        "model_type": "olmo",
        "architectures": ["OlmoForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::Olmo);
}

#[test]
fn model_architecture_detects_deepseek_v3() {
    let config = serde_json::json!({
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::DeepseekV3);
}

#[test]
fn model_architecture_detects_gpt_oss() {
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::GptOss);
}

#[test]
fn model_architecture_detects_kimi_linear() {
    let config = serde_json::json!({
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::KimiLinear);
}

#[test]
fn model_architecture_detects_olmo2() {
    let config = serde_json::json!({
        "model_type": "olmo2",
        "architectures": ["Olmo2ForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::Olmo2);
}

#[test]
fn model_architecture_detects_kimi_k2_as_deepseek_v3_runtime() {
    let config = serde_json::json!({
        "model_type": "kimi_k25",
        "architectures": ["DeepseekV3ForCausalLM"]
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::DeepseekV3);
}

#[test]
fn glm4_config_parses_partial_rotary_factor() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "glm4",
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "intermediate_size": 13696,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 151552,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": false,
        "hidden_act": "silu",
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": 151329
    }))
    .unwrap();

    assert_eq!(config.partial_rotary_factor, Some(0.5));
    assert_eq!(config.head_dim, Some(128));
}

#[test]
fn deepseek_v3_config_parses_moe_and_mla_fields() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "n_group": 8,
        "topk_group": 4,
        "num_experts_per_tok": 8,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 3,
        "vocab_size": 129280,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840,
        "tie_word_embeddings": false,
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": [0, 1]
    }))
    .unwrap();

    assert_eq!(config.moe_intermediate_size, Some(2048));
    assert_eq!(config.n_routed_experts, Some(256));
    assert_eq!(config.kv_lora_rank, Some(512));
    assert_eq!(config.q_lora_rank, Some(1536));
    assert_eq!(config.qk_rope_head_dim, Some(64));
    assert_eq!(config.qk_nope_head_dim, Some(128));
    assert_eq!(config.v_head_dim, Some(128));
    assert_eq!(config.n_group, Some(8));
    assert_eq!(config.topk_group, Some(4));
    assert_eq!(config.num_experts_per_tok, Some(8));
    assert_eq!(config.first_k_dense_replace, Some(3));
}

#[test]
fn lfm2_config_parses_conv_and_attention_layout() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "lfm2",
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "intermediate_size": 6656,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 65536,
        "rms_norm_eps": 0.00001,
        "max_position_embeddings": 128000,
        "tie_word_embeddings": false,
        "rope_theta": 1000000.0,
        "conv_bias": false,
        "conv_L_cache": 3,
        "block_norm_eps": 0.00001,
        "block_dim": 1024,
        "block_ff_dim": 6656,
        "block_multiple_of": 256,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": true,
        "full_attn_idxs": [2, 5, 8, 10, 12, 14],
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": 7
    }))
    .unwrap();

    assert_eq!(config.conv_l_cache, Some(3));
    assert_eq!(
        config.full_attn_idxs.as_deref(),
        Some(&[2, 5, 8, 10, 12, 14][..])
    );
    assert_eq!(config.block_norm_eps, Some(0.00001));
}

#[test]
fn gemma2_config_parses_attention_softcaps() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 2304,
        "num_hidden_layers": 26,
        "intermediate_size": 9216,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "vocab_size": 256000,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": false,
        "hidden_activation": "gelu_pytorch_tanh",
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "sliding_window": 4096,
        "cache_implementation": "hybrid",
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": 1
    }))
    .unwrap();

    assert_eq!(config.attn_logit_softcapping, Some(50.0));
    assert_eq!(config.final_logit_softcapping, Some(30.0));
    assert_eq!(config.sliding_window, Some(4096));
    assert_eq!(config.cache_implementation.as_deref(), Some("hybrid"));
}

#[test]
fn gemma2_real_hf_config_parses() {
    let raw = serde_json::json!({
        "architectures": ["Gemma2ForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0,
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "eos_token_id": [1, 107],
        "final_logit_softcapping": 30.0,
        "head_dim": 256,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 2304,
        "initializer_range": 0.02,
        "intermediate_size": 9216,
        "max_position_embeddings": 8192,
        "model_type": "gemma2",
        "num_attention_heads": 8,
        "num_hidden_layers": 26,
        "num_key_value_heads": 4,
        "pad_token_id": 0,
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "query_pre_attn_scalar": 256,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "sliding_window": 4096,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.42.4",
        "use_cache": true,
        "vocab_size": 256000
    });
    let config: ModelConfig = serde_json::from_value(normalized_model_config_json(&raw)).unwrap();

    assert_eq!(config.eos_token_id, vec![1, 107]);
    assert_eq!(config.cache_implementation.as_deref(), Some("hybrid"));
    assert_eq!(
        config.hidden_activation.as_deref(),
        Some("gelu_pytorch_tanh")
    );
}

#[test]
fn effective_text_config_extracts_gemma4_text_config() {
    let config = serde_json::json!({
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "quantization": {"group_size": 64, "bits": 4},
        "eos_token_id": [1, 106],
        "tie_word_embeddings": false,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 32768,
        "rope_theta": 10000.0,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2560,
            "hidden_size_per_layer_input": 256,
            "num_hidden_layers": 42,
            "intermediate_size": 10240,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 18,
            "head_dim": 256,
            "global_head_dim": 512,
            "query_pre_attn_scalar": 256.0,
            "vocab_size": 262400,
            "vocab_size_per_layer_input": 128,
            "layer_types": ["sliding_attention", "full_attention"],
            "final_logit_softcapping": 30.0
        }
    });

    let effective = effective_text_config_json(&config);
    let parsed: ModelConfig = serde_json::from_value(effective).unwrap();
    assert_eq!(parsed.hidden_size, 2560);
    assert_eq!(parsed.hidden_size_per_layer_input, Some(256));
    assert_eq!(parsed.head_dim, Some(256));
    assert_eq!(parsed.global_head_dim, Some(512));
    assert_eq!(parsed.num_kv_shared_layers, Some(18));
    assert_eq!(parsed.vocab_size_per_layer_input, Some(128));
    assert_eq!(
        parsed.layer_types.as_deref(),
        Some(
            &[
                "sliding_attention".to_string(),
                "full_attention".to_string()
            ][..]
        )
    );
    assert_eq!(parsed.final_logit_softcapping, Some(30.0));
}

#[test]
fn model_architecture_detects_gemma4_from_text_config() {
    let config = serde_json::json!({
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"model_type": "gemma4_text"}
    });

    assert_eq!(model_architecture(&config), ModelArchitecture::Gemma4);
}

#[test]
fn qwen3_flat_rope_parameters_are_accepted() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": true,
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": 151645,
        "rope_theta": 1000000.0,
        "rope_parameters": {
            "rope_theta": 1000000.0,
            "rope_type": "default"
        }
    }))
    .unwrap();

    let params = config.rope_parameters.unwrap();
    assert_eq!(
        params.get("default").and_then(|p| p.rope_theta),
        Some(1000000.0)
    );
}

#[test]
fn qwen3_real_hf_config_parses_qk_norm_and_rope_scaling() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": true,
        "rope_theta": 1000000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        },
        "eos_token_id": 151645
    }))
    .unwrap();

    assert_eq!(
        reasoning_family(&serde_json::json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"]
        })),
        ReasoningFamily::Qwen3
    );
    assert_eq!(config.head_dim, Some(128));
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.rope_theta, 1000000.0);
}

#[test]
fn olmo2_real_hf_config_parses_qk_norm_style_fields() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "olmo2",
        "architectures": ["Olmo2ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "vocab_size": 50304,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": false,
        "attention_bias": false,
        "rope_theta": 10000.0,
        "eos_token_id": 50279
    }))
    .unwrap();

    assert_eq!(
        model_architecture(&serde_json::json!({
            "model_type": "olmo2",
            "architectures": ["Olmo2ForCausalLM"]
        })),
        ModelArchitecture::Olmo2
    );
    assert_eq!(config.head_dim, Some(128));
    assert!(!config.tie_word_embeddings);
}

#[test]
fn gpt_oss_real_hf_config_parses_sliding_window_layers() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "hidden_size": 2880,
        "num_hidden_layers": 24,
        "intermediate_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 201088,
        "rms_norm_eps": 0.00001,
        "rope_theta": 150000.0,
        "max_position_embeddings": 131072,
        "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
        "num_experts_per_tok": 4,
        "tie_word_embeddings": false,
        "eos_token_id": [199999, 200002]
    }))
    .unwrap();

    assert_eq!(
        model_architecture(&serde_json::json!({
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"]
        })),
        ModelArchitecture::GptOss
    );
    assert_eq!(
        reasoning_family(&serde_json::json!({
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"]
        })),
        ReasoningFamily::GptOss
    );
    assert_eq!(config.sliding_window, Some(128));
    assert_eq!(
        config.layer_types.as_deref(),
        Some(
            &[
                "sliding_attention".to_string(),
                "full_attention".to_string(),
                "sliding_attention".to_string()
            ][..]
        )
    );
    assert_eq!(config.num_experts_per_tok, Some(4));
    assert_eq!(config.eos_token_id, vec![199999, 200002]);
}

#[test]
fn gemma3_real_hf_config_parses_hybrid_cache_fields() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "gemma3_text",
        "architectures": ["Gemma3ForCausalLM"],
        "hidden_size": 1152,
        "num_hidden_layers": 26,
        "intermediate_size": 6912,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "vocab_size": 262144,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "sliding_window": 512,
        "sliding_window_pattern": 6,
        "cache_implementation": "hybrid",
        "tie_word_embeddings": false,
        "eos_token_id": [1, 106]
    }))
    .unwrap();

    assert_eq!(config.sliding_window, Some(512));
    assert_eq!(config.sliding_window_pattern, Some(6));
    assert_eq!(config.cache_implementation.as_deref(), Some("hybrid"));
}

#[test]
fn attention_window_size_for_gpt_oss_uses_layer_types() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 2880,
        "num_hidden_layers": 3,
        "intermediate_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 201088,
        "rms_norm_eps": 0.00001,
        "rope_theta": 150000.0,
        "max_position_embeddings": 131072,
        "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
        "tie_word_embeddings": false,
        "eos_token_id": [199999, 200002]
    }))
    .unwrap();

    assert_eq!(
        attention_window_size_for_layer(
            ModelArchitecture::GptOss,
            &config,
            0,
            Some("sliding_attention")
        )
        .unwrap(),
        Some(128)
    );
    assert_eq!(
        attention_window_size_for_layer(
            ModelArchitecture::GptOss,
            &config,
            1,
            Some("full_attention")
        )
        .unwrap(),
        None
    );
}

#[test]
fn attention_window_size_for_gemma3_matches_hybrid_pattern() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 1152,
        "num_hidden_layers": 8,
        "intermediate_size": 6912,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "vocab_size": 262144,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "sliding_window": 512,
        "sliding_window_pattern": 3,
        "cache_implementation": "hybrid",
        "tie_word_embeddings": false,
        "eos_token_id": [1, 106]
    }))
    .unwrap();

    assert_eq!(
        attention_window_size_for_layer(ModelArchitecture::Gemma3, &config, 0, None).unwrap(),
        Some(512)
    );
    assert_eq!(
        attention_window_size_for_layer(ModelArchitecture::Gemma3, &config, 1, None).unwrap(),
        Some(512)
    );
    assert_eq!(
        attention_window_size_for_layer(ModelArchitecture::Gemma3, &config, 2, None).unwrap(),
        None
    );
    assert_eq!(
        attention_window_size_for_layer(ModelArchitecture::Gemma3, &config, 3, None).unwrap(),
        Some(512)
    );
}

#[test]
fn attention_window_size_for_gemma4_uses_layer_types() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "intermediate_size": 10240,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_kv_shared_layers": 2,
            "vocab_size": 262400,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "tie_word_embeddings": false,
            "eos_token_id": [1, 106]
        }))
        .unwrap();

    assert_eq!(
        attention_window_size_for_layer(
            ModelArchitecture::Gemma4,
            &config,
            0,
            Some("sliding_attention")
        )
        .unwrap(),
        None
    );
    assert_eq!(
        attention_window_size_for_layer(
            ModelArchitecture::Gemma4,
            &config,
            1,
            Some("full_attention")
        )
        .unwrap(),
        None
    );
}

#[test]
fn kv_shared_source_for_gemma4_matches_previous_layer_type() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 2560,
        "num_hidden_layers": 6,
        "intermediate_size": 10240,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_kv_shared_layers": 2,
        "vocab_size": 262400,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 32768,
        "rope_theta": 10000.0,
        "sliding_window": 512,
        "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ],
        "tie_word_embeddings": false,
        "eos_token_id": [1, 106]
    }))
    .unwrap();

    let non_shared = &config.layer_types.as_ref().unwrap()[..4];
    assert_eq!(
        kv_shared_source_for_layer(
            ModelArchitecture::Gemma4,
            &config,
            4,
            Some("sliding_attention"),
            Some(non_shared)
        ),
        Some(2)
    );
    assert_eq!(
        kv_shared_source_for_layer(
            ModelArchitecture::Gemma4,
            &config,
            5,
            Some("full_attention"),
            Some(non_shared)
        ),
        Some(3)
    );
    assert_eq!(
        kv_shared_source_for_layer(
            ModelArchitecture::Gemma4,
            &config,
            1,
            Some("full_attention"),
            Some(non_shared)
        ),
        None
    );
}

#[test]
fn gemma3_uses_scaled_embeddings() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 1152,
        "num_hidden_layers": 26,
        "intermediate_size": 6912,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "vocab_size": 262144,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": null,
        "hidden_activation": "gelu_pytorch_tanh",
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": [1, 106]
    }))
    .unwrap();

    let embed_scale = (config.hidden_size as f32).sqrt();
    assert!((embed_scale - 33.941124).abs() < 0.001);
}

#[test]
fn gemma4_uses_scaled_main_and_per_layer_embeddings() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 2560,
        "hidden_size_per_layer_input": 256,
        "num_hidden_layers": 42,
        "intermediate_size": 10240,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_kv_shared_layers": 18,
        "vocab_size": 262400,
        "vocab_size_per_layer_input": 128,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": false,
        "query_pre_attn_scalar": 256.0,
        "rope_theta": 10000.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "quantization": {
            "group_size": 64,
            "bits": 4
        },
        "eos_token_id": [1, 106]
    }))
    .unwrap();

    let embed_scale = (config.hidden_size as f32).sqrt();
    let per_layer_scale = (config.hidden_size_per_layer_input.unwrap() as f32).sqrt();
    assert!((embed_scale - 50.596443).abs() < 0.001);
    assert!((per_layer_scale - 16.0).abs() < 0.001);
}

#[test]
fn quant_params_for_uses_tensor_specific_overrides() {
    let config = serde_json::json!({
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "language_model.model.embed_tokens": {"group_size": 64, "bits": 6},
            "language_model.model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 8}
        }
    });

    assert_eq!(
        quant_params_for(&config, "language_model.model.embed_tokens", 64, 4),
        (64, 6)
    );
    assert_eq!(
        quant_params_for(
            &config,
            "language_model.model.layers.0.self_attn.q_proj",
            64,
            4
        ),
        (64, 8)
    );
    assert_eq!(
        quant_params_for(
            &config,
            "language_model.model.layers.0.mlp.down_proj",
            64,
            4
        ),
        (64, 4)
    );
}

#[test]
fn dense_model_config_is_allowed_without_quantization_block() {
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 151936,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": true,
        "eos_token_id": 151645
    }))
    .unwrap();

    assert!(config.quantization.is_none());
    assert!(config.tie_word_embeddings);
}

#[test]
#[serial]
fn dense_embeddings_can_project_logits_through_as_linear() {
    let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let embedding = QuantizedEmbedding {
        weight: weight.clone(),
        scales: array!(0.0f32),
        biases: array!(0.0f32),
        group_size: 0,
        bits: 0,
        dense_weight: Some(weight.clone()),
        dense_weight_t: Some(weight.transpose_axes(&[1, 0]).unwrap()),
    };
    let hidden = Array::from_slice(&[10.0f32, 20.0], &[1, 1, 2]);

    let logits = embedding.as_linear().forward(&hidden).unwrap();

    assert_eq!(logits.as_slice::<f32>(), &[50.0, 110.0, 170.0]);
}

fn dense_linear(weight: &[f32], out_dim: i32, in_dim: i32) -> QuantizedLinear {
    let weight = Array::from_slice(weight, &[out_dim, in_dim]);
    QuantizedLinear {
        weight: weight.clone(),
        scales: array!(0.0f32),
        biases: array!(0.0f32),
        bias: None,
        group_size: 0,
        bits: 0,
        dense_weight_t: Some(weight.transpose_axes(&[1, 0]).unwrap()),
    }
}

fn identity_dense_linear(dim: i32) -> QuantizedLinear {
    let mut weight = vec![0.0f32; (dim * dim) as usize];
    for i in 0..dim as usize {
        weight[i * dim as usize + i] = 1.0;
    }
    dense_linear(&weight, dim, dim)
}

fn assert_arrays_close(actual: &Array, expected: &Array, tol: f32) {
    let actual = actual.as_dtype(Dtype::Float32).unwrap();
    let expected = expected.as_dtype(Dtype::Float32).unwrap();
    let actual_slice = actual.as_slice::<f32>();
    let expected_slice = expected.as_slice::<f32>();
    assert_eq!(actual_slice.len(), expected_slice.len());
    for (idx, (a, b)) in actual_slice.iter().zip(expected_slice.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "mismatch at index {idx}: actual={a} expected={b} tol={tol}"
        );
    }
}

#[test]
#[serial]
fn attention_kv_cache_matches_no_cache_for_incremental_decode() {
    let attn = Attention {
        q_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        k_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        v_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        o_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        q_norm: None,
        k_norm: None,
        v_norm: None,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        scale: 1.0 / (2.0f32).sqrt(),
        attn_logit_softcapping: None,
        rope_dim: 2,
        rope_theta: 10000.0,
        rope_traditional: false,
        window_size: None,
        kv_shared_source: None,
    };

    let full = Array::from_slice(&[1.0f32, 0.0, 0.5, 1.0, -1.0, 0.25, 0.75, -0.5], &[1, 4, 2]);
    let expected = attn.forward_no_cache(&full).unwrap();

    let mut cache = KVCache::new();
    let mut outputs = Vec::new();
    for step in 0..4i32 {
        let x = full.index((0..1, step..step + 1, std::ops::RangeFull));
        outputs.push(attn.forward(&x, &mut cache, None).unwrap());
    }
    let output_refs: Vec<&Array> = outputs.iter().collect();
    let actual = mlx_rs::ops::concatenate_axis(&output_refs, 1).unwrap();

    assert_eq!(cache.offset(), 4);
    assert_arrays_close(&actual, &expected, 1e-4);
}

#[test]
#[serial]
fn attention_quantized_kv_cache_stays_close_to_dense_cache() {
    let dim = 32i32;
    let attn = Attention {
        q_proj: identity_dense_linear(dim),
        k_proj: identity_dense_linear(dim),
        v_proj: identity_dense_linear(dim),
        o_proj: identity_dense_linear(dim),
        q_norm: None,
        k_norm: None,
        v_norm: None,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: dim,
        scale: 1.0 / (dim as f32).sqrt(),
        attn_logit_softcapping: None,
        rope_dim: dim,
        rope_theta: 10000.0,
        rope_traditional: false,
        window_size: None,
        kv_shared_source: None,
    };

    let values = (0..(4 * dim))
        .map(|i| (i as f32 * 0.03125) - 1.0)
        .collect::<Vec<_>>();
    let full = Array::from_slice(&values, &[1, 4, dim]);

    let mut dense_cache = KVCache::new();
    let mut dense_outputs = Vec::new();
    for step in 0..4i32 {
        let x = full.index((0..1, step..step + 1, std::ops::RangeFull));
        dense_outputs.push(attn.forward(&x, &mut dense_cache, None).unwrap());
    }
    let dense_output_refs: Vec<&Array> = dense_outputs.iter().collect();
    let dense_actual = mlx_rs::ops::concatenate_axis(&dense_output_refs, 1).unwrap();

    let mut quantized_cache = KVCache::new_quantized(32, 8, 0);
    let mut quantized_outputs = Vec::new();
    for step in 0..4i32 {
        let x = full.index((0..1, step..step + 1, std::ops::RangeFull));
        quantized_outputs.push(attn.forward(&x, &mut quantized_cache, None).unwrap());
    }
    let quantized_output_refs: Vec<&Array> = quantized_outputs.iter().collect();
    let quantized_actual = mlx_rs::ops::concatenate_axis(&quantized_output_refs, 1).unwrap();

    assert_eq!(quantized_cache.offset(), 4);
    assert_arrays_close(&quantized_actual, &dense_actual, 5e-2);
}

#[test]
#[serial]
fn attention_quantized_kv_cache_threshold_migrates_after_dense_prefix() {
    let dim = 32i32;
    let attn = Attention {
        q_proj: identity_dense_linear(dim),
        k_proj: identity_dense_linear(dim),
        v_proj: identity_dense_linear(dim),
        o_proj: identity_dense_linear(dim),
        q_norm: None,
        k_norm: None,
        v_norm: None,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: dim,
        scale: 1.0 / (dim as f32).sqrt(),
        attn_logit_softcapping: None,
        rope_dim: dim,
        rope_theta: 10000.0,
        rope_traditional: false,
        window_size: None,
        kv_shared_source: None,
    };

    let values = (0..(6 * dim))
        .map(|i| (((i as usize % dim as usize) as f32) / dim as f32) - 0.5)
        .collect::<Vec<_>>();
    let full = Array::from_slice(&values, &[1, 6, dim]);

    let mut dense_cache = KVCache::new();
    let mut dense_outputs = Vec::new();
    for step in 0..6i32 {
        let x = full.index((0..1, step..step + 1, std::ops::RangeFull));
        dense_outputs.push(attn.forward(&x, &mut dense_cache, None).unwrap());
    }
    let dense_output_refs: Vec<&Array> = dense_outputs.iter().collect();
    let dense_actual = mlx_rs::ops::concatenate_axis(&dense_output_refs, 1).unwrap();

    let mut quantized_cache = KVCache::new_quantized(32, 8, 4);
    let mut quantized_outputs = Vec::new();
    for step in 0..6i32 {
        let x = full.index((0..1, step..step + 1, std::ops::RangeFull));
        quantized_outputs.push(attn.forward(&x, &mut quantized_cache, None).unwrap());
    }
    let quantized_output_refs: Vec<&Array> = quantized_outputs.iter().collect();
    let quantized_actual = mlx_rs::ops::concatenate_axis(&quantized_output_refs, 1).unwrap();

    assert_eq!(quantized_cache.offset(), 6);
    assert!(quantized_cache.qkeys.is_some());
    assert!(quantized_cache.qvalues.is_some());
    assert!(quantized_cache.keys.is_none());
    assert!(quantized_cache.values.is_none());
    let dense_prefix = dense_actual.index((0..1, 0..4, std::ops::RangeFull));
    let quantized_prefix = quantized_actual.index((0..1, 0..4, std::ops::RangeFull));
    let dense_tail = dense_actual.index((0..1, 4..6, std::ops::RangeFull));
    let quantized_tail = quantized_actual.index((0..1, 4..6, std::ops::RangeFull));
    assert_arrays_close(&quantized_prefix, &dense_prefix, 1e-4);
    assert_arrays_close(&quantized_tail, &dense_tail, 2e-1);
}

#[test]
#[serial]
fn rotating_kv_cache_cannot_trim_before_retained_window() {
    let mut cache = KVCache::new_rotating(2, 0);
    let k = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
    let v = Array::from_slice(&[3.0f32, 4.0], &[1, 1, 1, 2]);

    cache.update(k.clone(), v.clone()).unwrap();
    cache.update(k.clone(), v.clone()).unwrap();
    cache.update_cached(k, v).unwrap();

    assert_eq!(cache.offset(), 3);
    assert_eq!(cache.retained_start(), 1);
    assert!(cache.can_trim_to(1));
    assert!(!cache.can_trim_to(0));
}

#[test]
#[serial]
fn rotating_kv_cache_rewind_and_append_preserves_temporal_order() {
    let mut cache = KVCache::new_rotating(3, 0);
    for token in [1.0f32, 2.0, 3.0] {
        let k = Array::from_slice(&[token], &[1, 1, 1, 1]);
        let v = Array::from_slice(&[token + 10.0], &[1, 1, 1, 1]);
        cache.update(k, v).unwrap();
    }

    assert!(cache.trim_to(2).unwrap());

    let k = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
    let v = Array::from_slice(&[19.0f32], &[1, 1, 1, 1]);
    let (keys, values) = cache.update(k, v).unwrap();

    assert_eq!(cache.offset(), 3);
    assert_eq!(cache.retained_start(), 0);
    assert_eq!(keys.as_slice::<f32>(), &[1.0, 2.0, 9.0]);
    assert_eq!(values.as_slice::<f32>(), &[11.0, 12.0, 19.0]);
}

#[test]
#[serial]
fn standard_kv_cache_trim_materializes_prefix() {
    let mut cache = KVCache::new();
    let k = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let v = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[1, 1, 2, 2]);

    cache.update_cached(k, v).unwrap();
    assert!(cache.trim_to(1).unwrap());

    assert_eq!(cache.offset(), 1);
    assert_eq!(cache.keys.as_ref().unwrap().shape(), &[1, 1, 1, 2]);
    assert_eq!(cache.values.as_ref().unwrap().shape(), &[1, 1, 1, 2]);
    let (keys, values) = cache.views().unwrap();
    assert_eq!(keys.as_slice::<f32>(), &[1.0, 2.0]);
    assert_eq!(values.as_slice::<f32>(), &[5.0, 6.0]);
}

#[test]
#[serial]
fn quantized_kv_cache_trim_materializes_prefix() {
    let mut cache = KVCache::new_quantized(64, 8, 0);
    let k_data: Vec<f32> = (0..(3 * 64)).map(|i| (i as f32 / 64.0) - 1.0).collect();
    let v_data: Vec<f32> = (0..(3 * 64)).map(|i| 1.0 - (i as f32 / 64.0)).collect();
    let k = Array::from_slice(&k_data, &[1, 1, 3, 64]);
    let v = Array::from_slice(&v_data, &[1, 1, 3, 64]);

    cache.update_cached(k, v).unwrap();
    assert!(cache.trim_to(2).unwrap());

    assert_eq!(cache.offset(), 2);
    assert_eq!(cache.qkeys.as_ref().unwrap().data.shape()[2], 2);
    assert_eq!(cache.qvalues.as_ref().unwrap().data.shape()[2], 2);
}

#[test]
#[serial]
fn attention_sliding_window_cache_matches_no_cache_for_incremental_decode() {
    let attn = Attention {
        q_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        k_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        v_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        o_proj: dense_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2),
        q_norm: None,
        k_norm: None,
        v_norm: None,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        scale: 1.0 / (2.0f32).sqrt(),
        attn_logit_softcapping: None,
        rope_dim: 2,
        rope_theta: 10000.0,
        rope_traditional: false,
        window_size: Some(2),
        kv_shared_source: None,
    };

    let full = Array::from_slice(&[1.0f32, 0.0, 0.5, 1.0, -1.0, 0.25, 0.75, -0.5], &[1, 4, 2]);
    let expected = attn.forward_no_cache(&full).unwrap();

    let mut cache = KVCache::new_rotating(2, 0);
    let mut outputs = Vec::new();
    for step in 0..4i32 {
        let x = full.index((0..1, step..step + 1, std::ops::RangeFull));
        outputs.push(attn.forward(&x, &mut cache, None).unwrap());
    }
    let output_refs: Vec<&Array> = outputs.iter().collect();
    let actual = mlx_rs::ops::concatenate_axis(&output_refs, 1).unwrap();

    assert_eq!(cache.offset(), 4);
    assert_arrays_close(&actual, &expected, 1e-4);
}

#[test]
#[serial]
fn phi3_tensor_transform_splits_fused_attention_and_mlp_weights() {
    let prefixes = TensorPrefixes {
        model: "model".to_string(),
        lm_head: Some("lm_head".to_string()),
    };
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 128,
        "tie_word_embeddings": false,
        "quantization": {
            "group_size": 2,
            "bits": 4
        },
        "eos_token_id": 1
    }))
    .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.weight".to_string(),
        Array::from_slice(&vec![0u32; 24 * 3], &[24, 3]),
    );
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.scales".to_string(),
        Array::from_slice(&vec![0.0f32; 24 * 2], &[24, 2]),
    );
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.biases".to_string(),
        Array::from_slice(&vec![0.0f32; 24 * 2], &[24, 2]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_up_proj.weight".to_string(),
        Array::from_slice(&vec![0u32; 24 * 3], &[24, 3]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_up_proj.scales".to_string(),
        Array::from_slice(&vec![0.0f32; 24 * 2], &[24, 2]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_up_proj.biases".to_string(),
        Array::from_slice(&vec![0.0f32; 24 * 2], &[24, 2]),
    );

    families::apply_family_tensor_transforms(
        ModelArchitecture::LlamaLike,
        &mut tensors,
        &prefixes,
        &config,
        &serde_json::json!({"model_type": "phi3"}),
        2,
        4,
    )
    .unwrap();

    assert_eq!(
        tensors["model.layers.0.self_attn.q_proj.weight"].shape(),
        &[8, 3]
    );
    assert_eq!(
        tensors["model.layers.0.self_attn.k_proj.weight"].shape(),
        &[8, 3]
    );
    assert_eq!(
        tensors["model.layers.0.self_attn.v_proj.weight"].shape(),
        &[8, 3]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.gate_proj.weight"].shape(),
        &[12, 3]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.up_proj.weight"].shape(),
        &[12, 3]
    );
}

#[test]
#[serial]
fn gpt_oss_tensor_transform_splits_interleaved_expert_gate_up_tensors() {
    let prefixes = TensorPrefixes {
        model: "model".to_string(),
        lm_head: Some("lm_head".to_string()),
    };
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 128,
        "tie_word_embeddings": false,
        "quantization": {
            "group_size": 2,
            "bits": 4
        },
        "eos_token_id": 1
    }))
    .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.mlp.experts.gate_up_proj.weight".to_string(),
        Array::from_slice(
            &[
                0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0,
            ],
            &[6, 2],
        ),
    );
    tensors.insert(
        "model.layers.0.mlp.experts.gate_up_proj.scales".to_string(),
        Array::from_slice(
            &[
                0.5f32, 1.5, 10.5, 11.5, 20.5, 21.5, 30.5, 31.5, 40.5, 41.5, 50.5, 51.5,
            ],
            &[1, 6, 2],
        ),
    );
    tensors.insert(
        "model.layers.0.mlp.experts.gate_up_proj_bias".to_string(),
        Array::from_slice(&[0.0f32, 10.0, 20.0, 30.0, 40.0, 50.0], &[1, 6]),
    );
    tensors.insert(
        "model.layers.0.mlp.experts.down_proj_bias".to_string(),
        Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]),
    );

    families::apply_family_tensor_transforms(
        ModelArchitecture::GptOss,
        &mut tensors,
        &prefixes,
        &config,
        &serde_json::json!({"model_type": "gpt_oss"}),
        2,
        4,
    )
    .unwrap();

    assert_eq!(
        tensors["model.layers.0.mlp.experts.gate_proj.weight"].as_slice::<f32>(),
        &[0.0, 1.0, 20.0, 21.0, 40.0, 41.0]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.experts.up_proj.weight"].as_slice::<f32>(),
        &[10.0, 11.0, 30.0, 31.0, 50.0, 51.0]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.experts.gate_proj.scales"].shape(),
        &[1, 3, 2]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.experts.up_proj.scales"].shape(),
        &[1, 3, 2]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.experts.gate_proj.biases"].as_slice::<f32>(),
        &[0.0, 20.0, 40.0]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.experts.up_proj.biases"].as_slice::<f32>(),
        &[10.0, 30.0, 50.0]
    );
    assert_eq!(
        tensors["model.layers.0.mlp.experts.down_proj.biases"].as_slice::<f32>(),
        &[1.0, 2.0, 3.0]
    );
}

#[test]
#[serial]
fn gemma3_tensor_transform_drops_multimodal_tensors_and_tied_lm_head() {
    let prefixes = TensorPrefixes {
        model: "language_model.model".to_string(),
        lm_head: Some("language_model.lm_head".to_string()),
    };
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 128,
        "tie_word_embeddings": true,
        "eos_token_id": [1, 106]
    }))
    .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "vision_tower.encoder.weight".to_string(),
        Array::from_slice(&[1.0f32, 2.0], &[2]),
    );
    tensors.insert(
        "multi_modal_projector.linear.weight".to_string(),
        Array::from_slice(&[3.0f32, 4.0], &[2]),
    );
    tensors.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        Array::from_slice(&[5.0f32, 6.0], &[2]),
    );
    tensors.insert(
        "language_model.lm_head.weight".to_string(),
        Array::from_slice(&[7.0f32, 8.0], &[2]),
    );

    families::apply_family_tensor_transforms(
        ModelArchitecture::Gemma3,
        &mut tensors,
        &prefixes,
        &config,
        &serde_json::json!({"model_type": "gemma3", "text_config": {"model_type": "gemma3_text"}}),
        64,
        4,
    )
    .unwrap();

    assert!(tensors.contains_key("language_model.model.embed_tokens.weight"));
    assert!(!tensors.contains_key("vision_tower.encoder.weight"));
    assert!(!tensors.contains_key("multi_modal_projector.linear.weight"));
    assert!(!tensors.contains_key("language_model.lm_head.weight"));
}

#[test]
#[serial]
fn gemma4_tensor_transform_normalizes_text_prefixes_and_drops_multimodal_tensors() {
    let prefixes = TensorPrefixes {
        model: "language_model.model".to_string(),
        lm_head: Some("lm_head".to_string()),
    };
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 2560,
        "num_hidden_layers": 2,
        "intermediate_size": 10240,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_kv_shared_layers": 1,
        "vocab_size": 262400,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": true,
        "eos_token_id": [1, 106]
    }))
    .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.language_model.embed_tokens.weight".to_string(),
        Array::from_slice(&[1.0f32, 2.0], &[2]),
    );
    tensors.insert(
        "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
        Array::from_slice(&[3.0f32, 4.0], &[2]),
    );
    tensors.insert(
        "model.vision_tower.encoder.weight".to_string(),
        Array::from_slice(&[5.0f32, 6.0], &[2]),
    );
    tensors.insert(
        "model.audio_tower.encoder.weight".to_string(),
        Array::from_slice(&[7.0f32, 8.0], &[2]),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Array::from_slice(&[9.0f32, 10.0], &[2]),
    );

    families::apply_family_tensor_transforms(
        ModelArchitecture::Gemma4,
        &mut tensors,
        &prefixes,
        &config,
        &serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}),
        64,
        4,
    )
    .unwrap();

    assert!(tensors.contains_key("language_model.model.embed_tokens.weight"));
    assert!(tensors.contains_key("language_model.model.layers.0.self_attn.q_proj.weight"));
    assert!(!tensors.contains_key("model.language_model.embed_tokens.weight"));
    assert!(!tensors.contains_key("model.vision_tower.encoder.weight"));
    assert!(!tensors.contains_key("model.audio_tower.encoder.weight"));
    assert!(!tensors.contains_key("lm_head.weight"));
}

#[test]
#[serial]
fn olmo2_tensor_transform_drops_rotary_inv_freq_tensors() {
    let prefixes = TensorPrefixes {
        model: "model".to_string(),
        lm_head: Some("lm_head".to_string()),
    };
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 128,
        "tie_word_embeddings": false,
        "eos_token_id": 1
    }))
    .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
        Array::from_slice(&[1.0f32, 2.0], &[2]),
    );
    tensors.insert(
        "model.layers.1.self_attn.rotary_emb.inv_freq".to_string(),
        Array::from_slice(&[3.0f32, 4.0], &[2]),
    );
    tensors.insert(
        "model.layers.1.self_attn.q_proj.weight".to_string(),
        Array::from_slice(&[5.0f32, 6.0], &[2]),
    );

    families::apply_family_tensor_transforms(
        ModelArchitecture::Olmo2,
        &mut tensors,
        &prefixes,
        &config,
        &serde_json::json!({"model_type": "olmo2"}),
        64,
        4,
    )
    .unwrap();

    assert!(!tensors.contains_key("model.layers.0.self_attn.rotary_emb.inv_freq"));
    assert!(!tensors.contains_key("model.layers.1.self_attn.rotary_emb.inv_freq"));
    assert!(tensors.contains_key("model.layers.1.self_attn.q_proj.weight"));
}

#[test]
#[serial]
fn llama_like_tensor_transform_drops_inv_freq_and_tied_lm_head() {
    let prefixes = TensorPrefixes {
        model: "model".to_string(),
        lm_head: Some("lm_head".to_string()),
    };
    let config: ModelConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 128,
        "tie_word_embeddings": true,
        "eos_token_id": 1
    }))
    .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
        Array::from_slice(&[1.0f32, 2.0], &[2]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        Array::from_slice(&[3.0f32, 4.0], &[2]),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Array::from_slice(&[5.0f32, 6.0], &[2]),
    );

    families::apply_family_tensor_transforms(
        ModelArchitecture::LlamaLike,
        &mut tensors,
        &prefixes,
        &config,
        &serde_json::json!({"model_type": "llama"}),
        64,
        4,
    )
    .unwrap();

    assert!(!tensors.contains_key("model.layers.0.self_attn.rotary_emb.inv_freq"));
    assert!(!tensors.contains_key("lm_head.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
}

#[test]
#[ignore]
fn olmo_debug_cache_vs_no_cache_local() {
    let dir = std::path::Path::new(
            "/Users/jdumay/.cache/mesh-llm-debug/olmo-7b-instruct-hf-same-origin/mlx/olmo-7b-instruct-hf-bf16",
        );
    assert!(
        dir.exists(),
        "missing local OLMo artifact at {}",
        dir.display()
    );

    let model = MlxModel::load(dir).expect("load local olmo mlx artifact");
    let prompt =
        "<|endoftext|><|user|>\nWhat day comes after Monday? Reply with one word.\n<|assistant|>\n";
    let encoded = model
        .tokenizer
        .encode(prompt, false)
        .expect("tokenize prompt");
    let ids = encoded.get_ids().to_vec();
    let input = Array::from_slice(&ids, &[1, ids.len() as i32]);

    let h = model.embed_tokens.forward(&input).expect("embed");
    let ln = model.layers[0]
        .attn_in_norm
        .as_ref()
        .expect("attn_in_norm")
        .forward(&h)
        .expect("ln");
    let (q, k, v, q_rope, k_rope, attn_out, h, mlp_in, mlp, layer0_out) = match &model.layers[0] {
        Layer {
            attn: AttentionKind::Standard(attn),
            mlp,
            mlp_in_norm,
            ..
        } => {
            let shape = ln.shape();
            let (b, l) = (shape[0], shape[1]);

            let q = attn.q_proj.forward(&ln).expect("q_proj");
            let q = Attention::apply_qk_norm(
                q,
                attn.q_norm.as_ref(),
                b,
                l,
                attn.num_heads,
                attn.head_dim,
            )
            .expect("q norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("q transpose");
            let q_rope = apply_rope(
                &q,
                attn.rope_dim,
                attn.head_dim,
                attn.rope_theta,
                attn.rope_traditional,
                0,
            )
            .expect("q rope");

            let k = attn.k_proj.forward(&ln).expect("k_proj");
            let v = attn.v_proj.forward(&ln).expect("v_proj");
            let k = Attention::apply_qk_norm(
                k,
                attn.k_norm.as_ref(),
                b,
                l,
                attn.num_kv_heads,
                attn.head_dim,
            )
            .expect("k norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("k transpose");
            let v = v
                .reshape(&[b, l, attn.num_kv_heads, attn.head_dim])
                .expect("v reshape");
            let v = if let Some(norm) = &attn.v_norm {
                norm.forward(&v).expect("v norm")
            } else {
                v
            }
            .transpose_axes(&[0, 2, 1, 3])
            .expect("v transpose");
            let k_rope = apply_rope(
                &k,
                attn.rope_dim,
                attn.head_dim,
                attn.rope_theta,
                attn.rope_traditional,
                0,
            )
            .expect("k rope");

            let mask = if l > 1 {
                Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
            } else {
                None
            };
            let attn_out =
                mlx_rs::fast::scaled_dot_product_attention(&q_rope, &k_rope, &v, attn.scale, mask)
                    .expect("attn");
            let attn_out = attn_out
                .transpose_axes(&[0, 2, 1, 3])
                .expect("attn transpose")
                .reshape(&[b, l, attn.num_heads * attn.head_dim])
                .expect("attn reshape");
            let attn_out = attn.o_proj.forward(&attn_out).expect("o_proj");

            let h = &attn_out + &h;
            let mlp_in = if let Some(norm) = mlp_in_norm {
                norm.forward(&h).expect("mlp in norm")
            } else {
                h.clone()
            };
            let mlp = mlp.forward(&mlp_in).expect("mlp");
            let layer0_out = &mlp + &h;

            (
                q, k, v, q_rope, k_rope, attn_out, h, mlp_in, mlp, layer0_out,
            )
        }
        _ => panic!("expected standard attention"),
    };

    let embed_last = h
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("embed cast");
    let ln_last = ln
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("ln cast");
    let q_last = q
        .index((0, 0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("q cast");
    let k_last = k
        .index((0, 0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("k cast");
    let v_last = v
        .index((0, 0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("v cast");
    let q_rope_last = q_rope
        .index((0, 0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("q rope cast");
    let k_rope_last = k_rope
        .index((0, 0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("k rope cast");
    let attn_out_last = attn_out
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("attn_out cast");
    let h_last = h
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("h cast");
    let mlp_in_last = mlp_in
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("mlp_in cast");
    let mlp_last = mlp
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("mlp cast");
    let layer0_out_last = layer0_out
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("layer0_out cast");
    mlx_rs::transforms::eval([
        &embed_last,
        &ln_last,
        &q_last,
        &k_last,
        &v_last,
        &q_rope_last,
        &k_rope_last,
        &attn_out_last,
        &h_last,
        &mlp_in_last,
        &mlp_last,
        &layer0_out_last,
    ])
    .expect("eval debug slices");
    println!("embed {:?}", embed_last.as_slice::<f32>());
    println!("ln0 {:?}", ln_last.as_slice::<f32>());
    println!("q0 {:?}", q_last.as_slice::<f32>());
    println!("k0 {:?}", k_last.as_slice::<f32>());
    println!("v0 {:?}", v_last.as_slice::<f32>());
    println!("qrope0 {:?}", q_rope_last.as_slice::<f32>());
    println!("krope0 {:?}", k_rope_last.as_slice::<f32>());
    println!("attn_out0 {:?}", attn_out_last.as_slice::<f32>());
    println!("h0 {:?}", h_last.as_slice::<f32>());
    println!("mlp_in0 {:?}", mlp_in_last.as_slice::<f32>());
    println!("mlp0 {:?}", mlp_last.as_slice::<f32>());
    println!("layer0_out {:?}", layer0_out_last.as_slice::<f32>());

    let mut h_all = model.embed_tokens.forward(&input).expect("embed all");
    for (i, layer) in model.layers.iter().enumerate() {
        h_all = layer.forward_no_cache(&h_all, None).expect("layer forward");
        let slice = h_all
            .index((0, (ids.len() as i32 - 1), 0..4))
            .as_dtype(Dtype::Float32)
            .expect("layer slice");
        mlx_rs::transforms::eval([&slice]).expect("eval layer slice");
        println!("layer{idx}_h {:?}", slice.as_slice::<f32>(), idx = i);
    }

    let h_norm = model.norm.forward(&h_all).expect("final norm");
    let h_norm_last = h_norm
        .index((0, (ids.len() as i32 - 1), 0..8))
        .as_dtype(Dtype::Float32)
        .expect("norm cast");
    let logits = if let Some(lm_head) = &model.lm_head {
        lm_head.forward(&h_norm).expect("lm head")
    } else {
        model
            .embed_tokens
            .as_linear()
            .forward(&h_norm)
            .expect("tied lm head")
    };
    let h_norm_f32 = h_norm.as_dtype(Dtype::Float32).expect("norm f32");
    let logits_f32 = if let Some(lm_head) = &model.lm_head {
        lm_head.forward(&h_norm_f32).expect("lm head f32")
    } else {
        model
            .embed_tokens
            .as_linear()
            .forward(&h_norm_f32)
            .expect("tied lm head f32")
    };
    let logits_last = logits
        .index((0, (ids.len() as i32 - 1), std::ops::RangeFull))
        .as_dtype(Dtype::Float32)
        .expect("logits cast");
    let logits_f32_last = logits_f32
        .index((0, (ids.len() as i32 - 1), std::ops::RangeFull))
        .as_dtype(Dtype::Float32)
        .expect("logits f32 cast");
    mlx_rs::transforms::eval([&h_norm_last, &logits_last, &logits_f32_last])
        .expect("eval final outputs");
    let logits_slice = logits_last.as_slice::<f32>();
    let mut pairs: Vec<(usize, f32)> = logits_slice.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<(usize, f32, String)> = pairs
        .into_iter()
        .take(5)
        .map(|(idx, val)| {
            (
                idx,
                val,
                model.tokenizer.id_to_token(idx as u32).unwrap_or_default(),
            )
        })
        .collect();
    let logits_f32_slice = logits_f32_last.as_slice::<f32>();
    let mut pairs_f32: Vec<(usize, f32)> = logits_f32_slice.iter().copied().enumerate().collect();
    pairs_f32.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5_f32: Vec<(usize, f32, String)> = pairs_f32
        .into_iter()
        .take(5)
        .map(|(idx, val)| {
            (
                idx,
                val,
                model.tokenizer.id_to_token(idx as u32).unwrap_or_default(),
            )
        })
        .collect();
    println!("final_norm {:?}", h_norm_last.as_slice::<f32>());
    println!("top5 {:?}", top5);
    println!("top5_f32_norm {:?}", top5_f32);

    let no_cache_logits = model.forward_no_cache(&input).expect("no-cache forward");
    let mut caches = model.new_caches();
    let cache_logits = model.forward(&input, &mut caches).expect("cache forward");

    let no_cache_last = no_cache_logits
        .index((0, (ids.len() as i32 - 1), std::ops::RangeFull))
        .as_dtype(Dtype::Float32)
        .expect("no-cache logits cast");
    let cache_last = cache_logits
        .index((0, (ids.len() as i32 - 1), std::ops::RangeFull))
        .as_dtype(Dtype::Float32)
        .expect("cache logits cast");
    mlx_rs::transforms::eval([&no_cache_last, &cache_last]).expect("eval logits slices");
    let describe_top = |name: &str, logits_slice: &[f32]| {
        let mut pairs: Vec<(usize, f32)> = logits_slice.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top10: Vec<(usize, f32, String)> = pairs
            .into_iter()
            .take(10)
            .map(|(idx, val)| {
                (
                    idx,
                    val,
                    model.tokenizer.id_to_token(idx as u32).unwrap_or_default(),
                )
            })
            .collect();
        println!("{name} top10 {:?}", top10);
    };
    describe_top("gemma3 no_cache", no_cache_last.as_slice::<f32>());
    describe_top("gemma3 cache", cache_last.as_slice::<f32>());

    let no_cache_token = argmax_last(&no_cache_logits).expect("argmax no-cache");
    let cache_token = argmax_last(&cache_logits).expect("argmax cache");
    let no_cache_piece = model
        .tokenizer
        .id_to_token(no_cache_token)
        .unwrap_or_else(|| "<missing>".to_string());
    let cache_piece = model
        .tokenizer
        .id_to_token(cache_token)
        .unwrap_or_else(|| "<missing>".to_string());

    println!(
        "no_cache_token={} piece={:?} cache_token={} piece={:?}",
        no_cache_token, no_cache_piece, cache_token, cache_piece
    );

    assert_eq!(no_cache_token, cache_token);
}
