use super::family::{model_architecture, ModelArchitecture};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    #[allow(dead_code)]
    #[serde(default)]
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    pub vocab_size: i32,
    #[serde(default)]
    #[allow(dead_code)]
    pub vocab_size_per_layer_input: Option<i32>,
    #[serde(alias = "norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[allow(dead_code)]
    #[serde(alias = "model_max_length")]
    pub max_position_embeddings: i32,
    #[serde(default, deserialize_with = "deserialize_nullable_bool")]
    pub tie_word_embeddings: bool,
    #[serde(default, alias = "hidden_act")]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default, alias = "num_shared_experts")]
    pub n_shared_experts: Option<i32>,
    #[serde(default, alias = "num_experts")]
    pub n_routed_experts: Option<i32>,
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default)]
    pub kv_lora_rank: Option<i32>,
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<i32>,
    #[serde(default)]
    pub v_head_dim: Option<i32>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub topk_method: Option<String>,
    #[serde(default, alias = "moe_renormalize")]
    pub norm_topk_prob: Option<bool>,
    #[serde(default, alias = "num_expert_group")]
    pub n_group: Option<i32>,
    #[serde(default)]
    pub topk_group: Option<i32>,
    #[serde(default, alias = "num_experts_per_token")]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub num_local_experts: Option<i32>,
    #[serde(default)]
    pub moe_layer_freq: Option<i32>,
    #[serde(default)]
    pub first_k_dense_replace: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<i32>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_rope_parameters")]
    pub rope_parameters: Option<HashMap<String, RopeParameters>>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub sliding_window_pattern: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub cache_implementation: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub conv_bias: Option<bool>,
    #[serde(default, alias = "conv_L_cache")]
    pub conv_l_cache: Option<i32>,
    #[serde(default)]
    pub block_norm_eps: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_dim: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_ff_dim: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_multiple_of: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_ffn_dim_multiplier: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub block_auto_adjust_ff_dim: Option<bool>,
    #[serde(default)]
    pub full_attn_idxs: Option<Vec<i32>>,
    #[serde(default)]
    pub linear_attn_config: Option<LinearAttnConfig>,
    #[serde(default)]
    #[allow(dead_code)]
    pub moe_router_activation_func: Option<String>,
    pub quantization: Option<super::QuantConfig>,
    #[serde(default, deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
}

#[derive(Debug, serde::Deserialize, Clone)]
pub struct LinearAttnConfig {
    #[allow(dead_code)]
    pub full_attn_layers: Vec<i32>,
    pub kda_layers: Vec<i32>,
    pub num_heads: i32,
    pub head_dim: i32,
    #[serde(default)]
    pub short_conv_kernel_size: Option<i32>,
}

#[derive(Debug, serde::Deserialize, Clone)]
pub struct RopeParameters {
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

pub(super) fn deserialize_nullable_bool<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<bool, D::Error> {
    use serde::Deserialize;
    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or(false))
}

pub(super) fn deserialize_eos_token_id<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<u32>, D::Error> {
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum EosId {
        Single(u32),
        Multiple(Vec<u32>),
    }
    Ok(match EosId::deserialize(deserializer)? {
        EosId::Single(id) => vec![id],
        EosId::Multiple(ids) => ids,
    })
}

pub(super) fn deserialize_rope_parameters<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<HashMap<String, RopeParameters>>, D::Error> {
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RopeParametersField {
        PerLayer(HashMap<String, RopeParameters>),
        Flat(RopeParameters),
    }

    Ok(
        match Option::<RopeParametersField>::deserialize(deserializer)? {
            None => None,
            Some(RopeParametersField::PerLayer(map)) => Some(map),
            Some(RopeParametersField::Flat(params)) => {
                let mut map = HashMap::new();
                map.insert("default".to_string(), params);
                Some(map)
            }
        },
    )
}

pub(super) fn default_rope_theta() -> f32 {
    10_000.0
}

pub(super) fn effective_text_config_json(config: &Value) -> Value {
    let Some(text_config) = config
        .get("text_config")
        .and_then(|value| value.as_object())
    else {
        return config.clone();
    };

    let mut merged = serde_json::Map::new();
    for (key, value) in text_config {
        merged.insert(key.clone(), value.clone());
    }
    for key in [
        "quantization",
        "eos_token_id",
        "rope_theta",
        "rms_norm_eps",
        "head_dim",
        "max_position_embeddings",
        "tie_word_embeddings",
        "hidden_activation",
        "query_pre_attn_scalar",
        "global_head_dim",
        "vocab_size_per_layer_input",
        "vocab_size",
        "hidden_size_per_layer_input",
        "moe_intermediate_size",
        "n_shared_experts",
        "n_routed_experts",
        "routed_scaling_factor",
        "kv_lora_rank",
        "q_lora_rank",
        "qk_rope_head_dim",
        "v_head_dim",
        "qk_nope_head_dim",
        "topk_method",
        "norm_topk_prob",
        "n_group",
        "topk_group",
        "num_experts_per_tok",
        "moe_layer_freq",
        "first_k_dense_replace",
        "attention_bias",
        "num_kv_shared_layers",
        "layer_types",
        "rope_parameters",
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "sliding_window",
        "sliding_window_pattern",
        "cache_implementation",
        "conv_bias",
        "conv_L_cache",
        "block_dim",
        "block_ff_dim",
        "block_multiple_of",
        "block_ffn_dim_multiplier",
        "block_auto_adjust_ff_dim",
        "full_attn_idxs",
    ] {
        if !merged.contains_key(key) || merged.get(key).is_some_and(Value::is_null) {
            if let Some(value) = config.get(key) {
                merged.insert(key.to_string(), value.clone());
            }
        }
    }
    if !merged.contains_key("architectures") {
        if let Some(value) = config.get("architectures") {
            merged.insert("architectures".to_string(), value.clone());
        }
    }

    Value::Object(merged)
}

pub(super) fn normalized_model_config_json(config: &Value) -> Value {
    let mut normalized = effective_text_config_json(config);
    let Some(object) = normalized.as_object_mut() else {
        return normalized;
    };

    if !object.contains_key("hidden_activation") {
        if let Some(value) = object.get("hidden_act").cloned() {
            object.insert("hidden_activation".to_string(), value);
        }
    }
    object.remove("hidden_act");

    if model_architecture(config).is_gemma3() {
        let sliding_window_pattern = object
            .get("sliding_window_pattern")
            .and_then(|value| value.as_i64())
            .unwrap_or(6);
        if object.get("layer_types").is_none_or(Value::is_null) {
            if let Some(num_hidden_layers) = object
                .get("num_hidden_layers")
                .and_then(|value| value.as_i64())
            {
                let layer_types = (0..num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % sliding_window_pattern != 0 {
                            Value::String("sliding_attention".to_string())
                        } else {
                            Value::String("full_attention".to_string())
                        }
                    })
                    .collect::<Vec<_>>();
                object.insert("layer_types".to_string(), Value::Array(layer_types));
            }
        }

        if object.get("rope_parameters").is_none_or(Value::is_null) {
            let full_theta = object
                .get("rope_theta")
                .and_then(|value| value.as_f64())
                .unwrap_or(1_000_000.0);
            let sliding_theta = object
                .get("rope_local_base_freq")
                .and_then(|value| value.as_f64())
                .unwrap_or(10_000.0);
            object.insert(
                "rope_parameters".to_string(),
                serde_json::json!({
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": sliding_theta
                    },
                    "full_attention": {
                        "rope_type": "default",
                        "rope_theta": full_theta
                    }
                }),
            );
        }

        if object
            .get("use_bidirectional_attention")
            .is_none_or(Value::is_null)
        {
            object.insert(
                "use_bidirectional_attention".to_string(),
                Value::Bool(false),
            );
        }
    }

    normalized
}

pub(super) fn attention_window_size_for_layer(
    arch: ModelArchitecture,
    config: &ModelConfig,
    layer_idx: usize,
    layer_type: Option<&str>,
) -> Result<Option<i32>> {
    if arch.is_gpt_oss() {
        return if matches!(layer_type, Some("sliding_attention")) {
            Ok(Some(config.sliding_window.context(
                "missing sliding_window for gpt-oss sliding layer",
            )?))
        } else {
            Ok(None)
        };
    }

    if arch.is_gemma3() {
        let pattern = config.sliding_window_pattern.unwrap_or(1);
        return if pattern > 1 && (layer_idx as i32 % pattern) != (pattern - 1) {
            Ok(Some(config.sliding_window.context(
                "missing sliding_window for gemma3 sliding layer",
            )?))
        } else {
            Ok(None)
        };
    }

    Ok(None)
}

pub(super) fn kv_shared_source_for_layer(
    arch: ModelArchitecture,
    config: &ModelConfig,
    layer_idx: usize,
    layer_type: Option<&str>,
    non_shared_layer_types: Option<&[String]>,
) -> Option<usize> {
    if !arch.is_gemma4() {
        return None;
    }

    let first_kv_shared_layer_idx = config
        .num_kv_shared_layers
        .map(|n| (config.num_hidden_layers - n) as usize)
        .unwrap_or(config.num_hidden_layers as usize);

    if layer_idx < first_kv_shared_layer_idx {
        return None;
    }

    non_shared_layer_types.and_then(|types| {
        layer_type.and_then(|current| {
            types
                .iter()
                .rposition(|candidate| candidate == current)
                .map(|index| index)
        })
    })
}

pub(super) fn experimental_quantized_kv_config() -> Option<(i32, i32, usize)> {
    let bits = std::env::var("MESH_LLM_MLX_QUANTIZED_KV_BITS")
        .ok()?
        .parse::<i32>()
        .ok()?;
    if bits <= 0 {
        return None;
    }
    let group_size = std::env::var("MESH_LLM_MLX_QUANTIZED_KV_GROUP_SIZE")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(64);
    let min_dense_tokens = std::env::var("MESH_LLM_MLX_QUANTIZED_KV_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256);
    Some((group_size, bits, min_dense_tokens))
}
