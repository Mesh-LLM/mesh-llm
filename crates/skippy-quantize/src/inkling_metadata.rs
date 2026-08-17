use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::gguf_writer::GgufKv;
use crate::tokenizer_metadata::push_tokenizer_metadata;

const ARCH: &str = "inkling";

pub(crate) fn is_inkling_config(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some("inkling_mm_model")
}

pub(crate) fn mtp_layer_start(config: &Value) -> Result<Option<u32>> {
    if !is_inkling_config(config) {
        return Ok(None);
    }
    let nextn = nested_u32(config, &["mtp_config", "num_nextn_predict_layers"]).unwrap_or_default();
    if nextn == 0 {
        return Ok(None);
    }
    required_nested_u32(config, &["text_config", "num_hidden_layers"]).map(Some)
}

pub(crate) fn metadata(
    source: &Path,
    tensor_count: usize,
    config: &Value,
    include_mtp: bool,
) -> Result<Vec<GgufKv>> {
    let text = config
        .get("text_config")
        .context("Inkling config missing text_config")?;
    validate_supported_variant(text, config)?;
    let trunk_layers = required_u32(text, "num_hidden_layers")?;
    let mtp_layers = if include_mtp {
        nested_u32(config, &["mtp_config", "num_nextn_predict_layers"]).unwrap_or_default()
    } else {
        0
    };
    let local_flags = local_layer_flags(text, config, trunk_layers, mtp_layers)?;
    let local_kv_heads = required_u32(text, "swa_num_key_value_heads")?;
    let global_kv_heads = required_u32(text, "num_key_value_heads")?;
    let head_count_kv = local_flags
        .iter()
        .map(|local| {
            if *local {
                local_kv_heads
            } else {
                global_kv_heads
            }
        })
        .collect();
    let context_length = required_u32(text, "model_max_length")?;
    let sliding_window = required_u32(text, "sliding_window_size")?;
    let head_dim = required_u32(text, "head_dim")?;

    let mut result = vec![
        GgufKv::string("general.architecture", ARCH),
        GgufKv::string("general.name", model_name(source)),
        GgufKv::bool("skippy.convert.raw_safetensors", false),
        GgufKv::u64("skippy.convert.tensor_count", tensor_count as u64),
        GgufKv::u32("inkling.vocab_size", required_u32(text, "vocab_size")?),
        GgufKv::u32("inkling.context_length", context_length),
        GgufKv::u32(
            "inkling.embedding_length",
            required_u32(text, "hidden_size")?,
        ),
        GgufKv::u32("inkling.block_count", trunk_layers + mtp_layers),
        GgufKv::u32(
            "inkling.feed_forward_length",
            required_u32(text, "dense_intermediate_size")?,
        ),
        GgufKv::f32(
            "inkling.attention.layer_norm_rms_epsilon",
            required_f32(text, "rms_norm_eps")?,
        ),
        GgufKv::u32(
            "inkling.attention.head_count",
            required_u32(text, "num_attention_heads")?,
        ),
        GgufKv::array_u32("inkling.attention.head_count_kv", head_count_kv),
        GgufKv::u32("inkling.attention.key_length", head_dim),
        GgufKv::u32("inkling.attention.value_length", head_dim),
        GgufKv::u32("inkling.attention.sliding_window", sliding_window),
        GgufKv::array_bool("inkling.attention.sliding_window_pattern", local_flags),
        GgufKv::u32(
            "inkling.expert_count",
            required_u32(text, "n_routed_experts")?,
        ),
        GgufKv::u32(
            "inkling.expert_used_count",
            required_u32(text, "num_experts_per_tok")?,
        ),
        GgufKv::u32(
            "inkling.expert_shared_count",
            required_u32(text, "n_shared_experts")?,
        ),
        GgufKv::u32(
            "inkling.expert_feed_forward_length",
            required_u32(text, "intermediate_size")?,
        ),
        GgufKv::f32(
            "inkling.expert_weights_scale",
            required_f32(text, "route_scale")?,
        ),
        GgufKv::u32("inkling.expert_gating_func", 2),
        GgufKv::u32("inkling.d_rel", required_u32(text, "d_rel")?),
        GgufKv::u32("inkling.rel_extent", required_u32(text, "rel_extent")?),
        GgufKv::u32("inkling.rel_extent_swa", sliding_window),
        GgufKv::u32(
            "inkling.shortconv_kernel",
            required_u32(text, "sconv_kernel_size")?,
        ),
        GgufKv::u32(
            "inkling.dense_block_count",
            required_u32(text, "dense_mlp_idx")?,
        ),
        GgufKv::f32(
            "inkling.logit_scale_denom",
            required_f32(text, "logits_mup_width_multiplier")?,
        ),
        GgufKv::u32(
            "inkling.log_scaling_n_floor",
            required_u32(text, "log_scaling_n_floor")?,
        ),
        GgufKv::f32(
            "inkling.log_scaling_alpha",
            required_f32(text, "log_scaling_alpha")?,
        ),
        GgufKv::u32(
            "inkling.unpadded_vocab_size",
            required_u32(text, "unpadded_vocab_size")?,
        ),
    ];
    if mtp_layers > 0 {
        result.push(GgufKv::u32("inkling.nextn_predict_layers", mtp_layers));
    }
    push_tokenizer_metadata(&mut result, source, config)?;
    Ok(result)
}

fn validate_supported_variant(text: &Value, config: &Value) -> Result<()> {
    for key in [
        "norm_after_topk",
        "shared_expert_sink",
        "use_sconv",
        "use_embed_norm",
        "use_gate_bias",
        "use_global_scale",
    ] {
        ensure!(
            text.get(key).and_then(Value::as_bool) == Some(true),
            "unsupported Inkling {key}; native conversion requires true"
        );
    }
    ensure!(
        text.get("gate_activation").and_then(Value::as_str) == Some("sigmoid"),
        "unsupported Inkling gate_activation; native conversion requires sigmoid"
    );
    ensure!(
        nested_bool(config, &["mtp_config", "chain_hidden_post_norm"]) != Some(true),
        "Inkling chain_hidden_post_norm=true is not supported"
    );
    Ok(())
}

fn local_layer_flags(
    text: &Value,
    config: &Value,
    trunk_layers: u32,
    mtp_layers: u32,
) -> Result<Vec<bool>> {
    let mut result = flags_from_ids(text, "local_layer_ids", trunk_layers)?;
    if mtp_layers > 0 {
        let mtp = config
            .get("mtp_config")
            .context("Inkling config missing mtp_config")?;
        result.extend(flags_from_ids(mtp, "local_layer_ids", mtp_layers)?);
    }
    Ok(result)
}

fn flags_from_ids(config: &Value, key: &str, count: u32) -> Result<Vec<bool>> {
    let values = config
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("Inkling config missing array {key}"))?;
    let mut result = vec![false; count as usize];
    for value in values {
        let id = value
            .as_u64()
            .and_then(|id| usize::try_from(id).ok())
            .with_context(|| format!("invalid Inkling {key} entry"))?;
        ensure!(
            id < result.len(),
            "Inkling {key} entry {id} is out of range"
        );
        result[id] = true;
    }
    Ok(result)
}

fn required_nested_u32(config: &Value, path: &[&str]) -> Result<u32> {
    nested_u32(config, path).with_context(|| format!("config missing {}", path.join(".")))
}

fn nested_u32(config: &Value, path: &[&str]) -> Option<u32> {
    path.iter()
        .try_fold(config, |value, key| value.get(key))?
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
}

fn nested_bool(config: &Value, path: &[&str]) -> Option<bool> {
    path.iter()
        .try_fold(config, |value, key| value.get(key))?
        .as_bool()
}

fn required_u32(config: &Value, key: &str) -> Result<u32> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .with_context(|| format!("config missing positive u32 {key}"))
}

fn required_f32(config: &Value, key: &str) -> Result<f32> {
    config
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .with_context(|| format!("config missing f32 {key}"))
}

fn model_name(source: &Path) -> &str {
    source
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("Inkling")
}
