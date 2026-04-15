use super::family::ModelArchitecture;
use super::layer::Layer;
use super::{
    ModelConfig, QuantizedLinear, QuantizedMultiLinear, QuantizedSwitchLinear, TensorPrefixes,
};
use anyhow::Result;
use mlx_rs::Array;
use serde_json::Value;
use std::collections::HashMap;

mod deepseek_v3;
mod gemma3;
mod gemma4;
mod gpt_oss;
mod kimi;
mod lfm2;
mod llama_like;
mod olmo2;
mod phi3;

pub(crate) fn apply_family_tensor_transforms(
    arch: ModelArchitecture,
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
    config_json: &Value,
    default_group_size: i32,
    default_bits: i32,
) -> Result<()> {
    if matches!(arch, ModelArchitecture::LlamaLike) {
        llama_like::transform_llama_like_tensors(tensors, prefixes, config)?;
    }

    if arch.is_deepseek_v3() || arch.is_kimi_linear() {
        deepseek_v3::transform_deepseek_v3_tensors(
            tensors,
            prefixes,
            config,
            config_json,
            default_group_size,
            default_bits,
        )?;
    }

    if config_json
        .get("model_type")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("phi3"))
    {
        phi3::transform_phi3_tensors(tensors, prefixes, config)?;
    }

    if arch.is_gpt_oss() {
        gpt_oss::transform_gpt_oss_tensors(tensors, prefixes, config)?;
    }

    if arch.is_gemma3() {
        gemma3::transform_gemma3_tensors(tensors, prefixes, config)?;
    }

    if arch.is_gemma4() {
        gemma4::transform_gemma4_tensors(tensors, prefixes, config)?;
    }

    if arch.is_olmo2() {
        olmo2::transform_olmo2_tensors(tensors, prefixes, config)?;
    }

    Ok(())
}

pub(crate) fn build_deepseek_v3_layer<FQ, FM, FS>(
    tensors: &HashMap<String, Array>,
    p: &str,
    layer_index: i32,
    config: &ModelConfig,
    load_qlinear: &FQ,
    load_multi_linear: &FM,
    load_switch_linear: &FS,
) -> Result<Layer>
where
    FQ: Fn(&str) -> Result<QuantizedLinear>,
    FM: Fn(&str) -> Result<QuantizedMultiLinear>,
    FS: Fn(&str) -> Result<QuantizedSwitchLinear>,
{
    deepseek_v3::build_deepseek_v3_layer(
        tensors,
        p,
        layer_index,
        config,
        load_qlinear,
        load_multi_linear,
        load_switch_linear,
    )
}

pub(crate) fn build_lfm2_layer<FQ>(
    tensors: &HashMap<String, Array>,
    p: &str,
    layer_index: i32,
    config: &ModelConfig,
    head_dim: i32,
    load_qlinear: &FQ,
    load_conv_weight: &impl Fn(&str) -> Result<Array>,
) -> Result<Layer>
where
    FQ: Fn(&str) -> Result<QuantizedLinear>,
{
    lfm2::build_lfm2_layer(
        tensors,
        p,
        layer_index,
        config,
        head_dim,
        load_qlinear,
        load_conv_weight,
    )
}

pub(crate) fn build_kimi_linear_layer<FQ, FM, FS>(
    tensors: &HashMap<String, Array>,
    p: &str,
    layer_index: i32,
    config: &ModelConfig,
    load_qlinear: &FQ,
    load_multi_linear: &FM,
    load_switch_linear: &FS,
    load_conv_weight: &impl Fn(&str) -> Result<Array>,
) -> Result<Layer>
where
    FQ: Fn(&str) -> Result<QuantizedLinear>,
    FM: Fn(&str) -> Result<QuantizedMultiLinear>,
    FS: Fn(&str) -> Result<QuantizedSwitchLinear>,
{
    kimi::build_kimi_linear_layer(
        tensors,
        p,
        layer_index,
        config,
        load_qlinear,
        load_multi_linear,
        load_switch_linear,
        load_conv_weight,
    )
}

pub(crate) fn build_gpt_oss_layer<FQ, FS>(
    tensors: &HashMap<String, Array>,
    p: &str,
    config: &ModelConfig,
    head_dim: i32,
    window_size: Option<i32>,
    load_qlinear: &FQ,
    load_switch_linear: &FS,
) -> Result<Layer>
where
    FQ: Fn(&str) -> Result<QuantizedLinear>,
    FS: Fn(&str) -> Result<QuantizedSwitchLinear>,
{
    gpt_oss::build_gpt_oss_layer(
        tensors,
        p,
        config,
        head_dim,
        window_size,
        load_qlinear,
        load_switch_linear,
    )
}

pub(crate) fn build_standard_layer<FQ>(
    tensors: &HashMap<String, Array>,
    p: &str,
    layer_index: usize,
    arch: ModelArchitecture,
    config: &ModelConfig,
    layer_type: Option<&str>,
    head_dim: i32,
    rope_traditional: bool,
    non_shared_layer_types: Option<&[String]>,
    load_qlinear: &FQ,
    attention_window_size_for_layer: &impl Fn(
        ModelArchitecture,
        &ModelConfig,
        usize,
        Option<&str>,
    ) -> Result<Option<i32>>,
    kv_shared_source_for_layer: &impl Fn(
        ModelArchitecture,
        &ModelConfig,
        usize,
        Option<&str>,
        Option<&[String]>,
    ) -> Option<usize>,
) -> Result<Layer>
where
    FQ: Fn(&str) -> Result<QuantizedLinear>,
{
    llama_like::build_standard_layer(
        tensors,
        p,
        layer_index,
        arch,
        config,
        layer_type,
        head_dim,
        rope_traditional,
        non_shared_layer_types,
        load_qlinear,
        attention_window_size_for_layer,
        kv_shared_source_for_layer,
    )
}
