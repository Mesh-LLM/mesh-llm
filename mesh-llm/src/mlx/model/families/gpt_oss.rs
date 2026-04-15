use super::super::layer::Layer;
use super::super::{
    rms_norm_kind, Attention, AttentionKind, GptOssMoE, MlpKind, ModelConfig, QuantizedLinear,
    QuantizedSwitchLinear, TensorPrefixes,
};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::collections::HashMap;

fn split_even_odd_axis(tensor: &Array, axis: i32) -> Result<(Array, Array)> {
    let shape = tensor.shape();
    let ndim = shape.len() as i32;
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        anyhow::bail!("axis {axis} out of bounds for GPT-OSS tensor shape {shape:?}");
    }
    let axis_len = shape[axis as usize];
    let even_idx: Vec<u32> = (0..axis_len).step_by(2).map(|idx| idx as u32).collect();
    let odd_idx: Vec<u32> = (1..axis_len).step_by(2).map(|idx| idx as u32).collect();
    Ok((
        tensor.take_axis(
            &Array::from_slice(&even_idx, &[even_idx.len() as i32]),
            axis,
        )?,
        tensor.take_axis(&Array::from_slice(&odd_idx, &[odd_idx.len() as i32]), axis)?,
    ))
}

fn split_gate_up_proj(prefix: &str, tensors: &mut HashMap<String, Array>) -> Result<()> {
    if tensors.contains_key(&format!("{prefix}.gate_proj.weight")) {
        return Ok(());
    }

    for suffix in ["weight", "scales", "biases"] {
        let key = format!("{prefix}.gate_up_proj.{suffix}");
        if let Some(fused) = tensors.get(&key).cloned() {
            let (gate, up) = split_even_odd_axis(&fused, -2)?;
            tensors.insert(format!("{prefix}.gate_proj.{suffix}"), gate);
            tensors.insert(format!("{prefix}.up_proj.{suffix}"), up);
        }
    }

    let bias_key = format!("{prefix}.gate_up_proj_bias");
    if let Some(fused_bias) = tensors.get(&bias_key).cloned() {
        let (gate_bias, up_bias) = split_even_odd_axis(&fused_bias, -1)?;
        tensors.insert(format!("{prefix}.gate_proj.bias"), gate_bias);
        tensors.insert(format!("{prefix}.up_proj.bias"), up_bias);
    }

    Ok(())
}

fn normalize_down_proj_bias(prefix: &str, tensors: &mut HashMap<String, Array>) {
    let legacy_key = format!("{prefix}.down_proj_bias");
    let normalized_key = format!("{prefix}.down_proj.bias");
    if let Some(bias) = tensors.get(&legacy_key).cloned() {
        tensors.entry(normalized_key).or_insert(bias);
    }
}

fn normalize_expert_quant_bias(prefix: &str, tensors: &mut HashMap<String, Array>) {
    let legacy_key = format!("{prefix}.bias");
    let normalized_key = format!("{prefix}.biases");
    if tensors.contains_key(&normalized_key) {
        return;
    }
    if let Some(biases) = tensors.remove(&legacy_key) {
        tensors.insert(normalized_key, biases);
    }
}

pub(crate) fn transform_gpt_oss_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    for i in 0..config.num_hidden_layers {
        let mlp_prefix = format!("{}.layers.{i}.mlp.experts", prefixes.model);
        if tensors
            .keys()
            .any(|key| key.starts_with(&format!("{mlp_prefix}.gate_up_proj")))
        {
            split_gate_up_proj(&mlp_prefix, tensors)
                .with_context(|| format!("failed to sanitize GPT-OSS tensors for {mlp_prefix}"))?;
        }
        normalize_down_proj_bias(&mlp_prefix, tensors);
        normalize_expert_quant_bias(&format!("{mlp_prefix}.gate_proj"), tensors);
        normalize_expert_quant_bias(&format!("{mlp_prefix}.up_proj"), tensors);
        normalize_expert_quant_bias(&format!("{mlp_prefix}.down_proj"), tensors);
    }

    Ok(())
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
    Ok(Layer {
        attn: AttentionKind::Standard(Attention {
            q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
            k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
            v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
            o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
            q_norm: None,
            k_norm: None,
            v_norm: None,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            attn_logit_softcapping: None,
            rope_dim: head_dim,
            rope_theta: config.rope_theta,
            rope_traditional: false,
            window_size,
            kv_shared_source: None,
        }),
        mlp: MlpKind::GptOssMoE(GptOssMoE {
            switch_gate_proj: load_switch_linear(&format!("{p}.mlp.experts.gate_proj"))?,
            switch_up_proj: load_switch_linear(&format!("{p}.mlp.experts.up_proj"))?,
            switch_down_proj: load_switch_linear(&format!("{p}.mlp.experts.down_proj"))?,
            router: load_qlinear(&format!("{p}.mlp.router"))?,
            top_k: config.num_experts_per_tok.unwrap_or(1),
        }),
        attn_in_norm: Some(rms_norm_kind(
            tensors
                .get(&format!("{p}.input_layernorm.weight"))
                .cloned()
                .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
            config.rms_norm_eps,
            false,
        )),
        attn_out_norm: None,
        mlp_in_norm: Some(rms_norm_kind(
            tensors
                .get(&format!("{p}.post_attention_layernorm.weight"))
                .cloned()
                .with_context(|| format!("missing {p}.post_attention_layernorm.weight"))?,
            config.rms_norm_eps,
            false,
        )),
        mlp_out_norm: None,
        per_layer_input: None,
        layer_scalar: None,
    })
}
