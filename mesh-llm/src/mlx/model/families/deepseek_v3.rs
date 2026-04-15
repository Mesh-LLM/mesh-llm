use super::super::layer::Layer;
use super::super::{
    quant_params_for, quantize_stacked_weights, rms_norm_kind, DeepseekV3Attention, DeepseekV3MoE,
    MlpKind, ModelConfig, QuantizedLinear, QuantizedMultiLinear, QuantizedSwitchLinear, RMSNorm,
    TensorPrefixes, MLP,
};
use anyhow::{Context, Result};
use mlx_rs::ops::dequantize;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use serde_json::Value;
use std::collections::HashMap;

pub(crate) fn transform_deepseek_v3_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
    config_json: &Value,
    default_group_size: i32,
    default_bits: i32,
) -> Result<()> {
    let num_heads = config.num_attention_heads;
    let qk_nope_head_dim = config
        .qk_nope_head_dim
        .context("missing qk_nope_head_dim for DeepSeekV3")?;
    let v_head_dim = config
        .v_head_dim
        .context("missing v_head_dim for DeepSeekV3")?;
    let kv_lora_rank = config
        .kv_lora_rank
        .context("missing kv_lora_rank for DeepSeekV3")?;

    for i in 0..config.num_hidden_layers {
        let prefix = format!("{}.layers.{i}.self_attn", prefixes.model);
        if !tensors.contains_key(&format!("{prefix}.kv_b_proj.weight"))
            || tensors.contains_key(&format!("{prefix}.embed_q.weight"))
        {
            continue;
        }

        let (group_size, bits) = quant_params_for(
            config_json,
            &format!("{prefix}.kv_b_proj"),
            default_group_size,
            default_bits,
        );
        let weight = tensors
            .get(&format!("{prefix}.kv_b_proj.weight"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.weight"))?;
        let scales = tensors
            .get(&format!("{prefix}.kv_b_proj.scales"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.scales"))?;
        let biases = tensors
            .get(&format!("{prefix}.kv_b_proj.biases"))
            .cloned()
            .with_context(|| format!("missing {prefix}.kv_b_proj.biases"))?;
        let dense = dequantize(&weight, &scales, &biases, group_size, bits)?;
        let dense = dense.reshape(&[num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank])?;
        let wk = dense
            .index((std::ops::RangeFull, ..qk_nope_head_dim, std::ops::RangeFull))
            .transpose_axes(&[0, 2, 1])?;
        let wv = dense.index((std::ops::RangeFull, qk_nope_head_dim.., std::ops::RangeFull));
        let (wk_q, wk_s, wk_b) = quantize_stacked_weights(&wk, group_size, bits)?;
        let (wv_q, wv_s, wv_b) = quantize_stacked_weights(&wv, group_size, bits)?;
        tensors.insert(format!("{prefix}.embed_q.weight"), wk_q);
        tensors.insert(format!("{prefix}.embed_q.scales"), wk_s);
        tensors.insert(format!("{prefix}.embed_q.biases"), wk_b);
        tensors.insert(format!("{prefix}.unembed_out.weight"), wv_q);
        tensors.insert(format!("{prefix}.unembed_out.scales"), wv_s);
        tensors.insert(format!("{prefix}.unembed_out.biases"), wv_b);
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
    let qk_nope_head_dim = config
        .qk_nope_head_dim
        .context("missing qk_nope_head_dim for DeepSeekV3")?;
    let qk_rope_head_dim = config
        .qk_rope_head_dim
        .context("missing qk_rope_head_dim for DeepSeekV3")?;
    let kv_lora_rank = config
        .kv_lora_rank
        .context("missing kv_lora_rank for DeepSeekV3")?;
    let v_head_dim = config
        .v_head_dim
        .context("missing v_head_dim for DeepSeekV3")?;
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let is_moe_layer = config.n_routed_experts.is_some()
        && (layer_index >= config.first_k_dense_replace.unwrap_or(0))
        && (layer_index % config.moe_layer_freq.unwrap_or(1) == 0);
    let shared_intermediate = config
        .n_shared_experts
        .zip(config.moe_intermediate_size)
        .map(|(n_shared, hidden)| n_shared * hidden);
    let mlp_kind = if is_moe_layer {
        MlpKind::DeepseekV3MoE(DeepseekV3MoE {
            switch_gate_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.gate_proj"))?,
            switch_up_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.up_proj"))?,
            switch_down_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.down_proj"))?,
            gate_weight: tensors
                .get(&format!("{p}.mlp.gate.weight"))
                .cloned()
                .with_context(|| format!("missing {p}.mlp.gate.weight"))?,
            gate_bias: tensors
                .get(&format!("{p}.mlp.gate.e_score_correction_bias"))
                .cloned()
                .with_context(|| format!("missing {p}.mlp.gate.e_score_correction_bias"))?,
            top_k: config.num_experts_per_tok.unwrap_or(1),
            n_group: config.n_group.unwrap_or(1),
            topk_group: config.topk_group.unwrap_or(1),
            routed_scaling_factor: config.routed_scaling_factor.unwrap_or(1.0),
            norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
            shared_experts: shared_intermediate
                .map(|_intermediate_size| -> Result<MLP> {
                    Ok(MLP {
                        gate_up_proj: None,
                        gate_proj: Some(load_qlinear(&format!(
                            "{p}.mlp.shared_experts.gate_proj"
                        ))?),
                        up_proj: Some(load_qlinear(&format!("{p}.mlp.shared_experts.up_proj"))?),
                        down_proj: load_qlinear(&format!("{p}.mlp.shared_experts.down_proj"))?,
                        activation: super::super::mlp::Activation::Silu,
                    })
                })
                .transpose()?,
        })
    } else {
        MlpKind::Dense(MLP {
            gate_up_proj: None,
            gate_proj: Some(load_qlinear(&format!("{p}.mlp.gate_proj"))?),
            up_proj: Some(load_qlinear(&format!("{p}.mlp.up_proj"))?),
            down_proj: load_qlinear(&format!("{p}.mlp.down_proj"))?,
            activation: super::super::mlp::Activation::Silu,
        })
    };

    Ok(Layer {
        attn: super::super::AttentionKind::DeepseekV3(DeepseekV3Attention {
            q_proj: if config.q_lora_rank.is_some() {
                None
            } else {
                Some(load_qlinear(&format!("{p}.self_attn.q_proj"))?)
            },
            q_a_proj: config
                .q_lora_rank
                .is_some()
                .then(|| load_qlinear(&format!("{p}.self_attn.q_a_proj")))
                .transpose()?,
            q_a_layernorm: tensors
                .get(&format!("{p}.self_attn.q_a_layernorm.weight"))
                .cloned()
                .map(|weight| RMSNorm {
                    weight,
                    eps: 1e-6,
                    add_unit_offset: false,
                }),
            q_b_proj: config
                .q_lora_rank
                .is_some()
                .then(|| load_qlinear(&format!("{p}.self_attn.q_b_proj")))
                .transpose()?,
            kv_a_proj_with_mqa: load_qlinear(&format!("{p}.self_attn.kv_a_proj_with_mqa"))?,
            kv_a_layernorm: RMSNorm {
                weight: tensors
                    .get(&format!("{p}.self_attn.kv_a_layernorm.weight"))
                    .cloned()
                    .with_context(|| format!("missing {p}.self_attn.kv_a_layernorm.weight"))?,
                eps: 1e-6,
                add_unit_offset: false,
            },
            embed_q: load_multi_linear(&format!("{p}.self_attn.embed_q"))?,
            unembed_out: load_multi_linear(&format!("{p}.self_attn.unembed_out"))?,
            o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
            num_heads: config.num_attention_heads,
            q_head_dim,
            qk_rope_head_dim,
            qk_nope_head_dim,
            kv_lora_rank,
            v_head_dim,
            scale: 1.0 / (q_head_dim as f32).sqrt(),
            rope_theta: config.rope_theta,
        }),
        mlp: mlp_kind,
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
