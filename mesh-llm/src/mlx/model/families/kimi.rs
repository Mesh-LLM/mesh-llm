use super::super::layer::Layer;
use super::super::{
    rms_norm_kind, AttentionKind, DeepseekV3MoE, KimiDeltaAttention, KimiMlaAttention,
    KimiShortConv, MlpKind, ModelConfig, QuantizedLinear, QuantizedMultiLinear,
    QuantizedSwitchLinear, RMSNorm, MLP,
};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::collections::HashMap;

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
    let linear_cfg = config
        .linear_attn_config
        .as_ref()
        .context("missing linear_attn_config for Kimi Linear")?;
    let is_linear_layer = linear_cfg.kda_layers.contains(&(layer_index + 1));
    let projection_dim = linear_cfg.num_heads * linear_cfg.head_dim;
    let is_moe_layer = config.n_routed_experts.unwrap_or(0) > 0
        && (layer_index >= config.first_k_dense_replace.unwrap_or(0))
        && (layer_index % config.moe_layer_freq.unwrap_or(1) == 0);
    let mlp = if is_moe_layer {
        MlpKind::DeepseekV3MoE(DeepseekV3MoE {
            switch_gate_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.gate_proj"))?,
            switch_up_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.up_proj"))?,
            switch_down_proj: load_switch_linear(&format!("{p}.mlp.switch_mlp.down_proj"))?,
            gate_weight: tensors
                .get(&format!("{p}.mlp.gate.weight"))
                .cloned()
                .with_context(|| format!("missing {p}.mlp.gate.weight"))?,
            gate_bias: tensors
                .get(&format!("{p}.mlp.e_score_correction_bias"))
                .cloned()
                .with_context(|| format!("missing {p}.mlp.e_score_correction_bias"))?,
            top_k: config.num_experts_per_tok.unwrap_or(1),
            n_group: config.n_group.unwrap_or(1),
            topk_group: config.topk_group.unwrap_or(1),
            routed_scaling_factor: config.routed_scaling_factor.unwrap_or(1.0),
            norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
            shared_experts: config
                .n_shared_experts
                .filter(|n| *n > 0)
                .map(|_| -> Result<MLP> {
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

    let attn = if is_linear_layer {
        AttentionKind::KimiDelta(KimiDeltaAttention {
            q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
            k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
            v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
            q_conv: KimiShortConv {
                conv_weight: load_conv_weight(&format!("{p}.self_attn.q_conv.conv"))?,
                kernel_size: linear_cfg.short_conv_kernel_size.unwrap_or(4),
                channels: projection_dim,
            },
            k_conv: KimiShortConv {
                conv_weight: load_conv_weight(&format!("{p}.self_attn.k_conv.conv"))?,
                kernel_size: linear_cfg.short_conv_kernel_size.unwrap_or(4),
                channels: projection_dim,
            },
            v_conv: KimiShortConv {
                conv_weight: load_conv_weight(&format!("{p}.self_attn.v_conv.conv"))?,
                kernel_size: linear_cfg.short_conv_kernel_size.unwrap_or(4),
                channels: projection_dim,
            },
            f_a_proj: load_qlinear(&format!("{p}.self_attn.f_a_proj"))?,
            f_b_proj: load_qlinear(&format!("{p}.self_attn.f_b_proj"))?,
            b_proj: load_qlinear(&format!("{p}.self_attn.b_proj"))?,
            g_a_proj: load_qlinear(&format!("{p}.self_attn.g_a_proj"))?,
            g_b_proj: load_qlinear(&format!("{p}.self_attn.g_b_proj"))?,
            a_log: tensors
                .get(&format!("{p}.self_attn.A_log"))
                .cloned()
                .with_context(|| format!("missing {p}.self_attn.A_log"))?,
            dt_bias: tensors
                .get(&format!("{p}.self_attn.dt_bias"))
                .cloned()
                .with_context(|| format!("missing {p}.self_attn.dt_bias"))?,
            o_norm: RMSNorm {
                weight: tensors
                    .get(&format!("{p}.self_attn.o_norm.weight"))
                    .cloned()
                    .with_context(|| format!("missing {p}.self_attn.o_norm.weight"))?,
                eps: config.rms_norm_eps,
                add_unit_offset: false,
            },
            o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
            num_heads: linear_cfg.num_heads,
            head_dim: linear_cfg.head_dim,
            scale: (linear_cfg.head_dim as f32).powf(-0.5),
        })
    } else {
        let qk_nope_head_dim = config
            .qk_nope_head_dim
            .context("missing qk_nope_head_dim for Kimi Linear MLA")?;
        let qk_rope_head_dim = config
            .qk_rope_head_dim
            .context("missing qk_rope_head_dim for Kimi Linear MLA")?;
        let kv_lora_rank = config
            .kv_lora_rank
            .context("missing kv_lora_rank for Kimi Linear MLA")?;
        let v_head_dim = config
            .v_head_dim
            .context("missing v_head_dim for Kimi Linear MLA")?;
        AttentionKind::KimiMla(KimiMlaAttention {
            q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
            kv_a_proj_with_mqa: load_qlinear(&format!("{p}.self_attn.kv_a_proj_with_mqa"))?,
            kv_a_layernorm: RMSNorm {
                weight: tensors
                    .get(&format!("{p}.self_attn.kv_a_layernorm.weight"))
                    .cloned()
                    .with_context(|| format!("missing {p}.self_attn.kv_a_layernorm.weight"))?,
                eps: config.rms_norm_eps,
                add_unit_offset: false,
            },
            embed_q: load_multi_linear(&format!("{p}.self_attn.embed_q"))?,
            unembed_out: load_multi_linear(&format!("{p}.self_attn.unembed_out"))?,
            o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
            num_heads: config.num_attention_heads,
            q_head_dim: qk_nope_head_dim + qk_rope_head_dim,
            qk_rope_head_dim,
            qk_nope_head_dim,
            kv_lora_rank,
            v_head_dim,
            scale: 1.0 / ((qk_nope_head_dim + qk_rope_head_dim) as f32).sqrt(),
        })
    };

    Ok(Layer {
        attn,
        mlp,
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
