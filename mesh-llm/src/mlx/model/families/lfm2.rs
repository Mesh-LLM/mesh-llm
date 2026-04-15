use super::super::layer::Layer;
use super::super::{
    rms_norm_kind, Activation, Attention, AttentionKind, Lfm2ShortConv, MlpKind, ModelConfig,
    QuantizedLinear, RMSNorm, MLP,
};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::collections::HashMap;

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
    let full_attn_idxs = config
        .full_attn_idxs
        .as_ref()
        .with_context(|| format!("missing full_attn_idxs for LFM2 layer {}", layer_index))?;
    let is_attention_layer = full_attn_idxs.contains(&layer_index);
    let operator = if is_attention_layer {
        AttentionKind::Standard(Attention {
            q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
            k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
            v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
            o_proj: load_qlinear(&format!("{p}.self_attn.out_proj"))?,
            q_norm: tensors
                .get(&format!("{p}.self_attn.q_layernorm.weight"))
                .cloned()
                .map(|weight| RMSNorm {
                    weight,
                    eps: config.block_norm_eps.unwrap_or(config.rms_norm_eps),
                    add_unit_offset: false,
                }),
            k_norm: tensors
                .get(&format!("{p}.self_attn.k_layernorm.weight"))
                .cloned()
                .map(|weight| RMSNorm {
                    weight,
                    eps: config.block_norm_eps.unwrap_or(config.rms_norm_eps),
                    add_unit_offset: false,
                }),
            v_norm: None,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            attn_logit_softcapping: None,
            rope_dim: head_dim,
            rope_theta: config.rope_theta,
            rope_traditional: false,
            window_size: None,
            kv_shared_source: None,
        })
    } else {
        AttentionKind::Lfm2ShortConv(Lfm2ShortConv {
            conv_weight: load_conv_weight(&format!("{p}.conv.conv"))?,
            in_proj: load_qlinear(&format!("{p}.conv.in_proj"))?,
            out_proj: load_qlinear(&format!("{p}.conv.out_proj"))?,
            hidden_size: config.hidden_size,
            conv_l_cache: config.conv_l_cache.unwrap_or(3),
        })
    };

    Ok(Layer {
        attn: operator,
        mlp: MlpKind::Dense(MLP {
            gate_up_proj: None,
            gate_proj: Some(load_qlinear(&format!("{p}.feed_forward.w1"))?),
            up_proj: Some(load_qlinear(&format!("{p}.feed_forward.w3"))?),
            down_proj: load_qlinear(&format!("{p}.feed_forward.w2"))?,
            activation: Activation::Silu,
        }),
        attn_in_norm: Some(rms_norm_kind(
            tensors
                .get(&format!("{p}.operator_norm.weight"))
                .cloned()
                .with_context(|| format!("missing {p}.operator_norm.weight"))?,
            config.block_norm_eps.unwrap_or(config.rms_norm_eps),
            false,
        )),
        attn_out_norm: None,
        mlp_in_norm: Some(rms_norm_kind(
            tensors
                .get(&format!("{p}.ffn_norm.weight"))
                .cloned()
                .with_context(|| format!("missing {p}.ffn_norm.weight"))?,
            config.block_norm_eps.unwrap_or(config.rms_norm_eps),
            false,
        )),
        mlp_out_norm: None,
        per_layer_input: None,
        layer_scalar: None,
    })
}
