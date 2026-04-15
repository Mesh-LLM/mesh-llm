use super::super::layer::{Layer, PerLayerInputBlock};
use super::super::mlp::{Activation, MlpKind, MLP};
use super::super::{
    layer_norm_kind, rms_norm_kind, Attention, AttentionKind, ModelArchitecture, ModelConfig,
    NormKind, QuantizedLinear, RMSNorm, TensorPrefixes,
};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::collections::HashMap;

pub(crate) fn transform_llama_like_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    tensors.retain(|key, _| !key.contains("self_attn.rotary_emb.inv_freq"));

    if config.tie_word_embeddings {
        if let Some(prefix) = prefixes.lm_head.as_deref() {
            tensors.remove(&format!("{prefix}.weight"));
            tensors.remove(&format!("{prefix}.scales"));
            tensors.remove(&format!("{prefix}.biases"));
            tensors.remove(&format!("{prefix}.bias"));
        }
    }

    Ok(())
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
    let is_full_attention = arch.is_gemma4() && matches!(layer_type, Some("full_attention"));
    let layer_head_dim = if is_full_attention {
        config.global_head_dim.unwrap_or(head_dim)
    } else {
        head_dim
    };
    let rope_parameters = layer_type.and_then(|name| {
        config
            .rope_parameters
            .as_ref()
            .and_then(|map| map.get(name))
    });
    let rope_dim = if is_full_attention {
        ((layer_head_dim as f32)
            * rope_parameters
                .and_then(|params| params.partial_rotary_factor)
                .unwrap_or(1.0))
        .round() as i32
    } else if arch.is_glm4() {
        ((layer_head_dim as f32) * config.partial_rotary_factor.unwrap_or(1.0)).round() as i32
    } else {
        layer_head_dim
    };
    let rope_theta = rope_parameters
        .and_then(|params| params.rope_theta)
        .unwrap_or(config.rope_theta);
    let window_size = attention_window_size_for_layer(arch, config, layer_index, layer_type)?;
    let kv_shared_source = kv_shared_source_for_layer(
        arch,
        config,
        layer_index,
        layer_type,
        non_shared_layer_types,
    );
    let scale = if arch.is_gemma4() {
        1.0
    } else if let Some(query_pre_attn_scalar) = config.query_pre_attn_scalar {
        1.0 / query_pre_attn_scalar.sqrt()
    } else {
        1.0 / (layer_head_dim as f32).sqrt()
    };
    let mlp_in_norm_key = if arch.is_glm4() {
        format!("{p}.post_attention_layernorm.weight")
    } else if arch.is_gemma2() || arch.is_gemma3() || arch.is_gemma4() {
        format!("{p}.pre_feedforward_layernorm.weight")
    } else {
        format!("{p}.post_attention_layernorm.weight")
    };

    Ok(Layer {
        attn: AttentionKind::Standard(Attention {
            q_proj: load_qlinear(&format!("{p}.self_attn.q_proj"))?,
            k_proj: load_qlinear(&format!("{p}.self_attn.k_proj"))?,
            v_proj: load_qlinear(&format!("{p}.self_attn.v_proj"))?,
            o_proj: load_qlinear(&format!("{p}.self_attn.o_proj"))?,
            q_norm: tensors
                .get(&format!("{p}.self_attn.q_norm.weight"))
                .cloned()
                .map(|weight| RMSNorm {
                    weight,
                    eps: config.rms_norm_eps,
                    add_unit_offset: arch.uses_gemma_norm_offset(),
                }),
            k_norm: tensors
                .get(&format!("{p}.self_attn.k_norm.weight"))
                .cloned()
                .map(|weight| RMSNorm {
                    weight,
                    eps: config.rms_norm_eps,
                    add_unit_offset: arch.uses_gemma_norm_offset(),
                }),
            v_norm: arch.is_gemma4().then(|| RMSNorm {
                weight: mlx_rs::ops::ones::<f32>(&[layer_head_dim])
                    .expect("allocating v_norm scale"),
                eps: config.rms_norm_eps,
                add_unit_offset: false,
            }),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: layer_head_dim,
            scale,
            attn_logit_softcapping: arch
                .is_gemma2()
                .then_some(config.attn_logit_softcapping.unwrap_or(50.0)),
            rope_dim,
            rope_theta,
            rope_traditional,
            window_size,
            kv_shared_source,
        }),
        mlp: MlpKind::Dense(MLP {
            gate_up_proj: tensors
                .contains_key(&format!("{p}.mlp.gate_up_proj.weight"))
                .then(|| load_qlinear(&format!("{p}.mlp.gate_up_proj")))
                .transpose()?,
            gate_proj: tensors
                .contains_key(&format!("{p}.mlp.gate_proj.weight"))
                .then(|| load_qlinear(&format!("{p}.mlp.gate_proj")))
                .transpose()?,
            up_proj: tensors
                .contains_key(&format!("{p}.mlp.up_proj.weight"))
                .then(|| load_qlinear(&format!("{p}.mlp.up_proj")))
                .transpose()?,
            down_proj: load_qlinear(&format!("{p}.mlp.down_proj"))?,
            activation: match config.hidden_activation.as_deref() {
                Some("gelu_pytorch_tanh") | Some("gelu") => Activation::GeluApproximate,
                _ => Activation::Silu,
            },
        }),
        attn_in_norm: (!arch.is_olmo2())
            .then(|| -> Result<NormKind> {
                if arch.is_olmo() {
                    Ok(layer_norm_kind(1e-5))
                } else {
                    Ok(rms_norm_kind(
                        tensors
                            .get(&format!("{p}.input_layernorm.weight"))
                            .cloned()
                            .with_context(|| format!("missing {p}.input_layernorm.weight"))?,
                        config.rms_norm_eps,
                        arch.uses_gemma_norm_offset(),
                    ))
                }
            })
            .transpose()?,
        attn_out_norm: (arch.is_glm4()
            || arch.is_olmo2()
            || arch.is_gemma2()
            || arch.is_gemma3()
            || arch.is_gemma4())
        .then(|| -> Result<NormKind> {
            let key = if arch.is_glm4() {
                format!("{p}.post_self_attn_layernorm.weight")
            } else {
                format!("{p}.post_attention_layernorm.weight")
            };
            Ok(rms_norm_kind(
                tensors
                    .get(&key)
                    .cloned()
                    .with_context(|| format!("missing {key}"))?,
                config.rms_norm_eps,
                arch.uses_gemma_norm_offset(),
            ))
        })
        .transpose()?,
        mlp_in_norm: (!arch.is_olmo2())
            .then(|| -> Result<NormKind> {
                if arch.is_olmo() {
                    Ok(layer_norm_kind(1e-5))
                } else {
                    Ok(rms_norm_kind(
                        tensors.get(&mlp_in_norm_key).cloned().with_context(|| {
                            if arch.is_gemma2() || arch.is_gemma3() {
                                format!("missing {p}.pre_feedforward_layernorm.weight")
                            } else if arch.is_glm4() {
                                format!("missing {p}.post_attention_layernorm.weight")
                            } else {
                                format!("missing {p}.post_attention_layernorm.weight")
                            }
                        })?,
                        config.rms_norm_eps,
                        arch.uses_gemma_norm_offset(),
                    ))
                }
            })
            .transpose()?,
        mlp_out_norm: (arch.is_glm4()
            || arch.is_olmo2()
            || arch.is_gemma2()
            || arch.is_gemma3()
            || arch.is_gemma4())
        .then(|| -> Result<NormKind> {
            let key = if arch.is_glm4() {
                format!("{p}.post_mlp_layernorm.weight")
            } else {
                format!("{p}.post_feedforward_layernorm.weight")
            };
            Ok(rms_norm_kind(
                tensors
                    .get(&key)
                    .cloned()
                    .with_context(|| format!("missing {key}"))?,
                config.rms_norm_eps,
                arch.uses_gemma_norm_offset(),
            ))
        })
        .transpose()?,
        per_layer_input: arch
            .is_gemma4()
            .then(|| -> Result<PerLayerInputBlock> {
                Ok(PerLayerInputBlock {
                    input_gate: load_qlinear(&format!("{p}.per_layer_input_gate"))?,
                    projection: load_qlinear(&format!("{p}.per_layer_projection"))?,
                    post_norm: rms_norm_kind(
                        tensors
                            .get(&format!("{p}.post_per_layer_input_norm.weight"))
                            .cloned()
                            .with_context(|| {
                                format!("missing {p}.post_per_layer_input_norm.weight")
                            })?,
                        config.rms_norm_eps,
                        false,
                    ),
                    activation: match config.hidden_activation.as_deref() {
                        Some("gelu_pytorch_tanh") | Some("gelu") => Activation::GeluApproximate,
                        _ => Activation::Silu,
                    },
                })
            })
            .transpose()?,
        layer_scalar: arch
            .is_gemma4()
            .then(|| tensors.get(&format!("{p}.layer_scalar")).cloned())
            .flatten(),
    })
}
