use anyhow::{Result, anyhow, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorNameMap {
    Raw,
    HfToGguf,
    HfToGgufWithMtp { layer_start: u32 },
}

impl TensorNameMap {
    pub(crate) fn map_tensor_name(self, name: &str) -> Result<String> {
        match self {
            Self::Raw => Ok(name.to_string()),
            Self::HfToGguf => map_hf_to_gguf(name, None),
            Self::HfToGgufWithMtp { layer_start } => map_hf_to_gguf(name, Some(layer_start)),
        }
    }
}

fn map_hf_to_gguf(name: &str, mtp_layer_start: Option<u32>) -> Result<String> {
    if let Some(layer_start) = mtp_layer_start
        && let Some(normalized) = normalize_inkling_mtp_source_name(name, layer_start)?
    {
        return map_hf_to_gguf(&normalized, mtp_layer_start);
    }
    if let Some(layer_start) = mtp_layer_start
        && let Some(normalized) = normalize_qwen_mtp_source_name(name, layer_start)?
    {
        return map_hf_to_gguf(&normalized, mtp_layer_start);
    }
    if matches!(name, "model.embed_tokens.weight" | "model.llm.embed.weight") {
        return Ok("token_embd.weight".to_string());
    }
    if name == "embed_tokens.weight" {
        return Ok("token_embd.weight".to_string());
    }
    if matches!(name, "lm_head.weight" | "model.llm.unembed.weight") {
        return Ok("output.weight".to_string());
    }
    if matches!(name, "model.norm.weight" | "model.llm.norm.weight") {
        return Ok("output_norm.weight".to_string());
    }
    if name == "model.llm.embed_norm.weight" {
        return Ok("token_embd_norm.weight".to_string());
    }
    if name == "norm.weight" {
        return Ok("output_norm.weight".to_string());
    }
    if let Some(name) = map_mtp_source_tensor(name)? {
        return Ok(name);
    }
    let Some(layer) = HfLayerTensor::parse(name)? else {
        bail!("no HF->GGUF tensor mapping for {name}");
    };
    layer.map()
}

pub(crate) fn is_mtp_source_tensor(name: &str) -> bool {
    is_inkling_mtp_source_tensor(name)
        || is_qwen_mtp_source_tensor(name)
        || map_mtp_source_tensor(name).is_ok_and(|mapped| mapped.is_some())
}

pub(crate) fn hf_layer_id(name: &str) -> Result<Option<u32>> {
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return Ok(None);
    };
    let Some((layer, _suffix)) = rest.split_once('.') else {
        bail!("malformed layer tensor name {name}");
    };
    layer
        .parse::<u32>()
        .map(Some)
        .map_err(|err| anyhow!("malformed layer id in {name}: {err}"))
}

pub(crate) fn is_shared_mtp_context_tensor(name: &str) -> bool {
    matches!(
        name,
        "model.embed_tokens.weight"
            | "embed_tokens.weight"
            | "model.norm.weight"
            | "norm.weight"
            | "lm_head.weight"
            | "model.llm.embed.weight"
            | "model.llm.embed_norm.weight"
            | "model.llm.norm.weight"
            | "model.llm.unembed.weight"
    )
}

pub(crate) fn is_inkling_fused_w13(name: &str) -> bool {
    (name.starts_with("model.layers.") || name.starts_with("model.mtp.layers."))
        && name.ends_with(".mlp.w13_dn.weight")
}

pub(crate) fn inkling_mtp_depth(name: &str) -> Result<Option<u32>> {
    let Some(rest) = name.strip_prefix("model.mtp.layers.") else {
        return Ok(None);
    };
    let Some((depth, _)) = rest.split_once('.') else {
        bail!("malformed Inkling MTP tensor name {name}");
    };
    depth
        .parse::<u32>()
        .map(Some)
        .map_err(|err| anyhow!("malformed Inkling MTP depth in {name}: {err}"))
}

fn is_inkling_mtp_source_tensor(name: &str) -> bool {
    name.starts_with("model.mtp.layers.")
}

fn normalize_inkling_mtp_source_name(name: &str, layer_start: u32) -> Result<Option<String>> {
    let Some(rest) = name.strip_prefix("model.mtp.layers.") else {
        return Ok(None);
    };
    let Some((depth, suffix)) = rest.split_once('.') else {
        bail!("malformed Inkling MTP tensor name {name}");
    };
    let depth = depth
        .parse::<u32>()
        .map_err(|err| anyhow!("malformed Inkling MTP depth in {name}: {err}"))?;
    let layer = layer_start
        .checked_add(depth)
        .ok_or_else(|| anyhow!("Inkling MTP layer id overflow for {name}"))?;
    let suffix = match suffix {
        "embed_norm.weight" => "enorm.weight",
        "hidden_norm.weight" => "hnorm.weight",
        "input_proj.weight" => "eh_proj.weight",
        value => value.strip_prefix("transformer_block.").unwrap_or(value),
    };
    Ok(Some(format!("model.layers.{layer}.{suffix}")))
}

fn map_mtp_source_tensor(name: &str) -> Result<Option<String>> {
    let mapped = match name {
        "pre_projection" | "pre_projection.weight" => "nextn.pre_projection.weight".to_string(),
        "post_projection" | "post_projection.weight" => "nextn.post_projection.weight".to_string(),
        "d2t" | "d2t.weight" => "d2t.weight".to_string(),
        _ => {
            let Some(layer) = HfLayerTensor::parse(name)? else {
                return Ok(None);
            };
            let bid = layer.layer;
            match layer.suffix {
                "eh_proj" | "eh_proj.weight" => format!("blk.{bid}.nextn.eh_proj.weight"),
                "embed_tokens" | "embed_tokens.weight" => {
                    format!("blk.{bid}.nextn.embed_tokens.weight")
                }
                "enorm" | "enorm.weight" => format!("blk.{bid}.nextn.enorm.weight"),
                "hnorm" | "hnorm.weight" => format!("blk.{bid}.nextn.hnorm.weight"),
                "shared_head.head" | "shared_head.head.weight" | "shared_head.output.weight" => {
                    format!("blk.{bid}.nextn.shared_head_head.weight")
                }
                "shared_head.norm" | "shared_head.norm.weight" => {
                    format!("blk.{bid}.nextn.shared_head_norm.weight")
                }
                _ => return Ok(None),
            }
        }
    };
    Ok(Some(mapped))
}

fn is_qwen_mtp_source_tensor(name: &str) -> bool {
    let name = name.strip_prefix("model.").unwrap_or(name);
    name.starts_with("mtp.")
}

fn normalize_qwen_mtp_source_name(name: &str, layer_start: u32) -> Result<Option<String>> {
    let name = name.strip_prefix("model.").unwrap_or(name);
    if !name.starts_with("mtp.") {
        return Ok(None);
    }
    let parts = name.splitn(4, '.').collect::<Vec<_>>();
    if parts.len() == 4 && parts[1] == "layers" {
        let mtp_idx = parts[2]
            .parse::<u32>()
            .map_err(|err| anyhow!("malformed MTP layer id in {name}: {err}"))?;
        return Ok(Some(format!(
            "model.layers.{}.{}",
            layer_start + mtp_idx,
            parts[3]
        )));
    }
    if parts.len() == 3 {
        let suffix = match parts[1] {
            "fc" => "eh_proj",
            "pre_fc_norm_embedding" => "enorm",
            "pre_fc_norm_hidden" => "hnorm",
            "norm" => "shared_head.norm",
            _ => return Ok(None),
        };
        return Ok(Some(format!(
            "model.layers.{layer_start}.{suffix}.{}",
            parts[2]
        )));
    }
    Ok(None)
}

struct HfLayerTensor<'a> {
    layer: u32,
    suffix: &'a str,
}

impl<'a> HfLayerTensor<'a> {
    fn parse(name: &'a str) -> Result<Option<Self>> {
        let Some(rest) = name.strip_prefix("model.layers.") else {
            return Ok(None);
        };
        let Some((layer, suffix)) = rest.split_once('.') else {
            bail!("malformed layer tensor name {name}");
        };
        let layer = layer
            .parse::<u32>()
            .map_err(|err| anyhow!("malformed layer id in {name}: {err}"))?;
        Ok(Some(Self { layer, suffix }))
    }

    fn map(&self) -> Result<String> {
        let bid = self.layer;
        match self.suffix {
            "input_layernorm.weight" => Ok(format!("blk.{bid}.attn_norm.weight")),
            "post_attention_layernorm.weight" => Ok(format!("blk.{bid}.ffn_norm.weight")),
            "self_attn.q_proj.weight" => Ok(format!("blk.{bid}.attn_q.weight")),
            "self_attn.k_proj.weight" => Ok(format!("blk.{bid}.attn_k.weight")),
            "self_attn.v_proj.weight" => Ok(format!("blk.{bid}.attn_v.weight")),
            "self_attn.q_proj.bias" => Ok(format!("blk.{bid}.attn_q.bias")),
            "self_attn.k_proj.bias" => Ok(format!("blk.{bid}.attn_k.bias")),
            "self_attn.v_proj.bias" => Ok(format!("blk.{bid}.attn_v.bias")),
            "self_attn.o_proj.weight" => Ok(format!("blk.{bid}.attn_output.weight")),
            "self_attn.q_norm.weight" => Ok(format!("blk.{bid}.attn_q_norm.weight")),
            "self_attn.k_norm.weight" => Ok(format!("blk.{bid}.attn_k_norm.weight")),
            "self_attn.q_a_proj.weight" => Ok(format!("blk.{bid}.attn_q_a.weight")),
            "self_attn.q_b_proj.weight" => Ok(format!("blk.{bid}.attn_q_b.weight")),
            "self_attn.q_a_layernorm.weight" => Ok(format!("blk.{bid}.attn_q_a_norm.weight")),
            "self_attn.kv_a_proj_with_mqa.weight" => Ok(format!("blk.{bid}.attn_kv_a_mqa.weight")),
            "self_attn.kv_b_proj.weight" => Ok(format!("blk.{bid}.attn_kv_b.weight")),
            "self_attn.kv_a_layernorm.weight" => Ok(format!("blk.{bid}.attn_kv_a_norm.weight")),
            "attn_norm.weight" => Ok(format!("blk.{bid}.attn_norm.weight")),
            "attn.wq_du.weight" => Ok(format!("blk.{bid}.attn_q.weight")),
            "attn.wk_dv.weight" => Ok(format!("blk.{bid}.attn_k.weight")),
            "attn.wv_dv.weight" => Ok(format!("blk.{bid}.attn_v.weight")),
            "attn.wr_du.weight" => Ok(format!("blk.{bid}.attn_r.weight")),
            "attn.wo_ud.weight" => Ok(format!("blk.{bid}.attn_output.weight")),
            "attn.q_norm.weight" => Ok(format!("blk.{bid}.attn_q_norm.weight")),
            "attn.k_norm.weight" => Ok(format!("blk.{bid}.attn_k_norm.weight")),
            "attn.rel_logits_proj.proj" | "attn.rel_logits_proj.weight" => {
                Ok(format!("blk.{bid}.attn_rel_proj.weight"))
            }
            "attn.k_sconv.weight" => Ok(format!("blk.{bid}.shortconv_k.weight")),
            "attn.v_sconv.weight" => Ok(format!("blk.{bid}.shortconv_v.weight")),
            "attn_sconv.weight" => Ok(format!("blk.{bid}.shortconv_attn.weight")),
            "mlp_sconv.weight" => Ok(format!("blk.{bid}.shortconv_mlp.weight")),
            "mlp_norm.weight" => Ok(format!("blk.{bid}.ffn_norm.weight")),
            "mlp.w2_md.weight" => Ok(format!("blk.{bid}.ffn_down.weight")),
            "mlp.global_scale" | "mlp.global_scale.weight" => {
                Ok(format!("blk.{bid}.ffn_gscale.weight"))
            }
            "mlp.w13_dn.weight" => {
                bail!("Inkling fused w13 tensor requires streaming deinterleave")
            }
            "self_attn.indexer.k_norm.weight" => Ok(format!("blk.{bid}.indexer.k_norm.weight")),
            "self_attn.indexer.k_norm.bias" => Ok(format!("blk.{bid}.indexer.k_norm.bias")),
            "self_attn.indexer.weights_proj.weight" => Ok(format!("blk.{bid}.indexer.proj.weight")),
            "self_attn.indexer.wk.weight" => Ok(format!("blk.{bid}.indexer.attn_k.weight")),
            "self_attn.indexer.wq_b.weight" => Ok(format!("blk.{bid}.indexer.attn_q_b.weight")),
            "mlp.down_proj.weight" => Ok(format!("blk.{bid}.ffn_down.weight")),
            "mlp.gate_proj.weight" => Ok(format!("blk.{bid}.ffn_gate.weight")),
            "mlp.up_proj.weight" => Ok(format!("blk.{bid}.ffn_up.weight")),
            "mlp.gate.weight" => Ok(format!("blk.{bid}.ffn_gate_inp.weight")),
            "mlp.shared_expert_gate" => Ok(format!("blk.{bid}.ffn_gate_inp_shexp.weight")),
            "mlp.gate.e_score_correction_bias" => Ok(format!("blk.{bid}.exp_probs_b.bias")),
            "mlp.shared_expert.down_proj.weight" => Ok(format!("blk.{bid}.ffn_down_shexp.weight")),
            "mlp.shared_expert.gate_proj.weight" => Ok(format!("blk.{bid}.ffn_gate_shexp.weight")),
            "mlp.shared_expert.up_proj.weight" => Ok(format!("blk.{bid}.ffn_up_shexp.weight")),
            "mlp.shared_experts.down_proj.weight" => Ok(format!("blk.{bid}.ffn_down_shexp.weight")),
            "mlp.shared_experts.gate_proj.weight" => Ok(format!("blk.{bid}.ffn_gate_shexp.weight")),
            "mlp.shared_experts.up_proj.weight" => Ok(format!("blk.{bid}.ffn_up_shexp.weight")),
            suffix if suffix.starts_with("mlp.experts.") => {
                bail!(
                    "expert source tensor {suffix} requires streaming expert merge before GGUF write"
                )
            }
            suffix => bail!("no HF->GGUF mapping for layer tensor suffix {suffix:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_glm_moe_lite_direct_tensor_names() {
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.embed_tokens.weight")
                .unwrap(),
            "token_embd.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.7.self_attn.kv_a_proj_with_mqa.weight")
                .unwrap(),
            "blk.7.attn_kv_a_mqa.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.7.mlp.shared_experts.gate_proj.weight")
                .unwrap(),
            "blk.7.ffn_gate_shexp.weight"
        );
    }

    #[test]
    fn maps_glm_dsa_indexer_tensor_names() {
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.7.self_attn.indexer.k_norm.weight")
                .unwrap(),
            "blk.7.indexer.k_norm.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.7.self_attn.indexer.k_norm.bias")
                .unwrap(),
            "blk.7.indexer.k_norm.bias"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.7.self_attn.indexer.weights_proj.weight")
                .unwrap(),
            "blk.7.indexer.proj.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.7.self_attn.indexer.wk.weight")
                .unwrap(),
            "blk.7.indexer.attn_k.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.7.self_attn.indexer.wq_b.weight")
                .unwrap(),
            "blk.7.indexer.attn_q_b.weight"
        );
    }

    #[test]
    fn rejects_expert_source_tensors_until_merge_exists() {
        let err = TensorNameMap::HfToGguf
            .map_tensor_name("model.layers.1.mlp.experts.0.down_proj.weight")
            .unwrap_err()
            .to_string();

        assert!(err.contains("requires streaming expert merge"));
    }

    #[test]
    fn maps_qwen_dense_tensor_names() {
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.self_attn.q_proj.weight")
                .unwrap(),
            "blk.3.attn_q.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.self_attn.k_norm.weight")
                .unwrap(),
            "blk.3.attn_k_norm.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.self_attn.v_proj.bias")
                .unwrap(),
            "blk.3.attn_v.bias"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.mlp.up_proj.weight")
                .unwrap(),
            "blk.3.ffn_up.weight"
        );
    }

    #[test]
    fn maps_qwen2_moe_shared_expert_tensor_names() {
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.mlp.shared_expert_gate")
                .unwrap(),
            "blk.3.ffn_gate_inp_shexp.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.mlp.shared_expert.gate_proj.weight")
                .unwrap(),
            "blk.3.ffn_gate_shexp.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.mlp.shared_expert.down_proj.weight")
                .unwrap(),
            "blk.3.ffn_down_shexp.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.3.mlp.shared_expert.up_proj.weight")
                .unwrap(),
            "blk.3.ffn_up_shexp.weight"
        );
    }

    #[test]
    fn recognizes_known_nextn_mtp_source_tensors() {
        for name in [
            "pre_projection",
            "pre_projection.weight",
            "post_projection",
            "d2t",
            "model.layers.47.eh_proj.weight",
            "model.layers.47.embed_tokens.weight",
            "model.layers.47.enorm.weight",
            "model.layers.47.hnorm.weight",
            "model.layers.47.shared_head.head.weight",
            "model.layers.47.shared_head.norm.weight",
        ] {
            assert!(is_mtp_source_tensor(name), "{name}");
        }
        assert!(!is_mtp_source_tensor(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_mtp_source_tensor("model.embed_tokens.weight"));
    }

    #[test]
    fn maps_known_nextn_mtp_source_tensors() {
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.47.eh_proj.weight")
                .unwrap(),
            "blk.47.nextn.eh_proj.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("model.layers.47.shared_head.norm.weight")
                .unwrap(),
            "blk.47.nextn.shared_head_norm.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGguf
                .map_tensor_name("embed_tokens.weight")
                .unwrap(),
            "token_embd.weight"
        );
    }

    #[test]
    fn recognizes_and_maps_qwen_style_mtp_source_tensors() {
        for name in [
            "mtp.fc.weight",
            "model.mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
        ] {
            assert!(is_mtp_source_tensor(name), "{name}");
        }
        assert_eq!(
            TensorNameMap::HfToGgufWithMtp { layer_start: 32 }
                .map_tensor_name("mtp.fc.weight")
                .unwrap(),
            "blk.32.nextn.eh_proj.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGgufWithMtp { layer_start: 32 }
                .map_tensor_name("model.mtp.norm.weight")
                .unwrap(),
            "blk.32.nextn.shared_head_norm.weight"
        );
        assert_eq!(
            TensorNameMap::HfToGgufWithMtp { layer_start: 32 }
                .map_tensor_name("mtp.layers.1.self_attn.q_proj.weight")
                .unwrap(),
            "blk.33.attn_q.weight"
        );
    }

    #[test]
    fn recognizes_and_maps_inkling_mtp_tensors() {
        let map = TensorNameMap::HfToGgufWithMtp { layer_start: 66 };
        for (source, expected) in [
            (
                "model.mtp.layers.0.embed_norm.weight",
                "blk.66.nextn.enorm.weight",
            ),
            (
                "model.mtp.layers.1.hidden_norm.weight",
                "blk.67.nextn.hnorm.weight",
            ),
            (
                "model.mtp.layers.2.input_proj.weight",
                "blk.68.nextn.eh_proj.weight",
            ),
            (
                "model.mtp.layers.3.transformer_block.attn.wq_du.weight",
                "blk.69.attn_q.weight",
            ),
            (
                "model.mtp.layers.4.transformer_block.attn.rel_logits_proj.proj",
                "blk.70.attn_rel_proj.weight",
            ),
            (
                "model.mtp.layers.7.transformer_block.mlp.global_scale",
                "blk.73.ffn_gscale.weight",
            ),
        ] {
            assert!(is_mtp_source_tensor(source));
            assert_eq!(map.map_tensor_name(source).unwrap(), expected);
        }
        assert_eq!(
            map.map_tensor_name("model.llm.embed_norm.weight").unwrap(),
            "token_embd_norm.weight"
        );
        assert!(is_shared_mtp_context_tensor("model.llm.unembed.weight"));
        assert!(is_inkling_fused_w13(
            "model.mtp.layers.0.transformer_block.mlp.w13_dn.weight"
        ));
    }

    #[test]
    fn rejects_inkling_mtp_layer_overflow() {
        let error = TensorNameMap::HfToGgufWithMtp {
            layer_start: u32::MAX,
        }
        .map_tensor_name("model.mtp.layers.1.embed_norm.weight")
        .unwrap_err();

        assert!(error.to_string().contains("MTP layer id overflow"));
    }
}
