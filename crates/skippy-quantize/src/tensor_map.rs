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
        && let Some(normalized) = normalize_qwen_mtp_source_name(name, layer_start)?
    {
        return map_hf_to_gguf(&normalized, mtp_layer_start);
    }
    if name == "model.embed_tokens.weight" {
        return Ok("token_embd.weight".to_string());
    }
    if name == "embed_tokens.weight" {
        return Ok("token_embd.weight".to_string());
    }
    if name == "lm_head.weight" {
        return Ok("output.weight".to_string());
    }
    if name == "model.norm.weight" {
        return Ok("output_norm.weight".to_string());
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
    is_qwen_mtp_source_tensor(name)
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
    )
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
            "self_attn.o_proj.weight" => Ok(format!("blk.{bid}.attn_output.weight")),
            "self_attn.q_norm.weight" => Ok(format!("blk.{bid}.attn_q_norm.weight")),
            "self_attn.k_norm.weight" => Ok(format!("blk.{bid}.attn_k_norm.weight")),
            "self_attn.q_a_proj.weight" => Ok(format!("blk.{bid}.attn_q_a.weight")),
            "self_attn.q_b_proj.weight" => Ok(format!("blk.{bid}.attn_q_b.weight")),
            "self_attn.q_a_layernorm.weight" => Ok(format!("blk.{bid}.attn_q_a_norm.weight")),
            "self_attn.kv_a_proj_with_mqa.weight" => Ok(format!("blk.{bid}.attn_kv_a_mqa.weight")),
            "self_attn.kv_b_proj.weight" => Ok(format!("blk.{bid}.attn_kv_b.weight")),
            "self_attn.kv_a_layernorm.weight" => Ok(format!("blk.{bid}.attn_kv_a_norm.weight")),
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
}
