//! Nemotron-H MoE MTPv2 folds source fusion/attention block0 and MoE/head block1
//! into one trailing NextN block, unlike Qwen's independent MTP block depths.
use anyhow::{Result, bail};
fn expert_projection(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("experts.") else {
        return false;
    };
    let Some((id, projection)) = rest.split_once('.') else {
        return false;
    };
    !id.is_empty()
        && id.bytes().all(|b| b.is_ascii_digit())
        && id.parse::<u32>().is_ok()
        && matches!(projection, "up_proj.weight" | "down_proj.weight")
}

pub(super) fn normalize(name: &str, layer_start: u32) -> Result<String> {
    let shared = match name {
        "backbone.embeddings.weight" => Some("model.embed_tokens.weight"),
        "backbone.norm_f.weight" => Some("model.norm.weight"),
        "lm_head.weight" => Some("lm_head.weight"),
        _ => None,
    };
    if let Some(shared) = shared {
        return Ok(shared.into());
    }
    let suffix = if let Some(rest) = name.strip_prefix("mtp.layers.0.") {
        match rest {
            "enorm.weight" | "hnorm.weight" | "eh_proj.weight" => rest.to_string(),
            "norm.weight" => "input_layernorm.weight".into(),
            value if value.starts_with("mixer.") => {
                let value = value
                    .strip_prefix("mixer.")
                    .expect("matched one mixer prefix");
                if ![
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "q_proj.bias",
                    "k_proj.bias",
                    "v_proj.bias",
                ]
                .contains(&value)
                {
                    bail!("unsupported Nemotron MTP attention tensor {name}");
                }
                format!("self_attn.{value}")
            }
            _ => bail!("unsupported Nemotron MTP fusion tensor {name}"),
        }
    } else if let Some(rest) = name.strip_prefix("mtp.layers.1.") {
        match rest {
            "norm.weight" => "post_attention_layernorm.weight".into(),
            "final_layernorm.weight" => "shared_head.norm.weight".into(),
            value if value.starts_with("mixer.") => {
                let value = value
                    .strip_prefix("mixer.")
                    .expect("matched one mixer prefix");
                if value == "gate.e_score_correction.bias" {
                    "mlp.gate.e_score_correction_bias".into()
                } else if [
                    "gate.weight",
                    "shared_experts.up_proj.weight",
                    "shared_experts.down_proj.weight",
                    "fc1_latent_proj.weight",
                    "fc2_latent_proj.weight",
                ]
                .contains(&value)
                    || expert_projection(value)
                {
                    format!("mlp.{value}")
                } else {
                    bail!("unsupported Nemotron MTP MoE tensor {name}");
                }
            }
            _ => bail!("unsupported Nemotron MTP head tensor {name}"),
        }
    } else {
        bail!(
            "Nemotron MTP-only map refuses tensor outside source layers0/1/shared context: {name}"
        );
    };
    Ok(format!("model.layers.{layer_start}.{suffix}"))
}
pub(super) fn map(name: &str, layer_start: u32) -> Result<String> {
    let normalized = normalize(name, layer_start)?;
    // Nemotron-H NextN owns ATTN_POST_NORM; the generic Qwen FFN norm name
    // is a different tensor contract and must remain unchanged for that family.
    if normalized.ends_with(".post_attention_layernorm.weight") {
        return Ok(format!("blk.{layer_start}.post_attention_norm.weight"));
    }
    if normalized.ends_with(".mlp.fc1_latent_proj.weight") {
        return Ok(format!("blk.{layer_start}.ffn_latent_down.weight"));
    }
    if normalized.ends_with(".mlp.fc2_latent_proj.weight") {
        return Ok(format!("blk.{layer_start}.ffn_latent_up.weight"));
    }
    super::map_hf_to_gguf(&normalized, None)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nemotron_mtp_folds_attention_and_moe_into_same_nextn_block() {
        for (source, target) in [
            ("mtp.layers.0.enorm.weight", "nextn.enorm.weight"),
            ("mtp.layers.0.hnorm.weight", "nextn.hnorm.weight"),
            ("mtp.layers.0.eh_proj.weight", "nextn.eh_proj.weight"),
            ("mtp.layers.0.mixer.q_proj.weight", "attn_q.weight"),
            ("mtp.layers.1.norm.weight", "post_attention_norm.weight"),
            (
                "mtp.layers.1.final_layernorm.weight",
                "nextn.shared_head_norm.weight",
            ),
            (
                "mtp.layers.1.mixer.gate.e_score_correction.bias",
                "exp_probs_b.bias",
            ),
            (
                "mtp.layers.1.mixer.fc1_latent_proj.weight",
                "ffn_latent_down.weight",
            ),
            (
                "mtp.layers.1.mixer.fc2_latent_proj.weight",
                "ffn_latent_up.weight",
            ),
        ] {
            assert_eq!(map(source, 88).unwrap(), format!("blk.88.{target}"));
        }
        assert_eq!(
            normalize("mtp.layers.1.mixer.experts.3.up_proj.weight", 88).unwrap(),
            "model.layers.88.mlp.experts.3.up_proj.weight"
        );
    }
    #[test]
    fn nemotron_mtp_refuses_unknown_depth_projection_and_trunk_without_changing_qwen() {
        for source in [
            "mtp.layers.2.enorm.weight",
            "mtp.layers.0.mixer.experts.0.up_proj.weight",
            "mtp.layers.1.mixer.q_proj.weight",
            "backbone.layers.0.norm.weight",
            "mtp.layers.0.mixer.mixer.q_proj.weight",
            "mtp.layers.1.mixer.mixer.gate.weight",
            "mtp.layers.1.mixer.experts.0.gate_proj.weight",
            "mtp.layers.1.mixer.experts.-1.up_proj.weight",
        ] {
            assert!(normalize(source, 88).is_err());
        }
        assert_eq!(
            super::super::TensorNameMap::HfToGgufWithMtp { layer_start: 88 }
                .map_tensor_name("mtp.layers.1.self_attn.q_proj.weight")
                .unwrap(),
            "blk.89.attn_q.weight"
        );
    }
}
