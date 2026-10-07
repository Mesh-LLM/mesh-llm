//! Native Nemotron-H MoE single folded NextN metadata and explicit tokenizer custody.
mod profile;
use crate::{gguf_metadata::GgufKv, tensor_map::TensorNameMap};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;
const ARCH: &str = "nemotron_h_moe";
fn uint(config: &Value, keys: &[&str]) -> Result<u32> {
    let mut found = None;
    for key in keys {
        if let Some(value) = config.get(*key) {
            let value = u32::try_from(
                value
                    .as_u64()
                    .context("Nemotron metadata dimension must be unsigned integer")?,
            )?;
            ensure!(value > 0, "Nemotron metadata dimension must be positive");
            ensure!(
                found.is_none_or(|prior| prior == value),
                "conflicting Nemotron dimension aliases"
            );
            found = Some(value);
        }
    }
    found.context("Nemotron metadata missing required integer")
}
fn finite(config: &Value, key: &str) -> Result<f32> {
    let value = config
        .get(key)
        .and_then(Value::as_f64)
        .context("Nemotron metadata missing scalar")? as f32;
    ensure!(
        value.is_finite() && value > 0.0,
        "Nemotron scalar must be positive finite"
    );
    Ok(value)
}
fn merged(config: &Value) -> Result<Value> {
    let mut merged = config
        .as_object()
        .context("Nemotron config object")?
        .clone();
    if let Some(inner) = config.get("llm_config").filter(|value| !value.is_null()) {
        for (key, value) in inner.as_object().context("llm_config object")? {
            merged.insert(key.clone(), value.clone());
        }
    }
    let config = Value::Object(merged);
    ensure!(
        matches!(
            config["model_type"].as_str(),
            Some("nemotron_h" | "nemotron_h_moe")
        ),
        "explicit profile requires Nemotron-H MoE config"
    );
    ensure!(
        config.get("block_configs").is_none(),
        "Puzzle heterogeneous config is outside this profile"
    );
    ensure!(
        uint(&config, &["num_nextn_predict_layers"])? == 1,
        "Nemotron MTP requires one NextN block"
    );
    Ok(config)
}
fn pattern(config: &Value) -> Result<Vec<char>> {
    let value = config
        .get("hybrid_override_pattern")
        .filter(|value| !value.is_null())
        .or_else(|| config.get("layers_block_type"))
        .context("Nemotron layer pattern required")?;
    let layers = if let Some(text) = value.as_str() {
        text.chars().collect::<Vec<_>>()
    } else {
        value
            .as_array()
            .context("Nemotron layer pattern array/string")?
            .iter()
            .map(|value| match value.as_str() {
                Some("mamba" | "linear_attention") => Ok('M'),
                Some("attention" | "full_attention") => Ok('*'),
                Some("moe") => Ok('E'),
                _ => Err(anyhow::anyhow!("unsupported Nemotron layer pattern")),
            })
            .collect::<Result<Vec<_>>>()?
    };
    ensure!(
        layers.len() == uint(config, &["num_hidden_layers"])? as usize
            && layers.len() <= 65535
            && layers.iter().all(|c| matches!(c, 'M' | '*' | 'E')),
        "Nemotron pattern/count refused"
    );
    Ok(layers)
}
fn key(name: &str) -> String {
    format!("{ARCH}.{name}")
}
fn core(config: &Value, layers: &[char], count: usize) -> Result<Vec<GgufKv>> {
    let heads = uint(config, &["num_attention_heads"])?;
    let kv = uint(config, &["num_key_value_heads"])?;
    ensure!(
        heads >= kv && heads.is_multiple_of(kv),
        "Nemotron head ratio refused"
    );
    let ffn = uint(config, &["moe_intermediate_size"])?;
    let mut kv_counts = layers
        .iter()
        .map(|c| if *c == '*' { kv } else { 0 })
        .collect::<Vec<_>>();
    kv_counts.push(kv);
    let mut ffn_counts = layers
        .iter()
        .map(|c| if *c == 'E' { ffn } else { 0 })
        .collect::<Vec<_>>();
    ffn_counts.push(ffn);
    let norm = config
        .get("layer_norm_epsilon")
        .or_else(|| config.get("norm_eps"))
        .and_then(Value::as_f64)
        .context("Nemotron norm epsilon required")? as f32;
    ensure!(
        norm.is_finite() && norm > 0.0,
        "Nemotron norm epsilon refused"
    );
    let head_dim = uint(config, &["head_dim", "attention_head_dim"])?;
    let mut metadata = vec![
        GgufKv::string("general.architecture", ARCH),
        GgufKv::string("general.name", "Nemotron single folded MTP head"),
        GgufKv::u64("skippy.convert.tensor_count", count as u64),
        GgufKv::u32(
            &key("block_count"),
            u32::try_from(layers.len())?
                .checked_add(1)
                .context("block count overflow")?,
        ),
        GgufKv::u32(&key("nextn_predict_layers"), 1),
        GgufKv::u32(&key("context_length"), 1 << 20),
        GgufKv::u32(&key("embedding_length"), uint(config, &["hidden_size"])?),
        GgufKv::array_u32(&key("feed_forward_length"), ffn_counts),
        GgufKv::u32(&key("attention.head_count"), heads),
        GgufKv::array_u32(&key("attention.head_count_kv"), kv_counts),
        GgufKv::u32(&key("attention.key_length"), head_dim),
        GgufKv::u32(&key("attention.value_length"), head_dim),
        GgufKv::f32(&key("attention.layer_norm_epsilon"), norm),
        GgufKv::f32(&key("attention.layer_norm_rms_epsilon"), norm),
        GgufKv::bool(&key("rope.scaling.finetuned"), false),
    ];
    experts(config, &mut metadata)?;
    Ok(metadata)
}
fn experts(config: &Value, metadata: &mut Vec<GgufKv>) -> Result<()> {
    let count = uint(config, &["n_routed_experts"])?;
    let used = uint(config, &["num_experts_per_tok"])?;
    ensure!(used <= count, "expert top-k exceeds roster");
    for (name, field) in [
        ("expert_count", "n_routed_experts"),
        ("expert_used_count", "num_experts_per_tok"),
        ("expert_feed_forward_length", "moe_intermediate_size"),
        (
            "expert_shared_feed_forward_length",
            "moe_shared_expert_intermediate_size",
        ),
        ("expert_shared_count", "n_shared_experts"),
        ("expert_group_count", "n_group"),
    ] {
        metadata.push(GgufKv::u32(&key(name), uint(config, &[field])?));
    }
    metadata.push(GgufKv::bool(
        &key("expert_weights_norm"),
        config["norm_topk_prob"]
            .as_bool()
            .context("norm_topk_prob required")?,
    ));
    metadata.push(GgufKv::f32(
        &key("expert_weights_scale"),
        finite(config, "routed_scaling_factor")?,
    ));
    Ok(())
}
fn ssm(config: &Value, metadata: &mut Vec<GgufKv>) -> Result<()> {
    let rank = uint(config, &["n_heads", "num_heads"])?;
    let width = if config.get("mamba_head_dim").is_some() {
        uint(config, &["mamba_head_dim"])?
    } else {
        uint(config, &["hidden_size", "d_model"])?
    };
    let inner = rank
        .checked_mul(width)
        .context("Nemotron SSM inner size overflow")?;
    let groups = uint(config, &["n_groups", "num_groups", "mamba_n_groups"])?;
    ensure!(
        inner.is_multiple_of(groups),
        "Nemotron SSM grouping refused"
    );
    for (name, value) in [
        (
            "conv_kernel",
            uint(config, &["conv_kernel", "mamba_d_conv"])?,
        ),
        (
            "state_size",
            uint(config, &["ssm_state_size", "mamba_d_state"])?,
        ),
        ("inner_size", inner),
        ("time_step_rank", rank),
        ("group_count", groups),
    ] {
        metadata.push(GgufKv::u32(&key(&format!("ssm.{name}")), value));
    }
    if config.get("moe_latent_size").is_some() {
        metadata.push(GgufKv::u32(
            &key("moe_latent_size"),
            uint(config, &["moe_latent_size"])?,
        ));
    }
    Ok(())
}
/// Operator profile binds exact source bytes; it does not attest tokenizer behavior.
pub fn prepare(
    source: &Path,
    profile_path: &Path,
    count: usize,
) -> Result<(Vec<GgufKv>, TensorNameMap)> {
    let bound = profile::load(source, profile_path)?;
    let config = merged(&bound.config)?;
    let layers = pattern(&config)?;
    let mut metadata = core(&config, &layers, count)?;
    ssm(&config, &mut metadata)?;
    let mut tokenizer_config = config.clone();
    let vocab = uint(&config, &["vocab_size"])?;
    let padded = vocab.checked_add(7).context("vocab padding overflow")? / 8 * 8;
    tokenizer_config["vocab_size"] = serde_json::json!(padded);
    crate::tokenizer_metadata::push_bound_tokenizer_metadata(
        &mut metadata,
        &tokenizer_config,
        &bound.tokenizer,
        &bound.tokenizer_config,
        bound.pre.name(),
        bound.template.as_deref(),
    )?;
    metadata.push(GgufKv::string(
        "skippy.convert.tokenizer_profile_sha256",
        &bound.profile_sha256,
    ));
    metadata.push(GgufKv::string(
        "skippy.convert.tokenizer_profile_scope",
        "explicit source-byte-bound operator profile; real tokenizer equivalence unqualified",
    ));
    Ok((
        metadata,
        TensorNameMap::NemotronHMoeMtp {
            layer_start: u32::try_from(layers.len())?,
        },
    ))
}
#[cfg(test)]
#[path = "nemotron_mtp/tests.rs"]
mod tests;

/// Check the full pre-split source roster before any output file is created.
pub(crate) fn validate_roster<'a>(
    metadata: &[GgufKv],
    names: impl Iterator<Item = &'a str>,
) -> Result<()> {
    use std::collections::BTreeSet;
    let count = metadata
        .iter()
        .find_map(|kv| match kv {
            GgufKv::U32 { key, value } if key == "nemotron_h_moe.expert_count" => Some(*value),
            _ => None,
        })
        .context("Nemotron expert count required")?;
    let mut roster = BTreeSet::new();
    for name in names {
        let selected = name.starts_with("mtp.") || name.starts_with("lm_head.");
        ensure!(
            !selected
                || ![
                    ".weight_scale",
                    ".weight_scale_2",
                    ".weight_scale_inv",
                    ".input_scale",
                    ".input_global_scale",
                    ".weight_global_scale",
                    ".weight_packed",
                ]
                .iter()
                .any(|suffix| name.ends_with(suffix)),
            "Nemotron native BF16 profile refuses unsupported scaled/packed source: {name}"
        );
        if !name.starts_with("mtp.") {
            continue;
        }
        ensure!(roster.insert(name), "duplicate Nemotron source tensor");
    }
    for required in [
        "mtp.layers.0.enorm.weight",
        "mtp.layers.0.hnorm.weight",
        "mtp.layers.0.eh_proj.weight",
        "mtp.layers.0.norm.weight",
        "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.0.mixer.k_proj.weight",
        "mtp.layers.0.mixer.v_proj.weight",
        "mtp.layers.0.mixer.o_proj.weight",
        "mtp.layers.1.norm.weight",
        "mtp.layers.1.final_layernorm.weight",
        "mtp.layers.1.mixer.gate.weight",
        "mtp.layers.1.mixer.gate.e_score_correction.bias",
        "mtp.layers.1.mixer.shared_experts.up_proj.weight",
        "mtp.layers.1.mixer.shared_experts.down_proj.weight",
    ] {
        ensure!(
            roster.contains(required),
            "Nemotron required NextN projection absent: {required}"
        );
    }
    let declared_latent = metadata.iter().any(|kv| {
        matches!(kv,
        GgufKv::U32 { key, value } if key == "nemotron_h_moe.moe_latent_size" && *value > 0)
    });
    let latent_down = roster.contains("mtp.layers.1.mixer.fc1_latent_proj.weight");
    let latent_up = roster.contains("mtp.layers.1.mixer.fc2_latent_proj.weight");
    ensure!(
        if declared_latent {
            latent_down && latent_up
        } else {
            !latent_down && !latent_up
        },
        "Nemotron latent metadata requires both projections; undeclared latent tensors refused"
    );
    for projection in ["up_proj.weight", "down_proj.weight"] {
        let ids = roster
            .iter()
            .filter_map(|name| name.strip_prefix("mtp.layers.1.mixer.experts."))
            .filter_map(|rest| rest.split_once('.'))
            .filter(|(_, suffix)| *suffix == projection)
            .map(|(id, _)| id.parse::<u32>())
            .collect::<std::result::Result<BTreeSet<_>, _>>()?;
        ensure!(
            ids.len() as u64 == u64::from(count) && ids.iter().copied().eq(0..count),
            "Nemotron expert projection roster differs from configured count"
        );
    }
    Ok(())
}
