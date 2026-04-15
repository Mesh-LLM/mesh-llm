use super::super::{ModelConfig, TensorPrefixes};
use anyhow::{bail, Context, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use std::collections::HashMap;

fn slice_rows(tensor: &Array, start: i32, end: i32) -> Result<Array> {
    Ok(tensor.index((start..end, std::ops::RangeFull)))
}

fn split_fused_qkv(
    prefix: &str,
    tensors: &mut HashMap<String, Array>,
    q_rows: i32,
    kv_rows: i32,
) -> Result<()> {
    if tensors.contains_key(&format!("{prefix}.q_proj.weight")) {
        return Ok(());
    }

    let total_rows = q_rows + kv_rows + kv_rows;
    for suffix in ["weight", "scales", "biases"] {
        let key = format!("{prefix}.qkv_proj.{suffix}");
        let fused = tensors
            .get(&key)
            .cloned()
            .with_context(|| format!("missing {key}"))?;
        let shape = fused.shape();
        if shape.is_empty() || shape[0] != total_rows {
            bail!(
                "unexpected {key} shape {:?}; expected first dimension {}",
                shape,
                total_rows,
            );
        }
        tensors.insert(
            format!("{prefix}.q_proj.{suffix}"),
            slice_rows(&fused, 0, q_rows)?,
        );
        tensors.insert(
            format!("{prefix}.k_proj.{suffix}"),
            slice_rows(&fused, q_rows, q_rows + kv_rows)?,
        );
        tensors.insert(
            format!("{prefix}.v_proj.{suffix}"),
            slice_rows(&fused, q_rows + kv_rows, total_rows)?,
        );
    }

    Ok(())
}

fn split_fused_gate_up(
    prefix: &str,
    tensors: &mut HashMap<String, Array>,
    hidden_rows: i32,
) -> Result<()> {
    if tensors.contains_key(&format!("{prefix}.gate_proj.weight")) {
        return Ok(());
    }

    let total_rows = hidden_rows * 2;
    for suffix in ["weight", "scales", "biases"] {
        let key = format!("{prefix}.gate_up_proj.{suffix}");
        let fused = tensors
            .get(&key)
            .cloned()
            .with_context(|| format!("missing {key}"))?;
        let shape = fused.shape();
        if shape.is_empty() || shape[0] != total_rows {
            bail!(
                "unexpected {key} shape {:?}; expected first dimension {}",
                shape,
                total_rows,
            );
        }
        tensors.insert(
            format!("{prefix}.gate_proj.{suffix}"),
            slice_rows(&fused, 0, hidden_rows)?,
        );
        tensors.insert(
            format!("{prefix}.up_proj.{suffix}"),
            slice_rows(&fused, hidden_rows, total_rows)?,
        );
    }

    Ok(())
}

pub(crate) fn transform_phi3_tensors(
    tensors: &mut HashMap<String, Array>,
    prefixes: &TensorPrefixes,
    config: &ModelConfig,
) -> Result<()> {
    let head_dim = config
        .head_dim
        .unwrap_or_else(|| config.hidden_size / config.num_attention_heads);
    let q_rows = config.num_attention_heads * head_dim;
    let kv_rows = config.num_key_value_heads * head_dim;
    let mlp_rows = config.intermediate_size;

    for i in 0..config.num_hidden_layers {
        let attn_prefix = format!("{}.layers.{i}.self_attn", prefixes.model);
        if tensors.contains_key(&format!("{attn_prefix}.qkv_proj.weight")) {
            split_fused_qkv(&attn_prefix, tensors, q_rows, kv_rows)?;
        }

        let mlp_prefix = format!("{}.layers.{i}.mlp", prefixes.model);
        if tensors.contains_key(&format!("{mlp_prefix}.gate_up_proj.weight")) {
            split_fused_gate_up(&mlp_prefix, tensors, mlp_rows)?;
        }
    }

    Ok(())
}
