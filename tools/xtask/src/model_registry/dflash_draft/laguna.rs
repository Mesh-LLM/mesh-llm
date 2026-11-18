//! Poolside's Laguna DFlash drafter (`DFlashLagunaForCausalLM`) as the
//! patched llama.cpp converter writes it: `dflash` architecture,
//! `dflash.decoder_arch = laguna`, the per-feature aux norms stacked into
//! `enc.aux_norm`, the fused qkv projection split, and the attention gate.
//! Config keys the converter does not know are refused, so a new drafter
//! is never converted with metadata silently missing.

use std::path::Path;

use serde_json::Value as Json;

use super::gguf::{TensorEntry, TensorType, Value};
use super::safetensors::Safetensors;
use crate::command::DynResult;

/// Every draft config.json key this conversion accounts for.
const KNOWN_KEYS: &[&str] = &[
    "architectures",
    "attention_bias",
    "dflash_config",
    "draft_vocab_size",
    "eagle_aux_hidden_state_layer_ids",
    "gating",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_experts",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_theta",
    "sliding_window",
    "torch_dtype",
    "vocab_size",
];
const KNOWN_DFLASH_KEYS: &[&str] = &[
    "block_size",
    "causal",
    "mask_token_id",
    "num_target_layers",
    "target_layer_ids",
];

/// Where a GGUF tensor's values come from.
#[derive(Clone, Debug)]
pub(super) enum Source {
    Tensor(String),
    /// Whole tensors, concatenated in order.
    Stack(Vec<String>),
    /// `rows` rows of a 2-D tensor starting at row `start`.
    Rows {
        name: String,
        start: u64,
        rows: u64,
    },
}

pub(super) struct Plan {
    pub(super) metadata: Vec<(String, Value)>,
    pub(super) tensors: Vec<(TensorEntry, Source)>,
    pub(super) vocab_size: usize,
    pub(super) mask_token_id: Option<u32>,
}

fn u64_of(config: &Json, key: &str) -> DynResult<u64> {
    config
        .get(key)
        .and_then(Json::as_u64)
        .ok_or_else(|| format!("draft config.json needs integer {key}").into())
}

fn f32_of(config: &Json, key: &str) -> DynResult<f32> {
    Ok(config
        .get(key)
        .and_then(Json::as_f64)
        .ok_or_else(|| format!("draft config.json needs number {key}"))? as f32)
}

fn u32_value(value: u64, key: &str) -> DynResult<Value> {
    Ok(Value::U32(
        u32::try_from(value).map_err(|_| format!("{key} does not fit in u32"))?,
    ))
}

/// llama.cpp's size label: parameters scaled to K/M/B/T with at least two
/// significant digits.
pub(super) fn size_label(parameters: u64) -> String {
    let count = parameters as f64;
    let (scaled, suffix) = if count > 1e12 {
        (count * 1e-12, "T")
    } else if count > 1e9 {
        (count * 1e-9, "B")
    } else if count > 1e6 {
        (count * 1e-6, "M")
    } else {
        (count * 1e-3, "K")
    };
    let digits = match scaled.round_ties_even() as u64 {
        0 => 0,
        rounded => rounded.to_string().len(),
    };
    let decimals = 2usize.saturating_sub(digits);
    format!("{scaled:.decimals$}{suffix}")
}

/// The GGUF tensor for checkpoint tensor `name`, unless it is an aux norm
/// that joins the stack.
fn tensor_name(name: &str) -> DynResult<Option<&'static str>> {
    Ok(Some(match name {
        "fc.weight" => "fc.weight",
        "hidden_norm.weight" => "enc.output_norm.weight",
        "norm.weight" => "output_norm.weight",
        _ if name.starts_with("aux_hidden_norms.") => return Ok(None),
        _ => {
            let suffix = name
                .strip_prefix("layers.")
                .and_then(|rest| rest.split_once('.'))
                .map(|(_, suffix)| suffix)
                .ok_or_else(|| format!("unexpected draft tensor {name}"))?;
            match suffix {
                "input_layernorm.weight" => "attn_norm.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.g_proj.weight" => "attn_gate.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.qkv_proj.weight" => "attn_qkv.weight",
                _ => return Err(format!("unexpected draft tensor {name}").into()),
            }
        }
    }))
}

fn layer_of(name: &str) -> DynResult<u64> {
    name.strip_prefix("layers.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(layer, _)| layer.parse().ok())
        .ok_or_else(|| format!("no layer index in {name}").into())
}

/// Norms and vectors stay F32; matrices become Q8_0.
fn entry(name: String, shape: &[u64]) -> DynResult<TensorEntry> {
    let dims: Vec<u64> = shape.iter().rev().copied().collect();
    let kind = if dims.len() <= 1 || name.ends_with("_norm.weight") {
        TensorType::F32
    } else if dims[0].is_multiple_of(32) {
        TensorType::Q8_0
    } else {
        return Err(format!("{name}: rows of {} do not split into Q8_0 blocks", dims[0]).into());
    };
    Ok(TensorEntry { name, dims, kind })
}

/// The dimensions the checkpoint's tensors must agree with.
struct Shape {
    layers: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    aux: u64,
}

/// Checks the checkpoint holds exactly the tensors a Laguna drafter of this
/// config has, each with its shape, so a hash-matching output is also one
/// the DFlash loader accepts.
fn check_inventory(config: &Json, checkpoint: &Safetensors, shape: Shape) -> DynResult<()> {
    let hidden = u64_of(config, "hidden_size")?;
    let ffn = u64_of(config, "intermediate_size")?;
    let Shape {
        layers,
        heads,
        kv_heads,
        head_dim,
        aux,
    } = shape;
    if [hidden, ffn, heads, kv_heads, head_dim, aux, layers].contains(&0) {
        return Err("draft config.json has a zero dimension".into());
    }
    let mut expected: Vec<(String, Vec<Vec<u64>>)> = vec![
        ("fc.weight".into(), vec![vec![hidden, aux * hidden]]),
        ("hidden_norm.weight".into(), vec![vec![hidden]]),
        ("norm.weight".into(), vec![vec![hidden]]),
    ];
    for i in 0..aux {
        expected.push((format!("aux_hidden_norms.{i}.weight"), vec![vec![hidden]]));
    }
    for layer in 0..layers {
        let tensor = |suffix: &str| format!("layers.{layer}.{suffix}.weight");
        expected.extend([
            (tensor("input_layernorm"), vec![vec![hidden]]),
            (tensor("post_attention_layernorm"), vec![vec![hidden]]),
            (tensor("mlp.down_proj"), vec![vec![hidden, ffn]]),
            (tensor("mlp.gate_proj"), vec![vec![ffn, hidden]]),
            (tensor("mlp.up_proj"), vec![vec![ffn, hidden]]),
            // Per-head or per-element gating.
            (
                tensor("self_attn.g_proj"),
                vec![vec![heads, hidden], vec![heads * head_dim, hidden]],
            ),
            (tensor("self_attn.q_norm"), vec![vec![head_dim]]),
            (tensor("self_attn.k_norm"), vec![vec![head_dim]]),
            (
                tensor("self_attn.o_proj"),
                vec![vec![hidden, heads * head_dim]],
            ),
            (
                tensor("self_attn.qkv_proj"),
                vec![vec![(heads + 2 * kv_heads) * head_dim, hidden]],
            ),
        ]);
    }
    for (name, shapes) in &expected {
        let info = checkpoint
            .tensors()
            .get(name)
            .ok_or_else(|| format!("the checkpoint has no {name}"))?;
        if !shapes.contains(&info.shape) {
            return Err(format!(
                "{name} has shape {:?}, expected {:?}",
                info.shape, shapes[0]
            )
            .into());
        }
    }
    if let Some(extra) = checkpoint
        .tensors()
        .keys()
        .find(|name| !expected.iter().any(|(expected, _)| expected == *name))
    {
        return Err(format!("unexpected draft tensor {extra}").into());
    }
    Ok(())
}

pub(super) fn plan(source: &Path, checkpoint: &Safetensors, name: &str) -> DynResult<Plan> {
    let bytes = std::fs::read(source.join("config.json"))?;
    let config: Json = serde_json::from_slice(&bytes)?;
    let object = config
        .as_object()
        .ok_or("draft config.json is not an object")?;
    if let Some(key) = object
        .keys()
        .find(|key| !KNOWN_KEYS.contains(&key.as_str()))
    {
        return Err(
            format!("draft config.json key {key:?} is not handled by this converter").into(),
        );
    }
    let dflash = config.get("dflash_config").cloned().unwrap_or(Json::Null);
    if let Some(key) = dflash.as_object().and_then(|d| {
        d.keys()
            .find(|key| !KNOWN_DFLASH_KEYS.contains(&key.as_str()))
    }) {
        return Err(format!("dflash_config key {key:?} is not handled by this converter").into());
    }
    if config.pointer("/architectures/0").and_then(Json::as_str) != Some("DFlashLagunaForCausalLM")
        || config.get("model_type").and_then(Json::as_str) != Some("laguna")
        || config
            .get("attention_bias")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        || config
            .get("hidden_act")
            .and_then(Json::as_str)
            .unwrap_or("silu")
            != "silu"
        || config
            .get("num_experts")
            .and_then(Json::as_u64)
            .unwrap_or(0)
            != 0
    {
        return Err("not a dense Laguna DFlash drafter".into());
    }

    let layers = u64_of(&config, "num_hidden_layers")?;
    let heads = u64_of(&config, "num_attention_heads")?;
    let kv_heads = u64_of(&config, "num_key_value_heads")?;
    let head_dim = u64_of(&config, "head_dim")?;
    let vocab_size = u64_of(&config, "vocab_size")? as usize;
    if config
        .get("draft_vocab_size")
        .and_then(Json::as_u64)
        .is_some_and(|draft| draft as usize != vocab_size)
    {
        return Err("reduced draft vocabularies are not supported".into());
    }
    let target_layers: Vec<i32> = dflash
        .get("target_layer_ids")
        .and_then(Json::as_array)
        .ok_or("dflash_config needs target_layer_ids")?
        .iter()
        .map(|id| {
            id.as_u64()
                .and_then(|id| i32::try_from(id).ok())
                .and_then(|id| id.checked_add(1))
                .ok_or("bad target layer id")
        })
        .collect::<Result<_, _>>()?;
    check_inventory(
        &config,
        checkpoint,
        Shape {
            layers,
            heads,
            kv_heads,
            head_dim,
            aux: target_layers.len() as u64,
        },
    )?;

    let parameters: u64 = checkpoint
        .tensors()
        .values()
        .map(|t| t.element_count())
        .sum();
    let mut metadata = vec![
        ("general.architecture", Value::Str("dflash".into())),
        ("general.type", Value::Str("model".into())),
        ("general.name", Value::Str(name.into())),
        ("general.size_label", Value::Str(size_label(parameters))),
        (
            "dflash.block_count",
            u32_value(layers, "num_hidden_layers")?,
        ),
        (
            "dflash.context_length",
            u32_value(
                u64_of(&config, "max_position_embeddings")?,
                "max_position_embeddings",
            )?,
        ),
        (
            "dflash.embedding_length",
            u32_value(u64_of(&config, "hidden_size")?, "hidden_size")?,
        ),
        (
            "dflash.feed_forward_length",
            u32_value(u64_of(&config, "intermediate_size")?, "intermediate_size")?,
        ),
        (
            "dflash.attention.head_count",
            u32_value(heads, "num_attention_heads")?,
        ),
        (
            "dflash.attention.head_count_kv",
            u32_value(kv_heads, "num_key_value_heads")?,
        ),
        (
            "dflash.rope.freq_base",
            Value::F32(f32_of(&config, "rope_theta")?),
        ),
        (
            "dflash.attention.layer_norm_rms_epsilon",
            Value::F32(f32_of(&config, "rms_norm_eps")?),
        ),
        (
            "dflash.attention.key_length",
            u32_value(head_dim, "head_dim")?,
        ),
        (
            "dflash.attention.value_length",
            u32_value(head_dim, "head_dim")?,
        ),
        // LLAMA_FTYPE_MOSTLY_Q8_0
        ("general.file_type", Value::U32(7)),
        (
            "dflash.block_size",
            u32_value(
                dflash
                    .get("block_size")
                    .and_then(Json::as_u64)
                    .unwrap_or(16),
                "block_size",
            )?,
        ),
        ("dflash.target_layers", Value::I32s(target_layers.clone())),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect::<Vec<_>>();
    if let Some(causal) = dflash.get("causal").and_then(Json::as_bool) {
        metadata.push(("dflash.attention.causal".into(), Value::Bool(causal)));
    }
    metadata.push(("dflash.decoder_arch".into(), Value::Str("laguna".into())));
    let layer_types: Vec<&str> = config
        .get("layer_types")
        .and_then(Json::as_array)
        .map(|types| types.iter().filter_map(Json::as_str).collect())
        .unwrap_or_default();
    let window = config
        .get("sliding_window")
        .and_then(Json::as_u64)
        .unwrap_or(0);
    if window > 0 && layer_types.contains(&"sliding_attention") {
        if layer_types.len() as u64 != layers {
            return Err("layer_types does not name every layer".into());
        }
        metadata.push((
            "dflash.attention.sliding_window".into(),
            u32_value(window, "sliding_window")?,
        ));
        metadata.push((
            "dflash.attention.sliding_window_pattern".into(),
            Value::Bools(
                layer_types
                    .iter()
                    .map(|t| *t == "sliding_attention")
                    .collect(),
            ),
        ));
    }
    metadata.push(("general.quantization_version".into(), Value::U32(2)));

    // Tensors in checkpoint name order; the aux norms join one stacked
    // tensor where the last of them sorts.
    let aux_count = target_layers.len();
    let aux: Vec<String> = (0..aux_count)
        .map(|i| format!("aux_hidden_norms.{i}.weight"))
        .collect();
    let mut tensors = Vec::new();
    let mut aux_seen = 0;
    for (source_name, info) in checkpoint.tensors() {
        if source_name.starts_with("aux_hidden_norms.") {
            if !aux.contains(source_name) || info.shape.len() != 1 {
                return Err(format!("unexpected aux norm {source_name}").into());
            }
            aux_seen += 1;
            if aux_seen == aux_count {
                let width = info.shape[0];
                tensors.push((
                    entry("enc.aux_norm.weight".into(), &[aux_count as u64, width])?,
                    Source::Stack(aux.clone()),
                ));
            }
            continue;
        }
        let Some(gguf_name) = tensor_name(source_name)? else {
            continue;
        };
        if gguf_name == "attn_qkv.weight" {
            let layer = layer_of(source_name)?;
            let q_rows = heads * head_dim;
            let kv_rows = kv_heads * head_dim;
            if info.shape.len() != 2 || info.shape[0] != q_rows + 2 * kv_rows {
                return Err(format!("{source_name}: unexpected fused qkv shape").into());
            }
            let width = info.shape[1];
            for (part, start, rows) in [
                ("attn_q", 0, q_rows),
                ("attn_k", q_rows, kv_rows),
                ("attn_v", q_rows + kv_rows, kv_rows),
            ] {
                tensors.push((
                    entry(format!("blk.{layer}.{part}.weight"), &[rows, width])?,
                    Source::Rows {
                        name: source_name.clone(),
                        start,
                        rows,
                    },
                ));
            }
            continue;
        }
        let full_name = if source_name.starts_with("layers.") {
            format!("blk.{}.{gguf_name}", layer_of(source_name)?)
        } else {
            gguf_name.to_string()
        };
        tensors.push((
            entry(full_name, &info.shape)?,
            Source::Tensor(source_name.clone()),
        ));
    }
    if aux_seen != aux_count {
        return Err(format!("expected {aux_count} aux norms, found {aux_seen}").into());
    }
    let mask_token_id = dflash
        .get("mask_token_id")
        .and_then(Json::as_u64)
        .map(|id| match u32::try_from(id) {
            Ok(id) if (id as usize) < vocab_size => Ok(id),
            _ => Err("mask_token_id is outside the vocabulary"),
        })
        .transpose()?;
    Ok(Plan {
        metadata,
        tensors,
        vocab_size,
        mask_token_id,
    })
}
