//! Small GGUF fixture that exercises profile-driven graph construction.

use std::path::Path;

use runtimed_gguf::MetaValue;

use super::{gguf::Pair, graph, weights::Weights};

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn metadata(bytes: &mut Vec<u8>, key: &str, value: &MetaValue) {
    string(bytes, key);
    match value {
        MetaValue::U32(value) => {
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetaValue::Str(value) => {
            bytes.extend_from_slice(&8u32.to_le_bytes());
            string(bytes, value);
        }
        MetaValue::Arr(values) => {
            bytes.extend_from_slice(&9u32.to_le_bytes());
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                let MetaValue::U32(value) = value else {
                    panic!("fixture array is not u32");
                };
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => panic!("fixture metadata type is unsupported"),
    }
}

fn gguf(path: &Path, entries: &[(&str, MetaValue)], tensors: &[(String, Vec<usize>)]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, value) in entries {
        metadata(&mut bytes, key, value);
    }
    let mut data = Vec::new();
    for (name, shape) in tensors {
        string(&mut bytes, name);
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dimension in shape.iter().rev() {
            bytes.extend_from_slice(&(dimension as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        data.resize(data.len().next_multiple_of(32), 0);
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        for _ in 0..shape.iter().product::<usize>() {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}

fn layer(tensors: &mut Vec<(String, Vec<usize>)>, index: usize) {
    let prefix = format!("blk.{index}.");
    for (name, shape) in [
        ("attn_q.weight", vec![8, 8]),
        ("attn_k.weight", vec![4, 8]),
        ("attn_v.weight", vec![4, 8]),
        ("attn_output.weight", vec![8, 8]),
        ("ffn_gate.weight", vec![12, 8]),
        ("ffn_up.weight", vec![12, 8]),
        ("ffn_down.weight", vec![8, 12]),
        ("attn_norm.weight", vec![8]),
        ("attn_q_norm.weight", vec![4]),
        ("attn_k_norm.weight", vec![4]),
        ("ffn_norm.weight", vec![8]),
        ("post_attention_norm.weight", vec![8]),
        ("post_ffw_norm.weight", vec![8]),
        ("layer_output_scale.weight", vec![1]),
    ] {
        tensors.push((format!("{prefix}{name}"), shape));
    }
}

#[test]
fn builds_non_qwen_graph_with_optional_post_norms_and_output_scale() {
    let temp = tempfile::tempdir().unwrap();
    let draft = temp.path().join("draft.gguf");
    let target = temp.path().join("target.gguf");
    let mut tensors = vec![
        ("fc.weight".into(), vec![8, 16]),
        ("enc.output_norm.weight".into(), vec![8]),
        ("output_norm.weight".into(), vec![8]),
    ];
    layer(&mut tensors, 0);
    layer(&mut tensors, 1);
    gguf(
        &draft,
        &[
            ("general.architecture", MetaValue::Str("dflash".into())),
            ("dflash.block_count", MetaValue::U32(2)),
            ("dflash.embedding_length", MetaValue::U32(8)),
            ("dflash.attention.head_count", MetaValue::U32(2)),
            ("dflash.attention.head_count_kv", MetaValue::U32(1)),
            ("dflash.feed_forward_length", MetaValue::U32(12)),
            (
                "dflash.target_layers",
                MetaValue::Arr(vec![MetaValue::U32(0), MetaValue::U32(2)]),
            ),
            ("tokenizer.ggml.mask_token_id", MetaValue::U32(3)),
        ],
        &tensors,
    );
    gguf(
        &target,
        &[("general.architecture", MetaValue::Str("other".into()))],
        &[("token_embd.weight".into(), vec![12, 8])],
    );
    let pair = Pair::open(&draft, &target).unwrap();
    let mask = pair
        .target_embeddings(&temp.path().join("embeddings.f16"))
        .unwrap();
    let mut weights = Weights::create(&pair, &temp.path().join("weights.bin")).unwrap();
    let spec = graph::build(&mut weights, &mask, 8, 3, &pair.profile).unwrap();
    weights.finish().unwrap();
    for name in [
        "block_0_attn_post_norm",
        "block_1_ffn_post_norm",
        "block_1_output_scaled",
    ] {
        assert!(
            spec.windows(name.len())
                .any(|window| window == name.as_bytes()),
            "missing {name}"
        );
    }
}
