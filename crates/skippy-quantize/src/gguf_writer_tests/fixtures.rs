fn unique_temp_dir() -> PathBuf {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("skippy-gguf-writer-{nanos}-{id}"))
}

fn minimal_glm_dsa_metadata(block_count: u32, nextn_layers: u32) -> Vec<GgufKv> {
    let mut metadata = vec![
        GgufKv::string("general.architecture", "glm-dsa"),
        GgufKv::u32("glm-dsa.block_count", block_count),
    ];
    if nextn_layers > 0 {
        metadata.push(GgufKv::u32("glm-dsa.nextn_predict_layers", nextn_layers));
    }
    metadata
}

fn glm_dsa_kv_b_split_metadata() -> Vec<GgufKv> {
    vec![
        GgufKv::string("general.architecture", "glm-dsa"),
        GgufKv::u32("glm-dsa.block_count", 1),
        GgufKv::u32("glm-dsa.attention.head_count", 1),
        GgufKv::u32("glm-dsa.attention.key_length", 3),
        GgufKv::u32("glm-dsa.attention.key_length_mla", 4),
        GgufKv::u32("glm-dsa.rope.dimension_count", 1),
        GgufKv::u32("glm-dsa.attention.value_length", 3),
        GgufKv::u32("glm-dsa.attention.kv_lora_rank", 2),
    ]
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_safetensor(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
    let mut offset = 0_u64;
    let mut entries = serde_json::Map::new();
    for (name, dtype, shape, bytes) in tensors {
        let end = offset + bytes.len() as u64;
        entries.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let header = serde_json::Value::Object(entries).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for (_, _, _, tensor_bytes) in tensors {
        bytes.extend_from_slice(tensor_bytes);
    }
    fs::write(path, bytes).unwrap();
}

fn write_qwen_config_and_tokenizer(root: &Path) {
    fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "qwen3",
          "vocab_size": 4,
          "max_position_embeddings": 128,
          "hidden_size": 4,
          "intermediate_size": 8,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "rope_theta": 1000000,
          "rms_norm_eps": 1e-6
        }"#,
    )
    .unwrap();
    fs::write(
        root.join("tokenizer.json"),
        r#"{
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "<|endoftext|>": 2, "<|im_end|>": 3},
            "merges": ["a b"]
          },
          "decoder": {"type": "ByteLevel"},
          "added_tokens": [
            {"id": 2, "content": "<|endoftext|>", "special": true},
            {"id": 3, "content": "<|im_end|>", "special": true}
          ]
        }"#,
    )
    .unwrap();
    fs::write(
        root.join("tokenizer_config.json"),
        r#"{"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "add_bos_token": false}"#,
    )
    .unwrap();
}

fn write_qwen2_moe_config_and_tokenizer(root: &Path) {
    write_qwen_config_and_tokenizer(root);
    fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "qwen2_moe",
          "vocab_size": 4,
          "max_position_embeddings": 128,
          "hidden_size": 4,
          "intermediate_size": 8,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "moe_intermediate_size": 2,
          "shared_expert_intermediate_size": 8,
          "rope_theta": 1000000,
          "rms_norm_eps": 1e-6
        }"#,
    )
    .unwrap();
}

fn write_qwen3_moe_config_and_tokenizer(root: &Path) {
    write_qwen_config_and_tokenizer(root);
    fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "qwen3_moe",
          "vocab_size": 4,
          "max_position_embeddings": 128,
          "hidden_size": 4,
          "intermediate_size": 8,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "moe_intermediate_size": 2,
          "rope_theta": 1000000,
          "rms_norm_eps": 1e-6
        }"#,
    )
    .unwrap();
}

fn write_glm_dsa_config_and_tokenizer(root: &Path) {
    fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "glm_moe_dsa",
          "vocab_size": 8,
          "max_position_embeddings": 128,
          "hidden_size": 4,
          "intermediate_size": 8,
          "num_hidden_layers": 3,
          "num_nextn_predict_layers": 1,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "qk_nope_head_dim": 3,
          "qk_rope_head_dim": 2,
          "v_head_dim": 2,
          "q_lora_rank": 2,
          "kv_lora_rank": 2,
          "index_n_heads": 1,
          "index_head_dim": 4,
          "index_topk": 2,
          "index_topk_freq": 2,
          "index_skip_topk_offset": 1,
          "indexer_types": ["full", "shared", "full"],
          "n_routed_experts": 2,
          "num_experts_per_tok": 1,
          "n_shared_experts": 1,
          "moe_intermediate_size": 2,
          "first_k_dense_replace": 1,
          "routed_scaling_factor": 2.5,
          "norm_topk_prob": true,
          "rms_norm_eps": 1e-5
        }"#,
    )
    .unwrap();
    fs::write(
        root.join("tokenizer.json"),
        r#"{
          "model": {
            "type": "BPE",
            "vocab": {
              "a": 0,
              "b": 1,
              "[gMASK]": 2,
              "<|user|>": 3,
              "<|observation|>": 4,
              "<|endoftext|>": 5,
              "<|assistant|>": 6,
              "<|system|>": 7
            },
            "merges": ["a b"]
          },
          "decoder": {"type": "ByteLevel"},
          "added_tokens": [
            {"id": 2, "content": "[gMASK]", "special": true},
            {"id": 3, "content": "<|user|>", "special": true},
            {"id": 4, "content": "<|observation|>", "special": true},
            {"id": 5, "content": "<|endoftext|>", "special": true},
            {"id": 6, "content": "<|assistant|>", "special": true},
            {"id": 7, "content": "<|system|>", "special": true}
          ]
        }"#,
    )
    .unwrap();
    fs::write(
        root.join("tokenizer_config.json"),
        r#"{"eos_token": "<|assistant|>", "pad_token": "<|endoftext|>", "mask_token": "[gMASK]", "add_bos_token": false}"#,
    )
    .unwrap();
}

fn rewrite_config(root: &Path, replacements: &[(&str, &str)]) {
    let path = root.join("config.json");
    let mut config = fs::read_to_string(&path).unwrap();
    for (from, to) in replacements {
        assert!(config.contains(from), "config did not contain {from:?}");
        config = config.replace(from, to);
    }
    fs::write(path, config).unwrap();
}

fn write_llama_config_and_tokenizer(root: &Path) {
    fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "llama",
          "vocab_size": 4,
          "max_position_embeddings": 128,
          "hidden_size": 4,
          "intermediate_size": 8,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "rope_theta": 500000,
          "rms_norm_eps": 1e-5
        }"#,
    )
    .unwrap();
    fs::write(
        root.join("tokenizer.json"),
        r#"{
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "<|end_of_text|>": 2, "<|start_header_id|>": 3},
            "merges": ["a b"]
          },
          "decoder": {"type": "ByteLevel"},
          "added_tokens": [
            {"id": 2, "content": "<|end_of_text|>", "special": true},
            {"id": 3, "content": "<|start_header_id|>", "special": true}
          ]
        }"#,
    )
    .unwrap();
    fs::write(
        root.join("tokenizer_config.json"),
        r#"{"eos_token": "<|end_of_text|>", "add_bos_token": true}"#,
    )
    .unwrap();
}

fn write_dense_hf_safetensor(root: &Path) {
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "F32", &[1], &[1, 0, 0, 0]),
            (
                "model.layers.0.input_layernorm.weight",
                "F32",
                &[1],
                &[2, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "F32",
                &[1],
                &[3, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "F32",
                &[1],
                &[4, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "F32",
                &[1],
                &[5, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "F32",
                &[1],
                &[6, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.q_norm.weight",
                "F32",
                &[1],
                &[7, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.k_norm.weight",
                "F32",
                &[1],
                &[8, 0, 0, 0],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "F32",
                &[1],
                &[9, 0, 0, 0],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                &[1],
                &[10, 0, 0, 0],
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "F32",
                &[1],
                &[11, 0, 0, 0],
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "F32",
                &[1],
                &[12, 0, 0, 0],
            ),
            ("model.norm.weight", "F32", &[1], &[13, 0, 0, 0]),
            ("lm_head.weight", "F32", &[1], &[14, 0, 0, 0]),
        ],
    );
}

struct OwnedSafetensorTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn write_tiny_glm_dsa_safetensor(root: &Path) -> usize {
    let mut tensors = Vec::new();
    push_f32_tensor(&mut tensors, "model.embed_tokens.weight");
    push_f32_tensor(&mut tensors, "model.norm.weight");
    for layer in 0..3 {
        add_glm_dsa_attention_tensors(&mut tensors, layer);
    }
    add_glm_dsa_attention_tensors(&mut tensors, 3);
    add_glm_dsa_dense_ffn_tensors(&mut tensors, 0);
    for layer in [1, 2] {
        add_glm_dsa_moe_tensors(&mut tensors, layer);
    }
    add_glm_dsa_moe_tensors(&mut tensors, 3);
    add_glm_dsa_indexer_tensors(&mut tensors, 0);
    add_glm_dsa_indexer_tensors(&mut tensors, 2);
    add_glm_dsa_indexer_tensors(&mut tensors, 3);
    for suffix in ["eh_proj.weight", "enorm.weight", "hnorm.weight"] {
        push_f32_tensor(&mut tensors, format!("model.layers.3.{suffix}"));
    }
    let tensor_count = tensors.len();
    write_owned_safetensor(&root.join("model.safetensors"), &tensors);
    tensor_count
}

fn add_glm_dsa_attention_tensors(tensors: &mut Vec<OwnedSafetensorTensor>, layer: u32) {
    for suffix in [
        "input_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
    ] {
        push_layer_f32_tensor(tensors, layer, suffix);
    }
    push_bf16_tensor(
        tensors,
        format!("model.layers.{layer}.self_attn.kv_b_proj.weight"),
        &[5, 2],
    );
}

fn add_glm_dsa_dense_ffn_tensors(tensors: &mut Vec<OwnedSafetensorTensor>, layer: u32) {
    for suffix in [
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        "mlp.up_proj.weight",
    ] {
        push_layer_f32_tensor(tensors, layer, suffix);
    }
}

fn add_glm_dsa_moe_tensors(tensors: &mut Vec<OwnedSafetensorTensor>, layer: u32) {
    for suffix in [
        "mlp.gate.weight",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.down_proj.weight",
        "mlp.shared_experts.up_proj.weight",
    ] {
        push_layer_f32_tensor(tensors, layer, suffix);
    }
    for expert in 0..2 {
        for projection in ["gate_proj", "down_proj", "up_proj"] {
            push_layer_f32_tensor(
                tensors,
                layer,
                format!("mlp.experts.{expert}.{projection}.weight"),
            );
        }
    }
}

fn add_glm_dsa_indexer_tensors(tensors: &mut Vec<OwnedSafetensorTensor>, layer: u32) {
    for suffix in [
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.wq_b.weight",
    ] {
        push_layer_f32_tensor(tensors, layer, suffix);
    }
}

fn push_layer_f32_tensor(
    tensors: &mut Vec<OwnedSafetensorTensor>,
    layer: u32,
    suffix: impl AsRef<str>,
) {
    push_f32_tensor(tensors, format!("model.layers.{layer}.{}", suffix.as_ref()));
}

fn push_f32_tensor(tensors: &mut Vec<OwnedSafetensorTensor>, name: impl Into<String>) {
    tensors.push(OwnedSafetensorTensor {
        name: name.into(),
        dtype: "F32",
        shape: vec![1],
        bytes: vec![0, 0, 0x80, 0x3f],
    });
}

fn push_bf16_tensor(
    tensors: &mut Vec<OwnedSafetensorTensor>,
    name: impl Into<String>,
    shape: &[u64],
) {
    let elements = shape.iter().product::<u64>() as usize;
    tensors.push(OwnedSafetensorTensor {
        name: name.into(),
        dtype: "BF16",
        shape: shape.to_vec(),
        bytes: vec![0; elements * 2],
    });
}

fn write_owned_safetensor(path: &Path, tensors: &[OwnedSafetensorTensor]) {
    let mut offset = 0_u64;
    let mut entries = serde_json::Map::new();
    for tensor in tensors {
        let end = offset + tensor.bytes.len() as u64;
        entries.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let header = serde_json::Value::Object(entries).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for tensor in tensors {
        bytes.extend_from_slice(&tensor.bytes);
    }
    fs::write(path, bytes).unwrap();
}
