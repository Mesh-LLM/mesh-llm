#[test]
fn validates_qwen_dense_native_conversion_fixture() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_qwen_config_and_tokenizer(&root);
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
    let metadata = metadata_from_hf_config(&root, 14).unwrap();
    let validation = validate_raw_safetensors_gguf(
        &root,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(metadata.clone()),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    assert_eq!(validation.selected_tensor_count, 14);

    let output = root.join("qwen-native.gguf");
    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(metadata),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert!(parsed.metadata_count > 10);
    let attn_k = parsed.tensor("blk.0.attn_k.weight");
    assert_eq!(attn_k.ggml_type, GGML_TYPE_F32);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_glm_dsa_indexer_tensors_with_hf_name_mapping() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            (
                "model.layers.0.self_attn.indexer.k_norm.weight",
                "F32",
                &[1],
                &[1, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.k_norm.bias",
                "F32",
                &[1],
                &[2, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.weights_proj.weight",
                "F32",
                &[1],
                &[3, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.wk.weight",
                "F32",
                &[1],
                &[4, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.wq_b.weight",
                "F32",
                &[1],
                &[5, 0, 0, 0],
            ),
        ],
    );

    let output = root.join("glm-dsa-indexer.gguf");
    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: None,
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 5);
    parsed.tensor("blk.0.indexer.k_norm.weight");
    parsed.tensor("blk.0.indexer.k_norm.bias");
    parsed.tensor("blk.0.indexer.proj.weight");
    parsed.tensor("blk.0.indexer.attn_k.weight");
    parsed.tensor("blk.0.indexer.attn_q_b.weight");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn splits_glm_dsa_kv_b_projection_for_native_layout() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[(
            "model.layers.0.self_attn.kv_b_proj.weight",
            "F32",
            &[6, 2],
            &f32_bytes(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
        )],
    );
    let output = root.join("glm-dsa-kv-b.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 8,
            metadata: Some(glm_dsa_kv_b_split_metadata()),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: Some(ConvertOutputType::F32),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 2);
    let k_b = parsed.tensor("blk.0.attn_k_b.weight");
    let v_b = parsed.tensor("blk.0.attn_v_b.weight");
    assert_eq!(k_b.dims, vec![3, 2, 1]);
    assert_eq!(v_b.dims, vec![2, 3, 1]);
    assert_eq!(
        &bytes[k_b.absolute_offset..k_b.absolute_offset + 24],
        f32_bytes(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]).as_slice()
    );
    assert_eq!(
        &bytes[v_b.absolute_offset..v_b.absolute_offset + 24],
        f32_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).as_slice()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_inferred_glm_dsa_indexshare_types_to_gguf_metadata() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            (
                "model.layers.0.self_attn.indexer.k_norm.weight",
                "F32",
                &[1],
                &[1, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.k_norm.bias",
                "F32",
                &[1],
                &[2, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.weights_proj.weight",
                "F32",
                &[1],
                &[3, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.wk.weight",
                "F32",
                &[1],
                &[4, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.indexer.wq_b.weight",
                "F32",
                &[1],
                &[5, 0, 0, 0],
            ),
            (
                "model.layers.2.self_attn.indexer.k_norm.weight",
                "F32",
                &[1],
                &[6, 0, 0, 0],
            ),
            (
                "model.layers.2.self_attn.indexer.k_norm.bias",
                "F32",
                &[1],
                &[7, 0, 0, 0],
            ),
            (
                "model.layers.2.self_attn.indexer.weights_proj.weight",
                "F32",
                &[1],
                &[8, 0, 0, 0],
            ),
            (
                "model.layers.2.self_attn.indexer.wk.weight",
                "F32",
                &[1],
                &[9, 0, 0, 0],
            ),
            (
                "model.layers.2.self_attn.indexer.wq_b.weight",
                "F32",
                &[1],
                &[10, 0, 0, 0],
            ),
        ],
    );
    let output = root.join("glm-dsa-indexshare-types.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(minimal_glm_dsa_metadata(3, 0)),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let parsed = parse_test_gguf(&fs::read(&output).unwrap());

    assert_eq!(
        parsed
            .metadata_string_arrays
            .get("glm-dsa.attention.indexer.types"),
        Some(&vec![
            "full".to_string(),
            "shared".to_string(),
            "full".to_string(),
        ])
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn validates_qwen2_moe_native_conversion_fixture() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_qwen2_moe_config_and_tokenizer(&root);
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "F32", &[1], &[1, 0, 0, 0]),
            (
                "model.layers.0.mlp.shared_expert_gate",
                "F32",
                &[1],
                &[2, 0, 0, 0],
            ),
            (
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
                "F32",
                &[1],
                &[3, 0, 0, 0],
            ),
            (
                "model.layers.0.mlp.shared_expert.down_proj.weight",
                "F32",
                &[1],
                &[4, 0, 0, 0],
            ),
            (
                "model.layers.0.mlp.shared_expert.up_proj.weight",
                "F32",
                &[1],
                &[5, 0, 0, 0],
            ),
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "BF16",
                &[2],
                &[6, 7, 8, 9],
            ),
            (
                "model.layers.0.mlp.experts.1.gate_proj.weight",
                "BF16",
                &[2],
                &[10, 11, 12, 13],
            ),
        ],
    );
    let metadata = metadata_from_hf_config(&root, 7).unwrap();
    let validation = validate_raw_safetensors_gguf(
        &root,
        RawGgufWriteOptions {
            buffer_size: 3,
            metadata: Some(metadata.clone()),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    assert_eq!(validation.selected_tensor_count, 6);

    let output = root.join("qwen2-moe-native.gguf");
    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 3,
            metadata: Some(metadata),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);

    assert_eq!(
        parsed.tensor("blk.0.ffn_gate_inp_shexp.weight").ggml_type,
        GGML_TYPE_F32
    );
    assert_eq!(
        parsed.tensor("blk.0.ffn_gate_shexp.weight").ggml_type,
        GGML_TYPE_F32
    );
    let merged_experts = parsed.tensor("blk.0.ffn_gate_exps.weight");
    assert_eq!(merged_experts.dims, vec![2, 2]);
    assert_eq!(merged_experts.ggml_type, GGML_TYPE_BF16);
    assert_eq!(
        &bytes[merged_experts.absolute_offset..merged_experts.absolute_offset + 8],
        &[6, 7, 8, 9, 10, 11, 12, 13]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn validates_qwen3_moe_native_conversion_fixture() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_qwen3_moe_config_and_tokenizer(&root);
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
                "model.layers.0.self_attn.q_norm.weight",
                "F32",
                &[1],
                &[3, 0, 0, 0],
            ),
            (
                "model.layers.0.self_attn.k_norm.weight",
                "F32",
                &[1],
                &[4, 0, 0, 0],
            ),
            ("model.layers.0.mlp.gate.weight", "F32", &[1], &[5, 0, 0, 0]),
            (
                "model.layers.0.mlp.experts.0.down_proj.weight",
                "BF16",
                &[2],
                &[6, 7, 8, 9],
            ),
            (
                "model.layers.0.mlp.experts.1.down_proj.weight",
                "BF16",
                &[2],
                &[10, 11, 12, 13],
            ),
            ("model.norm.weight", "F32", &[1], &[14, 0, 0, 0]),
        ],
    );
    let metadata = metadata_from_hf_config(&root, 8).unwrap();
    let output = root.join("qwen3-moe-native.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 3,
            metadata: Some(metadata),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);

    assert_eq!(
        parsed.tensor("blk.0.attn_q_norm.weight").ggml_type,
        GGML_TYPE_F32
    );
    assert_eq!(
        parsed.tensor("blk.0.ffn_gate_inp.weight").ggml_type,
        GGML_TYPE_F32
    );
    let merged_experts = parsed.tensor("blk.0.ffn_down_exps.weight");
    assert_eq!(merged_experts.dims, vec![2, 2]);
    assert_eq!(merged_experts.ggml_type, GGML_TYPE_BF16);
    assert_eq!(
        &bytes[merged_experts.absolute_offset..merged_experts.absolute_offset + 8],
        &[6, 7, 8, 9, 10, 11, 12, 13]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn validates_glm_dsa_native_conversion_fixture() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_glm_dsa_config_and_tokenizer(&root);
    let tensor_count = write_tiny_glm_dsa_safetensor(&root);
    let metadata = metadata_from_hf_config(&root, tensor_count).unwrap();
    let validation = validate_raw_safetensors_gguf(
        &root,
        RawGgufWriteOptions {
            buffer_size: 8,
            metadata: Some(metadata.clone()),
            tensor_name_map: TensorNameMap::HfToGgufWithMtp { layer_start: 3 },
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    assert!(validation.selected_tensor_count > 0);

    let output = root.join("glm-dsa-native.gguf");
    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 8,
            metadata: Some(metadata),
            tensor_name_map: TensorNameMap::HfToGgufWithMtp { layer_start: 3 },
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    let parsed = parse_test_gguf(&fs::read(&output).unwrap());

    assert_eq!(
        parsed
            .metadata_string_arrays
            .get("glm-dsa.attention.indexer.types"),
        Some(&vec![
            "full".to_string(),
            "shared".to_string(),
            "full".to_string(),
        ])
    );
    assert_eq!(parsed.tensor("blk.0.attn_k_b.weight").dims, vec![3, 2, 1]);
    assert_eq!(parsed.tensor("blk.0.attn_v_b.weight").dims, vec![2, 2, 1]);
    parsed.tensor("blk.0.indexer.proj.weight");
    parsed.tensor("blk.2.indexer.proj.weight");
    parsed.tensor("blk.3.attn_norm.weight");
    parsed.tensor("blk.3.ffn_gate_inp.weight");
    parsed.tensor("blk.3.indexer.proj.weight");
    parsed.tensor("blk.3.nextn.eh_proj.weight");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_glm_dsa_indexer_type_frequency_conflict_from_config() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_glm_dsa_config_and_tokenizer(&root);
    rewrite_config(
        &root,
        &[("\"index_topk_freq\": 2", "\"index_topk_freq\": 1")],
    );

    let err = metadata_from_hf_config(&root, 1).unwrap_err();

    assert!(
        err.to_string()
            .contains("GLM-DSA indexer_types conflicts with index_topk_freq at layer 1"),
        "unexpected error: {err:#}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_glm_dsa_indexer_frequency_without_offset_from_config() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_glm_dsa_config_and_tokenizer(&root);
    rewrite_config(&root, &[("          \"index_skip_topk_offset\": 1,\n", "")]);

    let err = metadata_from_hf_config(&root, 1).unwrap_err();

    assert!(
        err.to_string().contains(
            "GLM-DSA index_skip_topk_offset/indexer_skip_top_k_offset is required when index_topk_freq is present"
        ),
        "unexpected error: {err:#}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn validates_llama_dense_native_conversion_fixture() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_llama_config_and_tokenizer(&root);
    write_dense_hf_safetensor(&root);
    let metadata = metadata_from_hf_config(&root, 14).unwrap();
    let validation = validate_raw_safetensors_gguf(
        &root,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(metadata.clone()),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    assert_eq!(validation.selected_tensor_count, 14);

    let output = root.join("llama-native.gguf");
    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(metadata),
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();
    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert!(parsed.metadata_count > 10);
    assert_eq!(
        parsed.tensor("blk.0.attn_q.weight").ggml_type,
        GGML_TYPE_F32
    );
    fs::remove_dir_all(root).unwrap();
}
