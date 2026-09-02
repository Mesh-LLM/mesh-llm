#[test]
fn streams_expert_tensors_as_merged_gguf_tensor() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            (
                "model.layers.1.mlp.experts.1.gate_proj.weight",
                "BF16",
                &[2, 2],
                &[5, 6, 7, 8, 9, 10, 11, 12],
            ),
            (
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "BF16",
                &[2, 2],
                &[1, 2, 3, 4, 13, 14, 15, 16],
            ),
        ],
    );
    let output = root.join("experts.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 3,
            metadata: None,
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 1);
    assert_eq!(parsed.tensors[0].name, "blk.1.ffn_gate_exps.weight");
    assert_eq!(parsed.tensors[0].dims, vec![2, 2, 2]);
    assert_eq!(
        &bytes[parsed.tensors[0].absolute_offset..parsed.tensors[0].absolute_offset + 16],
        &[1, 2, 3, 4, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_only_selected_split_with_split_metadata() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("a.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("b.weight", "F32", &[1], &[2, 0, 0, 0]),
            ("c.weight", "F32", &[1], &[3, 0, 0, 0]),
            ("d.weight", "F32", &[1], &[4, 0, 0, 0]),
        ],
    );
    let output = root.join("split.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 2,
            metadata: Some(vec![GgufKv::array_string(
                "tokenizer.ggml.tokens",
                vec!["a".to_string()],
            )]),
            tensor_name_map: TensorNameMap::Raw,
            split: Some(GgufSplit {
                split_index: 2,
                split_count: 2,
            }),
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 2);
    assert_eq!(parsed.metadata_count, 4);
    assert_eq!(parsed.tensors[0].name, "c.weight");
    assert_eq!(parsed.tensors[1].name, "d.weight");
    assert_eq!(parsed.tensors[0].absolute_offset, parsed.data_start);
    assert_eq!(
        &bytes[parsed.tensors[0].absolute_offset..parsed.tensors[0].absolute_offset + 4],
        &[3, 0, 0, 0]
    );
    assert_eq!(
        &bytes[parsed.tensors[1].absolute_offset..parsed.tensors[1].absolute_offset + 4],
        &[4, 0, 0, 0]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn infers_glm_dsa_indexshare_types_before_split_selection() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("blk.0.indexer.attn_k.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("blk.0.indexer.attn_q_b.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("blk.0.indexer.k_norm.bias", "F32", &[1], &[1, 0, 0, 0]),
            ("blk.0.indexer.k_norm.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("blk.0.indexer.proj.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("blk.2.indexer.attn_k.weight", "F32", &[1], &[2, 0, 0, 0]),
            ("blk.2.indexer.attn_q_b.weight", "F32", &[1], &[2, 0, 0, 0]),
            ("blk.2.indexer.k_norm.bias", "F32", &[1], &[2, 0, 0, 0]),
            ("blk.2.indexer.k_norm.weight", "F32", &[1], &[2, 0, 0, 0]),
            ("blk.2.indexer.proj.weight", "F32", &[1], &[2, 0, 0, 0]),
        ],
    );
    let output = root.join("split.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(minimal_glm_dsa_metadata(3, 0)),
            tensor_name_map: TensorNameMap::Raw,
            split: Some(GgufSplit {
                split_index: 2,
                split_count: 2,
            }),
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
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
fn native_splits_are_byte_balanced_not_tensor_count_balanced() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("a.weight", "F32", &[64], &[1; 256]),
            ("b.weight", "F32", &[1], &[2, 0, 0, 0]),
            ("c.weight", "F32", &[1], &[3, 0, 0, 0]),
            ("d.weight", "F32", &[1], &[4, 0, 0, 0]),
        ],
    );
    let output = root.join("split.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 64,
            metadata: None,
            tensor_name_map: TensorNameMap::Raw,
            split: Some(GgufSplit {
                split_index: 1,
                split_count: 2,
            }),
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 1);
    assert_eq!(parsed.tensors[0].name, "a.weight");
    assert_eq!(parsed.tensors[0].absolute_offset, parsed.data_start);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn recommends_enough_byte_balanced_splits_for_the_size_limit() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("a.weight", "F32", &[4], &[1; 16]),
            ("b.weight", "F32", &[4], &[2; 16]),
            ("c.weight", "F32", &[4], &[3; 16]),
        ],
    );

    let split_count = recommended_raw_safetensors_gguf_split_count(
        &root,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: None,
            tensor_name_map: TensorNameMap::Raw,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
        20,
    )
    .unwrap();

    assert_eq!(split_count, 3);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_a_size_limit_smaller_than_one_selected_tensor() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[("a.weight", "F32", &[4], &[1; 16])],
    );

    let error = recommended_raw_safetensors_gguf_split_count(
        &root,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: None,
            tensor_name_map: TensorNameMap::Raw,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
        15,
    )
    .unwrap_err();

    assert!(error.to_string().contains("largest selected tensor"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn keeps_rank_one_f32_tensor_as_f32_for_bf16_output() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[("a.weight", "F32", &[2], &[0, 0, 0x80, 0x3f, 0, 0, 0, 0x40])],
    );
    let output = root.join("bf16.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: None,
            tensor_name_map: TensorNameMap::Raw,
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensors[0].ggml_type, GGML_TYPE_F32);
    assert_eq!(
        &bytes[parsed.tensors[0].absolute_offset..parsed.tensors[0].absolute_offset + 8],
        &[0, 0, 0x80, 0x3f, 0, 0, 0, 0x40]
    );
    fs::remove_dir_all(root).unwrap();
}
