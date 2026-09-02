#[test]
fn writes_raw_gguf_from_safetensors_with_streamed_payloads() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("b.weight", "BF16", &[2], &[9, 8, 7, 6]),
            ("a.weight", "F32", &[1], &[1, 2, 3, 4]),
        ],
    );
    let output = root.join("raw.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 2,
            metadata: None,
            tensor_name_map: TensorNameMap::Raw,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::All,
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[..4], GGUF_MAGIC);
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 2);
    assert_eq!(parsed.metadata_count, 4);
    assert_eq!(parsed.tensors[0].name, "a.weight");
    assert_eq!(parsed.tensors[0].ggml_type, GGML_TYPE_F32);
    assert_eq!(
        &bytes[parsed.tensors[0].absolute_offset..parsed.tensors[0].absolute_offset + 4],
        &[1, 2, 3, 4]
    );
    assert_eq!(parsed.tensors[1].name, "b.weight");
    assert_eq!(parsed.tensors[1].ggml_type, GGML_TYPE_BF16);
    assert_eq!(
        &bytes[parsed.tensors[1].absolute_offset..parsed.tensors[1].absolute_offset + 4],
        &[9, 8, 7, 6]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_mapped_hf_tensor_names_when_requested() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[(
            "model.layers.0.input_layernorm.weight",
            "F32",
            &[1],
            &[1, 2, 3, 4],
        )],
    );
    let output = root.join("mapped.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 2,
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
    assert_eq!(parsed.tensors[0].name, "blk.0.attn_norm.weight");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn excludes_mtp_source_tensors_before_hf_name_mapping() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            (
                "model.layers.0.input_layernorm.weight",
                "F32",
                &[1],
                &[1, 2, 3, 4],
            ),
            (
                "model.layers.1.input_layernorm.weight",
                "F32",
                &[1],
                &[5, 6, 7, 8],
            ),
            (
                "model.layers.1.eh_proj.weight",
                "F32",
                &[1],
                &[9, 10, 11, 12],
            ),
            ("mtp.fc.weight", "F32", &[1], &[13, 14, 15, 16]),
        ],
    );
    let output = root.join("no-mtp.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 2,
            metadata: None,
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::ExcludeMtp { layer_start: 1 },
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 1);
    assert_eq!(parsed.tensors[0].name, "blk.0.attn_norm.weight");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_mtp_only_tensors_with_shared_context() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("lm_head.weight", "F32", &[1], &[2, 0, 0, 0]),
            (
                "model.layers.0.input_layernorm.weight",
                "F32",
                &[1],
                &[3, 0, 0, 0],
            ),
            (
                "model.layers.1.input_layernorm.weight",
                "F32",
                &[1],
                &[4, 0, 0, 0],
            ),
            ("model.layers.1.eh_proj.weight", "F32", &[1], &[5, 0, 0, 0]),
        ],
    );
    let output = root.join("mtp-only.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: None,
            tensor_name_map: TensorNameMap::HfToGguf,
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::MtpOnly { layer_start: 1 },
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    let names = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "blk.1.attn_norm.weight",
            "blk.1.nextn.eh_proj.weight",
            "output.weight",
            "token_embd.weight",
        ]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_qwen_style_mtp_only_tensors_with_shared_context() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("embed_tokens.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("norm.weight", "F32", &[1], &[2, 0, 0, 0]),
            (
                "model.layers.0.input_layernorm.weight",
                "F32",
                &[1],
                &[3, 0, 0, 0],
            ),
            ("mtp.fc.weight", "F32", &[1], &[4, 0, 0, 0]),
            ("model.mtp.norm.weight", "F32", &[1], &[5, 0, 0, 0]),
            (
                "mtp.layers.1.self_attn.q_proj.weight",
                "F32",
                &[1],
                &[6, 0, 0, 0],
            ),
        ],
    );
    let output = root.join("qwen-mtp-only.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: None,
            tensor_name_map: TensorNameMap::HfToGgufWithMtp { layer_start: 32 },
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::MtpOnly { layer_start: 32 },
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    let names = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "blk.32.nextn.eh_proj.weight",
            "blk.32.nextn.shared_head_norm.weight",
            "blk.33.attn_q.weight",
            "output_norm.weight",
            "token_embd.weight",
        ]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_inkling_mtp_streaming_transforms() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    let w13 = (1_u32..=8)
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let bf16_values = [0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40];
    write_safetensor(
        &root.join("model.safetensors"),
        &[
            ("model.llm.embed.weight", "F32", &[1], &[1, 0, 0, 0]),
            ("model.llm.embed_norm.weight", "F32", &[1], &[2, 0, 0, 0]),
            ("model.llm.norm.weight", "F32", &[1], &[3, 0, 0, 0]),
            ("model.llm.unembed.weight", "F32", &[1], &[4, 0, 0, 0]),
            (
                "model.mtp.layers.0.embed_norm.weight",
                "F32",
                &[1],
                &[5, 0, 0, 0],
            ),
            (
                "model.mtp.layers.0.transformer_block.attn.rel_logits_proj.proj",
                "BF16",
                &[2, 2],
                &bf16_values,
            ),
            (
                "model.mtp.layers.0.transformer_block.attn.k_sconv.weight",
                "BF16",
                &[2, 1, 2],
                &bf16_values,
            ),
            (
                "model.mtp.layers.0.transformer_block.mlp.w13_dn.weight",
                "F32",
                &[4, 2],
                &w13,
            ),
        ],
    );
    let output = root.join("inkling-mtp.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 9,
            metadata: None,
            tensor_name_map: TensorNameMap::HfToGgufWithMtp { layer_start: 66 },
            split: None,
            output_type: Some(ConvertOutputType::Bf16),
            tensor_selection: TensorSelection::MtpOnly { layer_start: 66 },
        },
    )
    .unwrap();

    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    assert_eq!(parsed.tensor_count, 9);
    let shortconv = parsed.tensor("blk.66.shortconv_k.weight");
    assert_eq!(shortconv.dims, vec![2, 2]);
    assert_eq!(shortconv.ggml_type, GGML_TYPE_F32);
    let rel_proj = parsed.tensor("blk.66.attn_rel_proj.weight");
    assert_eq!(rel_proj.ggml_type, GGML_TYPE_F32);
    let gate = parsed.tensor("blk.66.ffn_gate.weight");
    let up = parsed.tensor("blk.66.ffn_up.weight");
    assert_eq!(gate.dims, vec![2, 2]);
    assert_eq!(up.dims, vec![2, 2]);
    assert_eq!(gate.ggml_type, GGML_TYPE_BF16);
    assert_eq!(up.ggml_type, GGML_TYPE_BF16);
    let gate_expected = [0x80, 0x3f, 0x00, 0x40, 0xa0, 0x40, 0xc0, 0x40];
    let up_expected = [0x40, 0x40, 0x80, 0x40, 0xe0, 0x40, 0x00, 0x41];
    assert_eq!(
        &bytes[gate.absolute_offset..gate.absolute_offset + gate_expected.len()],
        gate_expected
    );
    assert_eq!(
        &bytes[up.absolute_offset..up.absolute_offset + up_expected.len()],
        up_expected
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writes_inkling_trunk_fused_w13_streaming_transforms() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    let w13 = (1_u32..=8)
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    write_safetensor(
        &root.join("model.safetensors"),
        &[("model.layers.3.mlp.w13_dn.weight", "F32", &[4, 2], &w13)],
    );
    let output = root.join("inkling-trunk.gguf");

    write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 9,
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
    assert_eq!(parsed.tensor_count, 2);
    let gate = parsed.tensor("blk.3.ffn_gate.weight");
    let up = parsed.tensor("blk.3.ffn_up.weight");
    assert_eq!(gate.dims, vec![2, 2]);
    assert_eq!(up.dims, vec![2, 2]);
    assert_eq!(gate.ggml_type, GGML_TYPE_BF16);
    assert_eq!(up.ggml_type, GGML_TYPE_BF16);
    let gate_expected = [0x80, 0x3f, 0x00, 0x40, 0xa0, 0x40, 0xc0, 0x40];
    let up_expected = [0x40, 0x40, 0x80, 0x40, 0xe0, 0x40, 0x00, 0x41];
    assert_eq!(
        &bytes[gate.absolute_offset..gate.absolute_offset + gate_expected.len()],
        gate_expected
    );
    assert_eq!(
        &bytes[up.absolute_offset..up.absolute_offset + up_expected.len()],
        up_expected
    );
    fs::remove_dir_all(root).unwrap();
}
