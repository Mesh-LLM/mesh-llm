use std::io::Read;
use std::path::PathBuf;

use crate::gguf_template::metadata_from_hf_config;
use crate::tensor_map::TensorNameMap;

use super::*;

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

struct ParsedGguf {
    tensor_count: u64,
    metadata_count: u64,
    data_start: usize,
    tensors: Vec<ParsedTensor>,
}

impl ParsedGguf {
    fn tensor(&self, name: &str) -> &ParsedTensor {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }
}

struct ParsedTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    absolute_offset: usize,
}

fn parse_test_gguf(bytes: &[u8]) -> ParsedGguf {
    let mut cursor = std::io::Cursor::new(bytes);
    let mut magic = [0_u8; 4];
    cursor.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, GGUF_MAGIC);
    assert_eq!(read_u32(&mut cursor), GGUF_VERSION);
    let tensor_count = read_u64(&mut cursor);
    let metadata_count = read_u64(&mut cursor);
    for _ in 0..metadata_count {
        let _key = read_string(&mut cursor);
        let value_type = read_u32(&mut cursor);
        match value_type {
            GGUF_TYPE_BOOL => {
                let mut value = [0_u8; 1];
                cursor.read_exact(&mut value).unwrap();
            }
            GGUF_TYPE_UINT16 => {
                let mut value = [0_u8; 2];
                cursor.read_exact(&mut value).unwrap();
            }
            GGUF_TYPE_INT32 => {
                let _ = read_u32(&mut cursor);
            }
            GGUF_TYPE_STRING => {
                let _ = read_string(&mut cursor);
            }
            GGUF_TYPE_UINT32 => {
                let _ = read_u32(&mut cursor);
            }
            GGUF_TYPE_FLOAT32 => {
                let _ = read_u32(&mut cursor);
            }
            GGUF_TYPE_UINT64 => {
                let _ = read_u64(&mut cursor);
            }
            GGUF_TYPE_ARRAY => skip_array(&mut cursor),
            other => panic!("unexpected metadata type {other}"),
        }
    }
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut cursor);
        let dim_count = read_u32(&mut cursor);
        let dims = (0..dim_count)
            .map(|_| read_u64(&mut cursor))
            .collect::<Vec<_>>();
        let ggml_type = read_u32(&mut cursor);
        let relative_offset = read_u64(&mut cursor);
        tensors.push((name, dims, ggml_type, relative_offset));
    }
    let data_start = align_to(cursor.position(), GGUF_ALIGNMENT) as usize;
    ParsedGguf {
        tensor_count,
        metadata_count,
        data_start,
        tensors: tensors
            .into_iter()
            .map(|(name, dims, ggml_type, relative_offset)| ParsedTensor {
                name,
                dims,
                ggml_type,
                absolute_offset: data_start + relative_offset as usize,
            })
            .collect(),
    }
}

fn read_string(cursor: &mut std::io::Cursor<&[u8]>) -> String {
    let len = read_u64(cursor);
    let mut bytes = vec![0_u8; len as usize];
    cursor.read_exact(&mut bytes).unwrap();
    String::from_utf8(bytes).unwrap()
}

fn read_u32(cursor: &mut std::io::Cursor<&[u8]>) -> u32 {
    let mut bytes = [0_u8; 4];
    cursor.read_exact(&mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}

fn read_u64(cursor: &mut std::io::Cursor<&[u8]>) -> u64 {
    let mut bytes = [0_u8; 8];
    cursor.read_exact(&mut bytes).unwrap();
    u64::from_le_bytes(bytes)
}

fn skip_array(cursor: &mut std::io::Cursor<&[u8]>) {
    let element_type = read_u32(cursor);
    let len = read_u64(cursor);
    for _ in 0..len {
        match element_type {
            GGUF_TYPE_STRING => {
                let _ = read_string(cursor);
            }
            GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 | GGUF_TYPE_UINT32 => {
                let _ = read_u32(cursor);
            }
            other => panic!("unexpected test array element type {other}"),
        }
    }
}

fn unique_temp_dir() -> PathBuf {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("skippy-gguf-writer-{nanos}-{id}"))
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
