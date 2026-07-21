use std::collections::BTreeMap;
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
fn infers_glm_dsa_indexshare_types_from_mapped_tensor_names() {
    let mut metadata = minimal_glm_dsa_metadata(3, 0);
    let tensors = vec![
        mock_tensor_source("blk.0.indexer.k_norm.weight"),
        mock_tensor_source("blk.0.indexer.k_norm.bias"),
        mock_tensor_source("blk.0.indexer.proj.weight"),
        mock_tensor_source("blk.0.indexer.attn_k.weight"),
        mock_tensor_source("blk.0.indexer.attn_q_b.weight"),
        mock_tensor_source("blk.2.indexer.k_norm.weight"),
        mock_tensor_source("blk.2.indexer.k_norm.bias"),
        mock_tensor_source("blk.2.indexer.proj.weight"),
        mock_tensor_source("blk.2.indexer.attn_k.weight"),
        mock_tensor_source("blk.2.indexer.attn_q_b.weight"),
    ];

    enrich_glm_dsa_indexshare_metadata(&mut metadata, &tensors).unwrap();

    assert_eq!(
        array_string_metadata(&metadata, "glm-dsa.attention.indexer.types"),
        Some(vec![
            "full".to_string(),
            "shared".to_string(),
            "full".to_string(),
        ])
    );
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
fn rejects_glm_dsa_partial_indexshare_group_during_metadata_enrichment() {
    let mut metadata = minimal_glm_dsa_metadata(2, 0);
    let tensors = vec![mock_tensor_source("blk.0.indexer.k_norm.weight")];

    let err = enrich_glm_dsa_indexshare_metadata(&mut metadata, &tensors).unwrap_err();

    assert!(
        err.to_string().contains("partial indexer tensor group"),
        "unexpected error: {err:#}"
    );
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
    metadata_string_arrays: BTreeMap<String, Vec<String>>,
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
    let mut metadata_string_arrays = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut cursor);
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
            GGUF_TYPE_ARRAY => {
                if let Some(value) = read_string_array_or_skip(&mut cursor) {
                    metadata_string_arrays.insert(key, value);
                }
            }
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
        metadata_string_arrays,
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

fn read_string_array_or_skip(cursor: &mut std::io::Cursor<&[u8]>) -> Option<Vec<String>> {
    let element_type = read_u32(cursor);
    let len = read_u64(cursor);
    if element_type == GGUF_TYPE_STRING {
        return Some((0..len).map(|_| read_string(cursor)).collect());
    }
    skip_array_items(cursor, element_type, len);
    None
}

fn skip_array_items(cursor: &mut std::io::Cursor<&[u8]>, element_type: u32, len: u64) {
    for _ in 0..len {
        match element_type {
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

fn mock_tensor_source(name: &str) -> TensorSource {
    TensorSource {
        segments: Vec::new(),
        name: name.to_string(),
        dims: vec![1],
        ggml_type: GGML_TYPE_F32,
        byte_len: 4,
        gguf_offset: 0,
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn array_string_metadata(metadata: &[GgufKv], key: &str) -> Option<Vec<String>> {
    metadata.iter().find_map(|kv| match kv {
        GgufKv::ArrayString {
            key: item_key,
            value,
        } if item_key == key => Some(value.clone()),
        _ => None,
    })
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
