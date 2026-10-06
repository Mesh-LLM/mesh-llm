// Synthetic native SafeTensors→GGUF tests, no Python oracle/native loader.
#[test]
fn writes_nemotron_mtp_folded_nextn_and_streams_experts_in_id_order() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    let first = 1_f32.to_le_bytes();
    let second = 2_f32.to_le_bytes();
    let norm = 3_f32.to_le_bytes();
    write_safetensor(
        &root.as_path().join("head.safetensors"),
        &[
            ("mtp.layers.0.enorm.weight", "F32", &[1], &norm),
            ("mtp.layers.0.eh_proj.weight", "F32", &[1, 1], &norm),
            ("mtp.layers.0.mixer.q_proj.weight", "F32", &[1, 1], &norm),
            ("mtp.layers.1.norm.weight", "F32", &[1], &norm),
            ("mtp.layers.1.final_layernorm.weight", "F32", &[1], &norm),
            (
                "mtp.layers.1.mixer.experts.1.up_proj.weight",
                "F32",
                &[1, 1],
                &second,
            ),
            (
                "mtp.layers.1.mixer.experts.0.up_proj.weight",
                "F32",
                &[1, 1],
                &first,
            ),
            (
                "mtp.layers.1.mixer.experts.1.down_proj.weight",
                "F32",
                &[1, 1],
                &second,
            ),
            (
                "mtp.layers.1.mixer.experts.0.down_proj.weight",
                "F32",
                &[1, 1],
                &first,
            ),
            ("backbone.embeddings.weight", "F32", &[1, 1], &norm),
        ],
    );
    let output = root.as_path().join("mtp.gguf");
    write_raw_safetensors_gguf(
        root.as_path(),
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(vec![
                GgufKv::string("general.architecture", "nemotron_h_moe"),
                GgufKv::u32("nemotron_h_moe.block_count", 89),
                GgufKv::u32("nemotron_h_moe.nextn_predict_layers", 1),
            ]),
            tensor_name_map: TensorNameMap::NemotronHMoeMtp { layer_start: 88 },
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::MtpOnly { layer_start: 88 },
        },
    )
    .unwrap();
    let bytes = fs::read(&output).unwrap();
    let parsed = parse_test_gguf(&bytes);
    let names = parsed
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "blk.88.attn_q.weight",
            "blk.88.ffn_down_exps.weight",
            "blk.88.ffn_up_exps.weight",
            "blk.88.nextn.eh_proj.weight",
            "blk.88.nextn.enorm.weight",
            "blk.88.nextn.shared_head_norm.weight",
            "blk.88.post_attention_norm.weight",
            "token_embd.weight"
        ]
    );
    assert!(!names.iter().any(|n| n.starts_with("blk.89.")));
    for name in ["blk.88.ffn_up_exps.weight", "blk.88.ffn_down_exps.weight"] {
        let tensor = parsed.tensors.iter().find(|t| t.name == name).unwrap();
        assert_eq!(tensor.dims, [1, 1, 2]);
        assert_eq!(
            &bytes[tensor.absolute_offset..tensor.absolute_offset + 8],
            [first, second].concat()
        );
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn nemotron_native_writer_refuses_unknown_fold_depth_and_noncontiguous_expert_roster() {
    for name in [
        "mtp.layers.2.enorm.weight",
        "mtp.layers.1.mixer.experts.1.up_proj.weight",
    ] {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let scalar = 1_f32.to_le_bytes();
        write_safetensor(
            &root.as_path().join("head.safetensors"),
            &[(name, "F32", &[1, 1], &scalar)],
        );
        let output = root.as_path().join("never.gguf");
        assert!(
            write_raw_safetensors_gguf(
                root.as_path(),
                &output,
                RawGgufWriteOptions {
                    buffer_size: 4,
                    metadata: Some(vec![GgufKv::string(
                        "general.architecture",
                        "nemotron_h_moe"
                    )]),
                    tensor_name_map: TensorNameMap::NemotronHMoeMtp { layer_start: 88 },
                    split: None,
                    output_type: None,
                    tensor_selection: TensorSelection::MtpOnly { layer_start: 88 },
                }
            )
            .is_err()
        );
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn nemotron_profile_writer_refuses_incomplete_nextn_before_output_creation() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    let bytes = 1_f32.to_le_bytes();
    write_safetensor(
        &root.join("head.safetensors"),
        &[("mtp.layers.0.enorm.weight", "F32", &[1], &bytes)],
    );
    let output = root.join("refused.gguf");
    let error = write_raw_safetensors_gguf(
        &root,
        &output,
        RawGgufWriteOptions {
            buffer_size: 4,
            metadata: Some(vec![
                GgufKv::u32("nemotron_h_moe.expert_count", 2),
                GgufKv::string(
                    "skippy.convert.tokenizer_profile_sha256",
                    "synthetic-ownership-marker",
                ),
            ]),
            tensor_name_map: TensorNameMap::NemotronHMoeMtp { layer_start: 3 },
            split: None,
            output_type: None,
            tensor_selection: TensorSelection::MtpOnly { layer_start: 3 },
        },
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("required NextN projection absent")
    );
    assert!(!output.exists());
    fs::remove_dir_all(root).unwrap();
}
