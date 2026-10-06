use super::*;
#[test]
fn fair_optional_capacity_uses_total_token_budget_and_pinned_native_override_without_gguf_flags() {
    let root = tempfile::tempdir().unwrap();
    let model_file = root.path().join("fixture.gguf");
    std::fs::write(&model_file, b"model").unwrap();
    let model_hash = crate::product::digest::file_sha256(&model_file)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    let binary = std::env::current_exe().unwrap();
    let binary_hash = crate::product::digest::file_sha256(&binary)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    let tokenizer = root.path().join("tokenizer");
    std::fs::create_dir(&tokenizer).unwrap();
    std::fs::write(tokenizer.join("tokenizer.json"), b"{}").unwrap();
    let config_target = tokenizer.join("config.json");
    std::fs::write(&config_target, b"{\"model_type\":\"fixture\"}").unwrap();
    let config_hash = crate::product::digest::file_sha256(&config_target)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    let config_directory = root.path().join("vllm-config");
    std::fs::create_dir(&config_directory).unwrap();
    let config_path = config_directory.join("config.json");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&config_target, &config_path).unwrap();
    #[cfg(not(unix))]
    std::fs::copy(&config_target, &config_path).unwrap();
    let hash = crate::product::digest::tree_sha256(&tokenizer)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));

    let model = json!({"key":"fixture","model_id":"fixture:model","sha256":model_hash,"tokenizer_sha256":hash,"vllm_hf_config":{"sha256":config_hash},"comparison_inputs":{"vllm":{"sha256":hash,"tensor_equivalence_sha256":"a".repeat(64)}},"vllm_capacity":{"reference_tokens":131072,"reference_blocks":2560}});
    let backend:Backend=serde_json::from_value(json!({"executable":{"path":binary,"sha256":binary_hash},"version_sha256":"b".repeat(64),"cwd":root.path(),"runtime":null,"tokenizer":{"path":tokenizer,"sha256":hash},"hf_config":{"path":config_path,"sha256":config_hash},"comparison_model":{"path":tokenizer,"sha256":hash},"match_kv_capacity":true})).unwrap();
    let artifact = Artifact {
        path: model_file,
        sha256: model_hash,
    };
    let cell = json!({"arm":"vllm"});
    let mut provenance = json!({});
    let selection = Selection {
        model: &model,
        cell: &cell,
        backend: &backend,
        artifact: &artifact,
        port: 1234,
        directory: root.path(),
        capacity: Capacity {
            context: 65536,
            lanes: 16,
        },
        cache: true,
    };
    let command = external(&selection, "vllm", "fixture:model", &mut provenance).unwrap();
    assert!(
        command
            .windows(2)
            .any(|pair| pair == ["--max-model-len", "65536"])
    );
    assert!(
        command
            .windows(2)
            .any(|pair| pair == ["--max-num-seqs", "16"])
    );
    assert!(
        command
            .windows(2)
            .any(|pair| pair == ["--num-gpu-blocks-override", "1280"])
    );
    assert!(
        command
            .windows(2)
            .any(|pair| pair[0] == "--hf-config-path"
                && pair[1] == config_directory.to_str().unwrap())
    );
    assert_ne!(hash, config_hash);
    std::fs::write(config_path.canonicalize().unwrap(), b"changed").unwrap();
    assert!(hf_config_directory(backend.hf_config.as_ref().unwrap()).is_err());
    assert!(!command.contains(&"--load-format".into()));
    assert!(!command.contains(&"--quantization".into()));
    assert_eq!(provenance["comparison_input_sha256"], hash);
    assert_eq!(provenance["tensor_equivalence_sha256"], "a".repeat(64));
    assert_eq!(
        capacity_args(&json!({}), "sglang", 131072, true).unwrap(),
        ["--max-total-tokens", "131072"]
    );
    root.close().unwrap();
}
#[test]
fn competitive_artifact_admission_rejects_byte_drift_and_stage_keeps_shared_context_policy() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("fixture.gguf");
    std::fs::write(&path, b"model").unwrap();
    let artifact = Artifact {
        sha256: crate::product::digest::file_sha256(&path)
            .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error)),
        path,
    };
    assert!(file(&artifact).is_ok());
    std::fs::write(&artifact.path, b"changed").unwrap();
    assert!(file(&artifact).is_err());
    let stage = stage(
        &json!({"key":"fixture","model_id":"fixture","layer_end":40,"cache_payload":"kv-recurrent"}),
        &artifact,
        1234,
        Capacity {
            context: 131072,
            lanes: 16,
        },
        true,
    );
    assert_eq!(stage["ctx_size"], 131072);
    assert_eq!(stage["lane_count"], 16);
    assert_eq!(stage["kv_cache"]["max_entries"], 512);
    assert_eq!(stage["kv_cache"]["shared_prefix_record_limit"], 2);
    assert_eq!(stage["native_mtp_enabled"], false);
    assert!(stage["downstream"].is_null());
    assert_eq!(
        capacity_args(&json!({}), "vllm", 17, true)
            .unwrap()
            .last()
            .unwrap(),
        "2"
    );
    assert!(
        capacity_args(
            &json!({"vllm_capacity":{"reference_tokens":0,"reference_blocks":1}}),
            "vllm",
            17,
            true
        )
        .is_err()
    );
    root.close().unwrap();
}
fn optional_fixture(root: &Path) -> (Value, Backend, Artifact) {
    let gguf = root.join("fixture.gguf");
    std::fs::write(&gguf, b"GGUF fixture bytes").unwrap();
    let model_hash = crate::product::digest::file_sha256(&gguf)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    let tokenizer = root.join("tokenizer");
    std::fs::create_dir(&tokenizer).unwrap();
    std::fs::write(tokenizer.join("tokenizer.json"), b"{}").unwrap();
    let pin = crate::product::digest::tree_sha256(&tokenizer)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    let binary = std::env::current_exe().unwrap();
    let binary_hash = crate::product::digest::file_sha256(&binary)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    let backend:Backend=serde_json::from_value(json!({"executable":{"path":binary,"sha256":binary_hash},"version_sha256":"b".repeat(64),"cwd":root,"runtime":null,"tokenizer":{"path":tokenizer,"sha256":pin},"hf_config":null,"comparison_model":null,"match_kv_capacity":false})).unwrap();
    (
        json!({"sha256":model_hash,"tokenizer_sha256":pin}),
        backend,
        Artifact {
            path: gguf,
            sha256: model_hash,
        },
    )
}
#[test]
fn sglang_uses_pinned_gguf_tokenizer_capacity_cache_flags_and_refuses_tokenizer_drift() {
    let root = tempfile::tempdir().unwrap();
    let (model, backend, artifact) = optional_fixture(root.path());
    let cell = json!({"arm":"sglang"});
    for cache in [true, false] {
        let selected = Selection {
            model: &model,
            cell: &cell,
            backend: &backend,
            artifact: &artifact,
            port: 1234,
            directory: root.path(),
            capacity: Capacity {
                context: 16384,
                lanes: 4,
            },
            cache,
        };
        let command = external(&selected, "sglang", "fixture-model", &mut json!({})).unwrap();
        for (flag, value) in [
            ("--model-path", artifact.path.to_str().unwrap()),
            (
                "--tokenizer-path",
                backend.tokenizer.as_ref().unwrap().path.to_str().unwrap(),
            ),
            ("--max-running-requests", "4"),
            ("--load-format", "gguf"),
            ("--quantization", "gguf"),
            ("--served-model-name", "fixture-model"),
        ] {
            assert!(
                command.windows(2).any(|pair| pair == [flag, value]),
                "{command:?}"
            );
        }
        assert_eq!(
            command.iter().any(|arg| arg == "--disable-radix-cache"),
            !cache
        );
    }
    std::fs::write(
        backend
            .tokenizer
            .as_ref()
            .unwrap()
            .path
            .join("tokenizer.json"),
        b"changed",
    )
    .unwrap();
    let selected = Selection {
        model: &model,
        cell: &cell,
        backend: &backend,
        artifact: &artifact,
        port: 1234,
        directory: root.path(),
        capacity: Capacity {
            context: 16384,
            lanes: 4,
        },
        cache: true,
    };
    assert!(external(&selected, "sglang", "fixture-model", &mut json!({})).is_err());
    root.close().unwrap();
}
#[test]
fn alternate_container_refuses_missing_source_pin_and_actual_tree_byte_drift() {
    let root = tempfile::tempdir().unwrap();
    let (mut model, mut backend, artifact) = optional_fixture(root.path());
    let alternate = root.path().join("alternate");
    std::fs::create_dir(&alternate).unwrap();
    std::fs::write(alternate.join("config.json"), b"{}").unwrap();
    let hash = crate::product::digest::tree_sha256(&alternate)
        .unwrap_or_else(|failure| panic!("fixture digest: {}", failure.error));
    backend.comparison_model = Some(Artifact {
        path: alternate.clone(),
        sha256: hash.clone(),
    });
    let cell = json!({"arm":"sglang"});
    {
        let selected = Selection {
            model: &model,
            cell: &cell,
            backend: &backend,
            artifact: &artifact,
            port: 1234,
            directory: root.path(),
            capacity: Capacity {
                context: 16384,
                lanes: 4,
            },
            cache: true,
        };
        assert!(comparison_source(&selected, "sglang", &mut json!({})).is_err());
    }
    model["comparison_inputs"] =
        json!({"sglang":{"sha256":hash,"tensor_equivalence_sha256":"a".repeat(64)}});
    let selected = Selection {
        model: &model,
        cell: &cell,
        backend: &backend,
        artifact: &artifact,
        port: 1234,
        directory: root.path(),
        capacity: Capacity {
            context: 16384,
            lanes: 4,
        },
        cache: true,
    };
    let mut provenance = json!({});
    assert_eq!(
        comparison_source(&selected, "sglang", &mut provenance).unwrap(),
        alternate
    );
    assert_eq!(provenance["comparison_input_sha256"], hash);
    std::fs::write(alternate.join("config.json"), b"drift").unwrap();
    assert!(comparison_source(&selected, "sglang", &mut json!({})).is_err());
    root.close().unwrap();
}
