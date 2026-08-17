use super::*;

pub(super) fn prefix_cache_test_config() -> StageConfig {
    StageConfig {
        run_id: "run".to_string(),
        topology_id: "topology".to_string(),
        model_id: "org/model:Q4_K_M".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: None,
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 4,
        ctx_size: 8192,
        lane_count: 2,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: 0,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: Default::default(),
        filter_tensors_on_load: false,
        selected_device: None,
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            payload: StageKvCachePayload::ResidentKv,
            max_entries: 8,
            max_bytes: 0,
            min_tokens: 256,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: 2,
        }),
        native_mtp_enabled: true,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: Some(PeerConfig {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            endpoint: "127.0.0.1:0".to_string(),
        }),
    }
}

pub(super) fn prefix_cache_test_base() -> MessageBase {
    MessageBase {
        schema_version: SCHEMA_VERSION,
        run_id: "run".to_string(),
        request_id: "request".to_string(),
        session_id: "session".to_string(),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        topology_id: "topology".to_string(),
        model_id: Some("org/model:Q4_K_M".to_string()),
        tokenizer_id: None,
        chat_template_id: Some("template".to_string()),
        seq: Some(1),
    }
}

pub(super) fn prefix_cache_base_with_request(request_id: &str, session_id: &str) -> MessageBase {
    MessageBase {
        request_id: request_id.to_string(),
        session_id: session_id.to_string(),
        ..prefix_cache_test_base()
    }
}

pub(super) fn seed_resident_prefix(kv: &KvStageIntegration, identity: &PrefillKvIdentity) {
    let token_count = identity.identity.token_count;
    let mut cache = kv.resident.lock().expect("resident cache lock poisoned");
    let allocation = cache
        .allocate_for_record(&identity.page_id, token_count, token_count, |_| Ok(()))
        .expect("synthetic resident prefix should allocate");
    assert!(allocation.should_retain);
    cache.commit_record(
        identity.page_id.clone(),
        allocation.seq_id,
        token_count,
        token_count,
    );
}

pub(super) fn unsupported_code(error: OpenAiError) -> Option<String> {
    error.body().error.code
}

pub(super) fn test_request_defaults() -> EmbeddedOpenAiRequestDefaults {
    EmbeddedOpenAiRequestDefaults {
        stop: Some(vec!["</stop>".to_string()]),
        temperature: Some(0.2),
        top_p: Some(0.9),
        presence_penalty: Some(1.25),
        frequency_penalty: Some(0.5),
        seed: Some(77),
        logit_bias: Some(std::collections::BTreeMap::from([
            ("123".to_string(), json!(-50.0)),
            ("456".to_string(), json!(12.5)),
        ])),
        top_k: Some(12),
        min_p: Some(0.1),
        repeat_penalty: Some(1.2),
        repeat_last_n: Some(64),
        reasoning_format: Some(EmbeddedReasoningFormat::Hidden),
        reasoning_enabled: Some(EmbeddedReasoningEnabled::Enabled),
        reasoning_budget: Some(EmbeddedReasoningBudget::Tokens(256)),
    }
}

pub(super) fn tool_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "look this up"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }]
    }))
    .unwrap()
}
