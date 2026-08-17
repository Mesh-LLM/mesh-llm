use super::support::*;
use super::*;

#[test]
fn proactive_eviction_attrs_are_bounded_and_request_free() {
    let attrs = proactive_eviction_attrs("error", Some("inactive_session"), 1024, 2, 768);

    assert_eq!(
        attrs.get("skippy.kv.decision"),
        Some(&json!("proactive_eviction"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTION_STATUS),
        Some(&json!("error"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTION_ERROR_KIND),
        Some(&json!("inactive_session"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTION_TARGET_TOKENS),
        Some(&json!(1024))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTED_ENTRIES),
        Some(&json!(2))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTED_TOKENS),
        Some(&json!(768))
    );
    assert!(!attrs.contains_key(attr_key::REQUEST_ID));
    assert!(!attrs.contains_key(attr_key::SESSION_ID));
    assert!(!attrs.contains_key("openai.prompt_cache_key"));
    assert!(!attrs.contains_key("openai.prompt_cache_retention"));
}

#[test]
fn openai_cache_stats_default_to_disabled() {
    let stats = GenerationCacheStats::default();

    assert_eq!(stats.status, "disabled");
    assert_eq!(stats.cached_prompt_tokens, 0);
    assert_eq!(stats.matched_prefix_tokens, 0);
    assert_eq!(stats.suffix_prefill_tokens, 0);
    assert_eq!(stats.hit_kind, None);
}

#[test]
fn cache_identity_reuses_repeated_prompts_without_client_cache_key() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let first_request = prefix_cache_base_with_request("request-a", "session-a");
    let second_request = prefix_cache_base_with_request("request-b", "session-b");
    let tokens = (0..1024).collect::<Vec<_>>();

    let recorded = kv.prefill_identity(&config, &first_request, 0, &tokens);
    let looked_up = kv.prefill_identity(&config, &second_request, 0, &tokens);

    assert_eq!(recorded.page_id, looked_up.page_id);
    assert_eq!(
        recorded.identity.prefix_hash,
        looked_up.identity.prefix_hash
    );
    assert_ne!(recorded.identity.session_id, looked_up.identity.session_id);

    seed_resident_prefix(&kv, &recorded);
    let hit = kv
        .probe_resident_prefix(&looked_up)
        .expect("repeated prompt should hit without a client cache key");
    assert_eq!(hit.page_id, recorded.page_id);
    assert_eq!(hit.token_count, tokens.len());
}

#[test]
fn cache_identity_namespaces_explicit_prompt_cache_keys() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let tokens = (0..1024).collect::<Vec<_>>();
    let mut default_namespace = prefix_cache_test_base();
    default_namespace.chat_template_id = None;
    let mut explicit_namespace = prefix_cache_test_base();
    explicit_namespace.chat_template_id = Some("openai:prompt_cache_key:tenant-a".to_string());

    let default_identity = kv.prefill_identity(&config, &default_namespace, 0, &tokens);
    let explicit_identity = kv.prefill_identity(&config, &explicit_namespace, 0, &tokens);

    assert_ne!(default_identity.page_id, explicit_identity.page_id);
    assert_ne!(
        default_identity.identity.prefix_hash,
        explicit_identity.identity.prefix_hash
    );
}

#[test]
fn disabled_cache_config_has_no_stage_integration() {
    let config = StageConfig {
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::Disabled,
            ..prefix_cache_test_config()
                .kv_cache
                .expect("test cache config")
        }),
        ..prefix_cache_test_config()
    };

    let kv = KvStageIntegration::from_config(&config).unwrap();

    assert!(kv.is_none());
}

#[test]
fn cold_resident_prefix_lookup_misses_before_recording() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let identity = kv.prefill_identity(
        &config,
        &prefix_cache_test_base(),
        0,
        &(0..1024).collect::<Vec<_>>(),
    );

    assert!(kv.probe_resident_prefix(&identity).is_none());
}

#[test]
fn resident_prefix_cache_hits_shared_prefix_grid() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let base = prefix_cache_test_base();
    let recorded_tokens = (0..2214).collect::<Vec<_>>();
    let mut lookup_tokens = recorded_tokens.clone();
    lookup_tokens.extend(100_000..100_017);
    let record_plan = crate::frontend::prefix_cache::stage0_full_prefill_record_identities(
        &kv,
        &config,
        &base,
        &recorded_tokens,
    );
    let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);
    let recorded_shared = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("record plan should include shared grid prefix");
    let lookup_shared = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("lookup plan should probe shared grid prefix");

    seed_resident_prefix(&kv, recorded_shared);
    let hit = kv
        .probe_resident_prefix(lookup_shared)
        .expect("different-tail prompt should hit shared prefix grid");

    assert_eq!(hit.page_id, recorded_shared.page_id);
    assert_eq!(hit.token_count, 2176);
}

#[test]
fn stage0_full_prefill_record_plan_includes_shared_prefix_candidate() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let base = prefix_cache_test_base();
    let recorded_tokens = (0..2214).collect::<Vec<_>>();
    let mut lookup_tokens = recorded_tokens.clone();
    lookup_tokens.extend(100_000..100_017);

    let record_plan = crate::frontend::prefix_cache::stage0_full_prefill_record_identities(
        &kv,
        &config,
        &base,
        &recorded_tokens,
    );
    let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

    let record_counts = record_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();
    let lookup_counts = lookup_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();

    assert_eq!(record_counts, vec![2214, 2176]);
    assert!(lookup_counts.contains(&2176));

    let recorded_shared = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("record plan should include shared grid prefix");
    let lookup_shared = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("lookup plan should probe shared grid prefix");
    let recorded_exact = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2214)
        .expect("record plan should keep exact first prompt");
    let lookup_exact = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2231)
        .expect("lookup plan should probe exact second prompt");

    assert_eq!(recorded_shared.page_id, lookup_shared.page_id);
    assert_ne!(recorded_exact.page_id, lookup_exact.page_id);
}

#[test]
fn stage0_chunked_prefill_record_plan_includes_shared_prefix_candidate() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let base = prefix_cache_test_base();
    let recorded_tokens = (0..2214).collect::<Vec<_>>();
    let mut lookup_tokens = recorded_tokens.clone();
    lookup_tokens.extend(100_000..100_017);

    let record_plan = crate::frontend::prefix_cache::stage0_prefill_record_identities(
        &kv,
        &config,
        &base,
        0,
        &recorded_tokens,
    );
    let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

    let record_counts = record_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();
    let lookup_counts = lookup_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();

    assert_eq!(record_counts, vec![2214, 2176]);
    assert!(lookup_counts.contains(&2176));

    let recorded_shared = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("chunked record plan should include shared grid prefix");
    let lookup_shared = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("lookup plan should probe shared grid prefix");
    let recorded_exact = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2214)
        .expect("chunked record plan should keep exact first prompt");
    let lookup_exact = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2231)
        .expect("lookup plan should probe exact second prompt");

    assert_eq!(recorded_shared.page_id, lookup_shared.page_id);
    assert_ne!(recorded_exact.page_id, lookup_exact.page_id);
}
