use super::recurrent_early_stop::generate;
use super::*;
use std::sync::atomic::Ordering;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_records_full_prompt_after_deferred_prefill() -> Result<()> {
    let (backend, _) = recurrent_test_backend("full-prompt", "full-prompt", 256, (32, 32), 8, 256)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    generate(&backend, &prompt, 8)?;
    let kv = backend.kv.as_ref().unwrap();
    anyhow::ensure!(
        kv.resident_recurrent_snapshots,
        "test requires hybrid resident KV"
    );
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("recurrent-resume"),
        true,
        None,
    );
    let base = backend.local_kv_message_base("recurrent-resume", &ids);
    let identity = kv.prefill_identity(&backend.config, &base, 0, &prompt);
    assert!(
        kv.radix
            .lock()
            .unwrap()
            .resident_exact(&identity.namespace, &identity.token_ids)
            .is_some(),
        "the deferred record must retain the full prompt at the native session position"
    );
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_native_batch_shape_parity() -> Result<()> {
    let (wide, _) = recurrent_test_backend("native-wide", "native-wide", 512, (128, 32), 8, 256)?;
    let (narrow, _) =
        recurrent_test_backend("native-narrow", "native-narrow", 512, (32, 32), 8, 256)?;
    let prompt = (1..=128).collect::<Vec<_>>();
    let mut wide_session = wide.runtime.lock().unwrap().model.create_session()?;
    let mut narrow_session = narrow.runtime.lock().unwrap().model.create_session()?;
    for chunk in prompt[..127].chunks(32) {
        wide_session.prefill_chunk(chunk)?;
        narrow_session.prefill_chunk(chunk)?;
    }
    let mut wide_token = wide_session.decode_step(prompt[127])?;
    let mut narrow_token = narrow_session.decode_step(prompt[127])?;
    assert_eq!(
        wide_token, narrow_token,
        "native first token differs across batch sizes"
    );
    for index in 1..8 {
        wide_token = wide_session.decode_step(wide_token)?;
        narrow_token = narrow_session.decode_step(narrow_token)?;
        assert_eq!(
            wide_token, narrow_token,
            "native generated token {index} differs across batch sizes"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_prefill_spans_native_microbatches() -> Result<()> {
    // The extended prompt reaches 150 tokens. Leave the 128-token decode
    // headroom requirement outside this microbatch-shape comparison.
    let (wide, _) = recurrent_test_backend("wide", "wide", 512, (128, 32), 8, 256)?;
    let (narrow, _) = recurrent_test_backend("narrow", "narrow", 512, (32, 32), 8, 256)?;
    let prompt = (1..=128).collect::<Vec<_>>();
    let (expected, _) = generate(&narrow, &prompt, 8)?;
    let (actual, _) = generate(&wide, &prompt, 8)?;
    assert_eq!(
        actual, expected,
        "native microbatch splitting changes output"
    );
    let (repeated, cached) = generate(&wide, &prompt, 8)?;
    assert!(cached > 0);
    assert_eq!(repeated, expected);
    let mut extended = prompt;
    extended.extend_from_slice(&actual);
    extended.extend(129..=150);
    assert_eq!(
        generate(&wide, &extended, 8)?.0,
        generate(&narrow, &extended, 8)?.0,
        "growing resident turns differ across batch sizes"
    );
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_unload_demotes_to_l3_and_reopens_after_restart() -> Result<()> {
    let root = tempfile::tempdir()?;
    let manager = skippy_cache::L3CacheManager::acquire(
        root.path(),
        skippy_cache::StoreLimits::new(1 << 30, 0),
    )?;
    let (mut backend, _) = recurrent_test_backend("durable", "durable", 256, (32, 32), 8, 256)?;
    let configure = |backend: &mut StageOpenAiBackend| -> Result<()> {
        let kv = backend.kv.as_ref().unwrap();
        backend.kv = KvStageIntegration::from_loaded_model_with_l3_manager(
            &backend.config,
            kv.loaded_state_kind,
            kv.loaded_memory_cache,
            Some(manager.clone()),
            None,
        )?
        .map(Arc::new);
        Ok(())
    };
    configure(&mut backend)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    let (expected, _) = generate(&backend, &prompt, 8)?;
    drain(&backend)?;
    let kv = backend.kv.as_ref().unwrap().clone();
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("recurrent-resume"),
        true,
        None,
    );
    let base = backend.local_kv_message_base("probe", &ids);
    let identity = kv.prefill_identity(&backend.config, &base, 0, &prompt[..63]);
    let l3 = kv.l3.as_ref().unwrap();
    assert!(
        l3.locate_longest(&identity.namespace, &identity.token_ids, 64)?
            .is_none()
    );
    drop(backend);
    kv.wait_for_exact_state_recording(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while l3
        .locate_longest(&identity.namespace, &identity.token_ids, 64)?
        .is_none()
    {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "resident demotion did not publish L3"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let (mut reopened, _) = recurrent_test_backend("durable", "durable", 256, (32, 32), 8, 256)?;
    configure(&mut reopened)?;
    let (actual, reused) = generate(&reopened, &prompt, 8)?;
    assert!(reused > 0);
    assert_eq!(expected, actual);
    assert_eq!(
        reopened
            .runtime
            .lock()
            .unwrap()
            .session_stats()
            .idle_resident_prefixes,
        1
    );
    Ok(())
}

fn drain(backend: &StageOpenAiBackend) -> Result<()> {
    let kv = backend.kv.as_ref().unwrap();
    kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    kv.wait_for_exact_state_recording(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)
}

fn drain_after_cleanup(backend: &StageOpenAiBackend) -> Result<()> {
    let kv = backend.kv.as_ref().unwrap();
    kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while kv.exact_state_records_pending.load(Ordering::Acquire) != 0 {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "recorder did not drain after cancellation"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    anyhow::ensure!(
        kv.exact_state_record_worker_panics.load(Ordering::Acquire) == 0,
        "recorder panicked during cancellation cleanup"
    );
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_cancellation_and_error_release_the_lane() -> Result<()> {
    let (backend, _) = recurrent_test_backend("cleanup", "cleanup", 256, (32, 32), 8, 256)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    for cancel in [true, false] {
        generate(&backend, &prompt, 8)?;
        drain_after_cleanup(&backend)?;
        let cancellation = skippy_inference_api::CancellationToken::new();
        let ids = InferenceGenerationIds::new_with_trust(
            InferenceCacheHints::default(),
            Some("recurrent-resume"),
            true,
            None,
        );
        let result = backend.generate_local_tokens(
            LocalGeneration {
                prompt_token_ids: &prompt,
                recurrent_cache_prefix_token_ids: None,
                max_tokens: 8,
                sampling: &SamplingConfig::default(),
                chat_sampling_metadata: None,
                speculative: &SpeculativeDecodeConfig::default(),
                native_mtp_enabled: false,
                hook_request: None,
                hook_runtime: None,
                cancellation: Some(&cancellation),
                ids: &ids,
            },
            |_| {
                if cancel {
                    cancellation.cancel();
                    Ok(TokenControl::Continue)
                } else {
                    Err(crate::frontend::util::openai_backend_error(
                        anyhow::anyhow!("test consumer failure"),
                    ))
                }
            },
        );
        if !cancel {
            assert!(result.is_err());
        }
        drain_after_cleanup(&backend)?;
        let stats = backend.runtime.lock().unwrap().session_stats();
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.idle_resident_prefixes, 0);
    }
    generate(&backend, &prompt, 8)?;
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_retains_lane_and_demotes_on_unrelated_prompt() -> Result<()> {
    let (backend, _) = recurrent_test_backend("resident", "resident", 256, (32, 32), 8, 256)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    let (first, _) = generate(&backend, &prompt, 8)?;
    drain(&backend)?;
    let kv = backend.kv.as_ref().unwrap();
    let runtime = backend.runtime.as_ref();
    assert_eq!(
        runtime
            .lock()
            .unwrap()
            .session_stats()
            .idle_resident_prefixes,
        1
    );

    let mut extension = prompt.clone();
    extension.extend_from_slice(&first);
    extension.extend_from_slice(&[7, 8]);
    let (warm, reused) = generate(&backend, &extension, 8)?;
    assert!(reused >= 63);
    let (cold, _) = recurrent_test_backend("cold", "cold", 256, (32, 32), 8, 256)?;
    generate(&cold, &prompt, 8)?;
    drain(&cold)?;
    cold.runtime.lock().unwrap().evict_idle_recurrent()?;
    drain(&cold)?;
    let (control, control_reused) = generate(&cold, &extension, 8)?;
    assert_eq!(
        reused, control_reused,
        "compare the same checkpoint boundary"
    );
    assert_eq!(warm, control);
    drain(&backend)?;

    let unrelated = (100..164).collect::<Vec<_>>();
    generate(&backend, &unrelated, 8)?;
    drain(&backend)?;
    assert!(
        kv.exact_state_records_queued.load(Ordering::Relaxed) > 0,
        "lane eviction must offer serialized checkpoints to the existing tiers"
    );
    let (restored, reused) = generate(&backend, &prompt, 8)?;
    assert!(
        reused > 0,
        "evicted prefix must remain recoverable from snapshots"
    );
    assert_eq!(first, restored);
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_reopens_evicted_prefix_from_lane_recorded_fallback() -> Result<()> {
    let (backend, _) = recurrent_test_backend("fallback", "fallback", 256, (32, 32), 8, 256)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    let (first, _) = generate(&backend, &prompt, 8)?;
    drain(&backend)?;
    let kv = backend.kv.as_ref().unwrap();
    assert!(kv.l3.is_none(), "the fallback must work without disk");
    assert!(kv.records_lane_owned_recurrent());

    let unrelated = (100..164).collect::<Vec<_>>();
    let eviction_ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("eviction-control"),
        true,
        None,
    );
    let runtime_session_id = eviction_ids.session_label.clone();
    let mut output_tokens = 0usize;
    backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &unrelated,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 8,
            sampling: &SamplingConfig::default(),
            chat_sampling_metadata: None,
            speculative: &SpeculativeDecodeConfig::default(),
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &eviction_ids,
        },
        |_| {
            if output_tokens == 0 {
                let mut runtime = backend.runtime.lock().unwrap();
                let evicted = kv
                    .evict_resident_prefix_for_tokens(&mut runtime, &runtime_session_id, u64::MAX)
                    .map_err(crate::frontend::util::openai_backend_error)?;
                assert!(evicted.evicted_entries > 0);
            }
            output_tokens += 1;
            Ok(TokenControl::Continue)
        },
    )?;
    drain(&backend)?;
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("recurrent-resume"),
        true,
        None,
    );
    let base = backend.local_kv_message_base("recurrent-resume", &ids);
    let identity = kv.prefill_identity(&backend.config, &base, 0, &prompt);
    assert!(kv.probe_resident_prefix(&identity).is_none());
    let (restored, reused) = generate(&backend, &prompt, 8)?;
    assert!(reused > 0, "eviction lost the exact-state fallback");
    assert_eq!(first, restored);
    Ok(())
}
