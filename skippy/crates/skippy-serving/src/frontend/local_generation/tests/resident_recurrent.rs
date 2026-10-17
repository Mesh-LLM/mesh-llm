use super::recurrent_early_stop::generate;
use super::*;
use std::sync::atomic::Ordering;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_demotes_to_l3_and_reopens_after_restart() -> Result<()> {
    let root = tempfile::tempdir()?;
    let manager = skippy_cache::L3CacheManager::acquire(
        root.path(),
        skippy_cache::StoreLimits::new(1 << 30, 0),
    )?;
    let (mut backend, _) = recurrent_test_backend("durable", "durable", 256, 32, 8, 256)?;
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
    let kv = backend.kv.as_ref().unwrap();
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
    assert_eq!(backend.runtime.lock().unwrap().evict_idle_recurrent()?, 1);
    drain(&backend)?;
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
    let (mut reopened, _) = recurrent_test_backend("durable", "durable", 256, 32, 8, 256)?;
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

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_cancellation_and_error_release_the_lane() -> Result<()> {
    let (backend, _) = recurrent_test_backend("cleanup", "cleanup", 256, 32, 8, 256)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    for cancel in [true, false] {
        generate(&backend, &prompt, 8)?;
        drain(&backend)?;
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
        drain(&backend)?;
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
    let (backend, _) = recurrent_test_backend("resident", "resident", 256, 32, 8, 256)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    let (first, _) = generate(&backend, &prompt, 8)?;
    drain(&backend)?;
    let kv = backend.kv.as_ref().unwrap();
    assert_eq!(
        kv.exact_state_records_queued.load(Ordering::Relaxed),
        0,
        "hot captures must not serialize attention KV"
    );
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
    let (cold, _) = recurrent_test_backend("cold", "cold", 256, 32, 8, 256)?;
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
    assert_eq!(
        kv.exact_state_records_queued.load(Ordering::Relaxed),
        0,
        "growing resident turns must not export attention merely to replace old checkpoints"
    );

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
