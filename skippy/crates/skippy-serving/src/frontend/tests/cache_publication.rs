//! Exact-state cache requests must distinguish queued capture from publication.
use super::*;
use crate::runtime_state::{loaded_memory_cache_capabilities, loaded_model_state_kind};

fn cache_stats(kv: &KvStageIntegration) -> Value {
    let mut stats = serde_json::to_value(
        kv.attrs()
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>(),
    )
    .unwrap();
    // attrs deliberately reports busy instead of blocking on blob hashing.
    // Read the radix independently so the report distinguishes an empty
    // serving index from unavailable aggregate statistics.
    stats["observed_published_entries"] = json!(kv.radix.lock().unwrap().stats().recurrent_entries);
    stats
}

fn request(model: &str) -> Result<ChatCompletionRequest> {
    Ok(serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": include_str!(
            "../../../../../../ci/fixtures/skippy-cache-prompt-input.txt"
        )}],
        "max_tokens": 8,
        "reasoning_effort": "none"
    }))?)
}

fn cached_tokens(response: &ChatCompletionResponse) -> u64 {
    serde_json::to_value(response).unwrap()["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires SKIPPY_WARMUP_STAGE_CONFIG and a local causal GGUF"]
async fn full_state_repeat_reuses_cache_after_publication() -> Result<()> {
    let config_path = env::var_os("SKIPPY_WARMUP_STAGE_CONFIG")
        .context("set SKIPPY_WARMUP_STAGE_CONFIG to the cache smoke config")?;
    let mut config: StageConfig = serde_json::from_slice(&fs::read(config_path)?)?;
    // Reproduce an unadmitted config's conservative full-state fallback.
    config.kv_graph_state.clear();
    let model = config.model_id.clone();
    let mut backend = support::local_openai_backend(config.clone(), model.clone())?;
    let kv = Arc::new(
        KvStageIntegration::from_loaded_model_with_l3_manager(
            &config,
            loaded_model_state_kind(Some(&backend.runtime)),
            loaded_memory_cache_capabilities(Some(&backend.runtime)),
            None,
            None,
        )?
        .context("fixture must enable cache recording")?,
    );
    backend.kv = Some(kv.clone());
    let seed = backend.chat_completion(request(&model)?).await?;
    let after_seed = cache_stats(&kv);
    let immediate = backend.chat_completion(request(&model)?).await?;
    let after_immediate = cache_stats(&kv);
    let drain_started = std::time::Instant::now();
    kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    kv.wait_for_exact_state_recording(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    let drain_ms = drain_started.elapsed().as_secs_f64() * 1000.0;
    let after_drain = cache_stats(&kv);
    let published = backend.chat_completion(request(&model)?).await?;
    assert_eq!(cached_tokens(&seed), 0);
    assert!(
        cached_tokens(&published) > 0,
        "published snapshot must be reusable"
    );
    if let Some(path) = env::var_os("SKIPPY_CACHE_PUBLICATION_REPORT") {
        fs::write(
            path,
            serde_json::to_vec_pretty(&json!({
                "seed_cached_tokens": cached_tokens(&seed),
                "immediate_cached_tokens": cached_tokens(&immediate),
                "published_cached_tokens": cached_tokens(&published),
                "drain_ms": drain_ms,
                "after_seed": after_seed,
                "after_immediate": after_immediate,
                "after_drain": after_drain
            }))?,
        )?;
    }
    Ok(())
}
