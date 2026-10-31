//! Exact-state cache requests must distinguish queued capture from publication.
use super::*;
use crate::kv_integration::StagePrefixCachePayload;
use crate::runtime_state::{loaded_memory_cache_capabilities, loaded_model_state_kind};
use skippy_inference_api::chat::ChatCompletionRequest;

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

fn request(model: &str) -> Result<GenerationRequest> {
    Ok(serde_json::from_value::<ChatCompletionRequest>(json!({
        "model": model,
        "messages": [{"role": "user", "content": include_str!(
            "../../../../../../ci/fixtures/skippy-cache-prompt-input.txt"
        )}],
        "max_tokens": 8,
        "reasoning_effort": "none"
    }))?
    .into())
}

fn cached_tokens(response: &GenerationResponse) -> u64 {
    serde_json::to_value(skippy_inference_api::chat::ChatCompletionResponse::from(
        response.clone(),
    ))
    .unwrap()["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires SKIPPY_WARMUP_STAGE_CONFIG and a resident-cache-capable GGUF"]
async fn loaded_runtime_selects_cache_and_reuses_immediate_repeat() -> Result<()> {
    let config_path = env::var_os("SKIPPY_WARMUP_STAGE_CONFIG")
        .context("set SKIPPY_WARMUP_STAGE_CONFIG to the cache smoke config")?;
    let config: StageConfig = serde_json::from_slice(&fs::read(config_path)?)?;
    // Cache selection must come from the loaded runtime, with no capability hints.
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
    assert_eq!(
        kv.payload,
        StagePrefixCachePayload::ResidentKv,
        "this regression requires a runtime that admits resident cache state"
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
        cached_tokens(&immediate) > 0,
        "immediate repeat must reuse admitted state"
    );
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
