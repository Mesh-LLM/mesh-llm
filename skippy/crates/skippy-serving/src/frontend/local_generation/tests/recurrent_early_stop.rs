use super::*;

pub(super) fn generate(
    backend: &StageOpenAiBackend,
    prompt: &[i32],
    stop_after: usize,
) -> Result<(Vec<i32>, u32)> {
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("recurrent-resume"),
        true,
        None,
    );
    let mut output = Vec::new();
    let stats = backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: prompt,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 8,
            sampling: &SamplingConfig::default(),
            chat_sampling_metadata: None,
            speculative: &SpeculativeDecodeConfig::default(),
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &ids,
        },
        |token| {
            output.push(token);
            Ok(if output.len() >= stop_after {
                TokenControl::Stop
            } else {
                TokenControl::Continue
            })
        },
    )?;
    Ok((output, stats.cached_prompt_tokens))
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_prompt_checkpoint_survives_early_stop() -> Result<()> {
    for stop_after in [2, 3] {
        check_prompt_checkpoint_after_stop(stop_after)?;
    }
    Ok(())
}

fn check_prompt_checkpoint_after_stop(stop_after: usize) -> Result<()> {
    let (backend, _) = recurrent_test_backend("resume", "resume", 256, (32, 32), 8, 256)?;
    let prompt = (1..=64).collect::<Vec<_>>();
    // Cover consumer stop before and after handing decode back to the scheduler.
    let (first, _) = generate(&backend, &prompt, stop_after)?;
    assert_eq!(first.len(), stop_after);
    let kv = backend.kv.as_ref().unwrap();
    kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    kv.wait_for_exact_state_recording(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;

    // Reopening the same prompt after consumer stop must preserve the
    // exact checkpoint and reproduce the generated output.
    let (repeated, cached_tokens) = generate(&backend, &prompt, 8)?;
    assert!(cached_tokens > 0, "the prompt checkpoint must be reusable");
    assert_eq!(repeated[..stop_after], first);
    let mut extended = prompt.clone();
    extended.extend_from_slice(&first);
    extended.push(7);
    let (warm, _) = generate(&backend, &extended, 8)?;
    let (cold, _) = recurrent_test_backend("control", "control", 256, (32, 32), 8, 256)?;
    assert_eq!(warm, generate(&cold, &extended, 8)?.0);
    Ok(())
}
