//! Changed-tail follow-ups must reuse a nearby exact recurrent checkpoint.
use super::*;

fn generate_chat_tail(backend: &StageOpenAiBackend, prompt: &[i32]) -> Result<(Vec<i32>, u32)> {
    generate_chat_tail_session(backend, prompt, "shortened-tail")
}

fn generate_chat_tail_session(
    backend: &StageOpenAiBackend,
    prompt: &[i32],
    session: &str,
) -> Result<(Vec<i32>, u32)> {
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some(session),
        true,
        None,
    );
    let mut output = Vec::new();
    let stats = backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: prompt,
            recurrent_cache_prefix_token_ids: Some(&prompt[..prompt.len() - 4]),
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
            Ok(TokenControl::Continue)
        },
    )?;
    let kv = backend.kv.as_ref().unwrap();
    kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    Ok((output, stats.cached_prompt_tokens))
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_shortened_tail_reuses_nearby_checkpoint() -> Result<()> {
    check_shortened_tail(false)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_shortened_tail_survives_demotion() -> Result<()> {
    check_shortened_tail(true)
}

fn check_shortened_tail(demote: bool) -> Result<()> {
    let (warm, _) = recurrent_test_backend("tail", "tail", 256, (32, 32), 8, 256)?;
    let original = (1..=192).collect::<Vec<_>>();
    generate_chat_tail(&warm, &original)?;
    if demote {
        warm.runtime.lock().unwrap().evict_idle_recurrent()?;
        warm.kv
            .as_ref()
            .unwrap()
            .wait_for_exact_state_recording(Duration::from_secs(30))
            .map_err(anyhow::Error::msg)?;
    }

    // The next prompt changes before both the previous chat boundary (188)
    // and its full prompt (192). The last complete microbatch boundary before
    // that tail (128) remains valid, so replaying the prefix is unnecessary.
    let mut changed = original[..176].to_vec();
    changed.extend_from_slice(&[501, 502, 503, 504]);
    let (actual, cached) = generate_chat_tail(&warm, &changed)?;
    assert_eq!(
        cached, 128,
        "shortened chat tail discarded its nearby checkpoint"
    );
    assert!(cached <= 176, "restored beyond the unchanged prefix");

    let (cold, _) = recurrent_test_backend("cold-tail", "cold-tail", 256, (32, 32), 8, 256)?;
    assert_eq!(
        actual,
        generate_chat_tail(&cold, &changed)?.0,
        "restoring a shortened tail must preserve generated tokens"
    );
    let kv = warm.kv.as_ref().unwrap();
    let exports = kv
        .exact_state_records_queued
        .load(std::sync::atomic::Ordering::Relaxed);
    if demote {
        assert!(
            exports > 0,
            "eviction must offer checkpoints to lower tiers"
        );
    } else {
        assert_eq!(
            exports, 0,
            "resident checkpoints must not export attention KV"
        );
    }
    Ok(())
}

fn generate_concurrent_tails(
    backend: &StageOpenAiBackend,
    prompts: &[Vec<i32>],
) -> Result<Vec<(Vec<i32>, u32)>> {
    let start = std::sync::Barrier::new(prompts.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| {
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    generate_chat_tail_session(backend, prompt, &format!("tail-{index}"))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("generation worker panicked"))
            .collect()
    })
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_hybrid_shortened_tail_preserves_concurrent_lanes() -> Result<()> {
    for lanes in [2, 4] {
        let (warm, _) = recurrent_test_backend_with_lanes(
            "concurrent-tails",
            "concurrent-tails",
            256,
            (32, 32, lanes),
            8,
            256 * lanes as usize,
            None,
        )?;
        let prompts: Vec<Vec<i32>> = (0..lanes)
            .map(|lane| (1..=192).map(|token| token + 257 * lane as i32).collect())
            .collect();
        generate_concurrent_tails(&warm, &prompts)?;
        let changed: Vec<_> = prompts
            .iter()
            .map(|prompt| {
                let mut changed = prompt[..176].to_vec();
                changed.extend_from_slice(&[501, 502, 503, 504]);
                changed
            })
            .collect();
        let actual = generate_concurrent_tails(&warm, &changed)?;
        let (cold, _) = recurrent_test_backend("cold-tails", "cold-tails", 256, (32, 32), 8, 256)?;
        for (index, prompt) in changed.iter().enumerate() {
            assert_eq!(actual[index].1, 128, "c={lanes}, lane={index}");
            assert_eq!(
                actual[index].0,
                generate_chat_tail(&cold, prompt)?.0,
                "concurrent checkpoint restore changes tokens: c={lanes}, lane={index}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn shortened_tail_cold_concurrency_matches_serial() -> Result<()> {
    let (parallel, _) = recurrent_test_backend_with_lanes(
        "cold-parallel",
        "cold-parallel",
        256,
        (32, 32, 4),
        8,
        1024,
        None,
    )?;
    let prompts: Vec<Vec<i32>> = (0..4)
        .map(|lane| {
            let mut prompt: Vec<_> = (1..=176).map(|token| token + 257 * lane).collect();
            prompt.extend_from_slice(&[501, 502, 503, 504]);
            prompt
        })
        .collect();
    let actual = generate_concurrent_tails(&parallel, &prompts)?;
    let (serial, _) = recurrent_test_backend("cold-serial", "cold-serial", 256, (32, 32), 8, 256)?;
    for (lane, prompt) in prompts.iter().enumerate() {
        assert_eq!(
            actual[lane].0,
            generate_chat_tail(&serial, prompt)?.0,
            "cold concurrency changes tokens: lane={lane}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn shortened_tail_fourth_prompt_matches_serial_restore() -> Result<()> {
    let (warm, _) = recurrent_test_backend("fourth-warm", "fourth-warm", 256, (32, 32), 8, 256)?;
    let original: Vec<_> = (1..=192).map(|token| token + 257 * 3).collect();
    generate_chat_tail(&warm, &original)?;
    let mut changed = original[..176].to_vec();
    changed.extend_from_slice(&[501, 502, 503, 504]);
    let actual = generate_chat_tail(&warm, &changed)?;
    let (cold, _) = recurrent_test_backend("fourth-cold", "fourth-cold", 256, (32, 32), 8, 256)?;
    assert_eq!(actual.1, 128);
    assert_eq!(actual.0, generate_chat_tail(&cold, &changed)?.0);
    Ok(())
}
