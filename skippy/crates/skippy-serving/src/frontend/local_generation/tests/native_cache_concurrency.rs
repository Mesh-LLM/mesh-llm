//! The same model-backed test runs against dense and hybrid native capabilities.
use super::recurrent_early_stop::generate_after_first_token;
use super::*;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_cache_concurrency_matches_serial_and_reuses_prefixes() -> Result<()> {
    for lanes in [1, 2, 4] {
        let (backend, _) = recurrent_test_backend_with_lanes(
            "native-concurrent",
            "native-concurrent",
            1024,
            (128, 32, lanes),
            8,
            lanes as usize * 1024,
            None,
        )?;
        let prompts: Vec<Vec<i32>> = (0..lanes)
            .map(|lane| (1..=256).map(|token| token + lane as i32 * 257).collect())
            .collect();
        let start = std::sync::Barrier::new(lanes as usize);
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = prompts
                .iter()
                .enumerate()
                .map(|(index, prompt)| {
                    let backend = &backend;
                    let start = &start;
                    scope.spawn(move || {
                        start.wait();
                        generate_after_first_token(
                            backend,
                            prompt,
                            8,
                            &format!("lane-{index}"),
                            || Ok(()),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("generation worker panicked"))
                .collect::<Result<Vec<_>>>()
        })?;
        let kv = backend.kv.as_ref().unwrap();
        kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
            .map_err(anyhow::Error::msg)?;
        kv.wait_for_exact_state_recording(Duration::from_secs(30))
            .map_err(anyhow::Error::msg)?;
        let (control, _) =
            recurrent_test_backend("serial-control", "serial-control", 1024, (128, 32), 8, 1024)?;
        for (index, prompt) in prompts.iter().enumerate() {
            let expected = generate_after_first_token(
                &control,
                prompt,
                8,
                &format!("control-{index}"),
                || Ok(()),
            )?
            .0;
            assert_eq!(
                outputs[index].0, expected,
                "cold parity at c={lanes}, lane={index}"
            );
            let (warm, cached) =
                generate_after_first_token(&backend, prompt, 8, &format!("lane-{index}"), || {
                    Ok(())
                })?;
            assert!(cached > 0, "prefix miss at c={lanes}, lane={index}");
            assert_eq!(warm, expected, "warm parity at c={lanes}, lane={index}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_cache_replay_matches_cold_with_token_margin_evidence() -> Result<()> {
    // Reproduce the prompt that diverged on CUDA without concurrent scheduling.
    // Keep the exact-output assertion: margins are diagnostic evidence, not a
    // tolerance that can turn a wrong continuation into a passing test.
    let prompt: Vec<i32> = (515..=770).collect();
    for lanes in [1, 4] {
        let (backend, _) = recurrent_test_backend_with_lanes(
            "cache-replay",
            "cache-replay",
            1024,
            (128, 32, lanes),
            8,
            lanes as usize * 1024,
            None,
        )?;
        let cold = generate_with_token_margins(&backend, &prompt)?;
        let warm = generate_with_token_margins(&backend, &prompt)?;
        assert!(warm.1 > 0, "prefix miss with {lanes} lanes");
        assert_eq!(
            cold.0, warm.0,
            "cache replay with {lanes} lanes; cold signals={:?}; warm signals={:?}",
            cold.2, warm.2,
        );
    }
    Ok(())
}

fn generate_with_token_margins(
    backend: &StageOpenAiBackend,
    prompt: &[i32],
) -> Result<(Vec<i32>, u32, Vec<skippy_runtime::TokenSignal>)> {
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("cache-replay"),
        true,
        None,
    );
    let mut output = Vec::new();
    let mut signals = Vec::new();
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
            // This test runs one request at a time, so no other decode can
            // replace the current logits before the lazy signal is observed.
            signals.push(
                backend
                    .runtime
                    .lock()
                    .unwrap()
                    .last_token_signal(&ids.session_label)
                    .map_err(crate::frontend::util::openai_backend_error)?,
            );
            Ok(TokenControl::Continue)
        },
    )?;
    Ok((output, stats.cached_prompt_tokens, signals))
}
