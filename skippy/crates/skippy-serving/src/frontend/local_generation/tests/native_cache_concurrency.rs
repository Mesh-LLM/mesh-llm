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
            if warm != expected {
                let divergence = warm
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(warm.len().min(expected.len()));
                let tokens = u32::try_from(divergence + 1)?;
                let warm_evidence = generate_with_token_margins(
                    &backend,
                    prompt,
                    &format!("lane-{index}"),
                    tokens,
                )?;
                let control_evidence = generate_with_token_margins(
                    &control,
                    prompt,
                    &format!("control-{index}"),
                    tokens,
                )?;
                panic!(
                    "warm parity at c={lanes}, lane={index}: expected={expected:?}, actual={warm:?}; repeated warm={warm_evidence:?}; serial warm={control_evidence:?}"
                );
            }
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
        let cold = generate_with_token_margins(&backend, &prompt, "cache-replay", 8)?;
        let warm = generate_with_token_margins(&backend, &prompt, "cache-replay", 8)?;
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
    session: &str,
    max_tokens: u32,
) -> Result<(Vec<i32>, u32, Vec<skippy_runtime::TokenSignal>)> {
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some(session),
        true,
        None,
    );
    let mut output = Vec::new();
    let mut signals = Vec::new();
    let stats = backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: prompt,
            recurrent_cache_prefix_token_ids: None,
            max_tokens,
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
            // Planned decode can buffer events before callbacks consume them.
            // Inspect only the terminal token, with no other request active;
            // earlier callbacks may already observe the final logits.
            if output.len() == max_tokens as usize {
                let signal = backend
                    .runtime
                    .lock()
                    .unwrap()
                    .last_token_signal(&ids.session_label)
                    .map_err(crate::frontend::util::openai_backend_error)?;
                assert_eq!(
                    signal.top_token, token,
                    "terminal signal must match its token"
                );
                signals.push(signal);
            }
            Ok(TokenControl::Continue)
        },
    )?;
    Ok((output, stats.cached_prompt_tokens, signals))
}
