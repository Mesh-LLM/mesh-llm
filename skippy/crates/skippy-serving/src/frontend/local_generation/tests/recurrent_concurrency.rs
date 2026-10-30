use super::*;
use recurrent_early_stop::{generate, generate_after_first_token};
use skippy_inference_api::InferenceError;
use std::sync::mpsc;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_concurrent_stop_preserves_both_prompt_checkpoints() -> Result<()> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "concurrent",
        "concurrent",
        256,
        (32, 32, 2),
        8,
        512,
        None,
    )?;
    let prompts = [(1..=64).collect::<Vec<_>>(), (1..=64).rev().collect()];
    let outputs = concurrent_outputs(&backend, &prompts)?;
    assert_eq!(outputs[0].len(), 3);
    assert_eq!(outputs[1].len(), 8);
    let kv = backend.kv.as_ref().unwrap();
    kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    kv.wait_for_exact_state_recording(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    if kv.resident_recurrent_snapshots {
        assert!(
            kv.radix.lock().unwrap().stats().resident_entries >= 2,
            "both prompts must retain resident attention and recurrent snapshots"
        );
    } else {
        assert_eq!(
            backend
                .runtime
                .lock()
                .unwrap()
                .session_stats()
                .idle_resident_prefixes,
            2
        );
    }
    for (index, prompt) in prompts.iter().enumerate() {
        let session = format!("concurrent-{index}");
        let (repeated, cached) =
            generate_after_first_token(&backend, prompt, 8, &session, || Ok(()))?;
        assert!(
            cached > 0,
            "prompt {index} cache miss: {}; {:?}",
            capture_trace::render(),
            kv.attrs()
        );
        assert_eq!(repeated[..outputs[index].len()], outputs[index]);
        let mut extended = prompt.clone();
        extended.extend_from_slice(&outputs[index]);
        extended.push(7);
        let (warm, _) = generate_after_first_token(&backend, &extended, 8, &session, || Ok(()))?;
        let (cold, _) = recurrent_test_backend("control", "control", 256, (32, 32), 8, 256)?;
        assert_eq!(warm, generate(&cold, &extended, 8)?.0);
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_concurrent_retention_respects_idle_pool_limit() -> Result<()> {
    let (backend, _) =
        recurrent_test_backend_with_lanes("bounded", "bounded", 256, (32, 32, 2), 8, 512, Some(1))?;
    let prompts = [(1..=64).collect::<Vec<_>>(), (1..=64).rev().collect()];
    let outputs = concurrent_outputs(&backend, &prompts)?;
    let runtime = backend.runtime.lock().unwrap();
    let stats = runtime.session_stats();
    assert_eq!(stats.active_sessions, 0);
    assert_eq!(stats.idle_sessions, 1);
    assert_eq!(stats.idle_resident_prefixes, 1);
    drop(runtime);
    let kv = backend.kv.as_ref().unwrap();
    kv.wait_for_exact_state_recording(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    // Both the parked lane and the successful overflow lane must survive.
    for (index, prompt) in prompts.iter().enumerate() {
        let (repeated, cached) = generate_after_first_token(
            &backend,
            prompt,
            8,
            &format!("concurrent-{index}"),
            || Ok(()),
        )?;
        assert!(
            cached > 0,
            "overflow completion lost prompt {index}: {}; {:?}",
            capture_trace::render(),
            kv.attrs()
        );
        assert_eq!(repeated[..outputs[index].len()], outputs[index]);
        kv.wait_for_exact_state_recording(Duration::from_secs(30))
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_concurrent_idle_eviction_preserves_active_lane() -> Result<()> {
    let (backend, _) =
        recurrent_test_backend_with_lanes("eviction", "eviction", 256, (32, 32, 2), 8, 512, None)?;
    let prompts = [(1..=64).collect::<Vec<_>>(), (1..=64).rev().collect()];
    concurrent_outputs(&backend, &prompts)?;
    let (actual, cached) =
        generate_after_first_token(&backend, &prompts[0], 8, "concurrent-0", || {
            let mut runtime = backend.runtime.lock().unwrap();
            assert_eq!(runtime.session_stats().active_sessions, 1);
            assert_eq!(
                runtime
                    .evict_idle_recurrent()
                    .map_err(|error| { InferenceError::internal(error.to_string()) })?,
                1
            );
            assert_eq!(runtime.session_stats().active_sessions, 1);
            Ok(())
        })?;
    assert!(cached > 0);
    let (cold, _) = recurrent_test_backend("control", "control", 256, (32, 32), 8, 256)?;
    assert_eq!(actual, generate(&cold, &prompts[0], 8)?.0);
    Ok(())
}

fn concurrent_outputs(
    backend: &StageOpenAiBackend,
    prompts: &[Vec<i32>; 2],
) -> Result<Vec<Vec<i32>>> {
    thread::scope(|scope| {
        let (ready, received) = mpsc::channel();
        let mut releases = Vec::new();
        let mut tasks = Vec::new();
        for (index, prompt) in prompts.iter().enumerate() {
            let ready = ready.clone();
            let (release, released) = mpsc::channel();
            releases.push(release);
            tasks.push(scope.spawn(move || {
                generate_after_first_token(
                    backend,
                    prompt,
                    if index == 0 { 3 } else { 8 },
                    &format!("concurrent-{index}"),
                    || {
                        ready
                            .send(())
                            .map_err(|error| InferenceError::internal(error.to_string()))?;
                        released
                            .recv_timeout(Duration::from_secs(30))
                            .map_err(|error| InferenceError::internal(error.to_string()))?;
                        Ok(())
                    },
                )
            }));
        }
        // Both native sessions must be live before either continues decoding.
        for _ in 0..2 {
            received.recv_timeout(Duration::from_secs(30))?;
        }
        for release in releases {
            release.send(())?;
        }
        tasks
            .into_iter()
            .map(|task| {
                task.join()
                    .expect("concurrent generation thread panicked")
                    .map(|output| output.0)
            })
            .collect()
    })
}
