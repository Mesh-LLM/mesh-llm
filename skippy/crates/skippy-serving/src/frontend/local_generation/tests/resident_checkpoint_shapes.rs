//! Compare resident restore with cold execution at identical native batch shapes.
//! Cross-shape numerical parity is covered separately in `native_decode_shapes`.
use super::*;
use native_decode_shapes::iterate;
use skippy_runtime::IterationBatchPhase;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn resident_shortened_tail_matches_cold_at_fixed_batch_shapes() -> Result<()> {
    for lanes in [2, 4] {
        compare_resident_restore(lanes)?;
    }
    Ok(())
}

fn compare_resident_restore(lanes: u32) -> Result<()> {
    let (warm, _) = recurrent_test_backend_with_lanes(
        "fixed-warm",
        "fixed-warm",
        256,
        (128, 128, lanes),
        8,
        256 * lanes as usize,
        None,
    )?;
    let (cold, _) = recurrent_test_backend_with_lanes(
        "fixed-cold",
        "fixed-cold",
        256,
        (128, 128, lanes),
        8,
        256 * lanes as usize,
        None,
    )?;
    let names: Vec<_> = (0..lanes).map(|lane| format!("fixed-{lane}")).collect();
    let sessions: Vec<_> = names.iter().map(String::as_str).collect();
    let prompts: Vec<Vec<i32>> = (0..lanes)
        .map(|lane| (1..=192).map(|token| token + 257 * lane as i32).collect())
        .collect();
    let changed: Vec<_> = prompts
        .iter()
        .map(|prompt| {
            let mut changed = prompt[..176].to_vec();
            changed.extend([501, 502, 503, 504]);
            changed
        })
        .collect();
    let ids: Vec<_> = sessions
        .iter()
        .map(|session| {
            InferenceGenerationIds::new_with_trust(
                InferenceCacheHints::default(),
                Some(session),
                true,
                None,
            )
        })
        .collect();
    let kv = warm.kv.as_ref().unwrap();
    let mut warm_runtime = warm.runtime.lock().unwrap();
    let mut cold_runtime = cold.runtime.lock().unwrap();
    for session in &sessions {
        warm_runtime.enable_resident_recurrent_request(session);
    }
    // Both arms execute the same native shapes up to the retained boundary.
    // The warm arm then visits an obsolete tail before restoring it.
    fixed_prefill(&mut warm_runtime, &sessions, &prompts, 0, 128)?;
    fixed_prefill(&mut cold_runtime, &sessions, &prompts, 0, 128)?;
    for ((session, prompt), ids) in sessions.iter().zip(&prompts).zip(&ids) {
        assert!(warm.record_exact_state_at_tokens(
            &mut warm_runtime,
            session,
            ids,
            &prompt[..128],
            "shape_checkpoint",
            true,
        ));
    }
    fixed_prefill(&mut warm_runtime, &sessions, &prompts, 128, 192)?;
    for session in &sessions {
        assert!(warm_runtime.park_resident_recurrent(session));
    }
    for ((session, prompt), ids) in sessions.iter().zip(&changed).zip(&ids) {
        warm_runtime.enable_resident_recurrent_request(session);
        let base = warm.local_kv_message_base(session, ids);
        let identity = kv.prefill_identity(&warm.config, &base, 0, prompt);
        let restored = warm_runtime
            .restore_resident_recurrent(session, &[identity])?
            .expect("resident checkpoint must be available");
        assert_eq!(restored.token_count, 128);
        assert_eq!(restored.source, "resident_recurrent");
    }
    let warm_output = fixed_continuation(&mut warm_runtime, &sessions, &changed)?;
    let cold_output = fixed_continuation(&mut cold_runtime, &sessions, &changed)?;
    assert_eq!(
        warm_output, cold_output,
        "resident restore changed output at c={lanes}"
    );
    for session in &sessions {
        assert!(
            warm_runtime.export_recurrent_cache_state(session)?
                == cold_runtime.export_recurrent_cache_state(session)?,
            "resident restore changed recurrent state at c={lanes}, session={session}",
        );
    }
    assert_eq!(
        kv.exact_state_records_queued
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    Ok(())
}

fn fixed_prefill(
    runtime: &mut RuntimeState,
    sessions: &[&str],
    prompts: &[Vec<i32>],
    start: usize,
    end: usize,
) -> Result<Vec<i32>> {
    let mut predictions = Vec::new();
    for offset in (start..end).step_by(16) {
        let tokens: Vec<_> = prompts
            .iter()
            .map(|prompt| &prompt[offset..(offset + 16).min(end)])
            .collect();
        predictions = iterate(runtime, sessions, &tokens, IterationBatchPhase::Prefill)?;
    }
    Ok(predictions)
}

fn fixed_continuation(
    runtime: &mut RuntimeState,
    sessions: &[&str],
    prompts: &[Vec<i32>],
) -> Result<Vec<Vec<i32>>> {
    let mut next = fixed_prefill(runtime, sessions, prompts, 128, 180)?;
    let mut output: Vec<_> = next.iter().map(|&token| vec![token]).collect();
    for _ in 1..8 {
        let inputs: Vec<_> = next.iter().map(std::slice::from_ref).collect();
        next = iterate(runtime, sessions, &inputs, IterationBatchPhase::Decode)?;
        for (tokens, &token) in output.iter_mut().zip(&next) {
            tokens.push(token);
        }
    }
    Ok(output)
}
