//! Exercise the Rust accounting against actual native submissions and cache fences.
use super::*;
use crate::runtime_state::RuntimeIterationBatchRequest;
use skippy_runtime::IterationBatchPhase;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_execution_accounting_tracks_prefill_decode_and_cache_fences() -> Result<()> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "completion-accounting",
        "completion-accounting",
        1024,
        (128, 32, 4),
        8,
        4096,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let sessions: Vec<_> = (0..4).map(|lane| format!("accounting-{lane}")).collect();
    for session in &sessions {
        exercise_lane(&mut runtime, session)?;
    }
    let sampling = SamplingConfig::default();
    for lanes in [1, 2, 4] {
        let requests: Vec<_> = sessions
            .iter()
            .take(lanes)
            .map(|session| RuntimeIterationBatchRequest {
                session_id: session,
                token_ids: &[4],
                positions: &[],
                sampling: Some(&sampling),
                input: None,
                sample_last: true,
                phase: IterationBatchPhase::Decode,
            })
            .collect();
        let output = runtime.iteration_batch_sampled(&requests)?;
        assert_eq!(output.samples.len(), lanes);
        let batch = runtime
            .take_completed_execution()
            .expect("completed decode batch");
        assert_eq!(
            (batch.prompt_tokens, batch.decode_tokens),
            (0, lanes as u64)
        );
    }
    assert!(
        iteration(
            &mut runtime,
            &sessions[0],
            &[],
            IterationBatchPhase::Decode,
            true
        )
        .is_err()
    );
    assert!(runtime.take_completed_execution().is_none());
    iteration(
        &mut runtime,
        &sessions[0],
        &[5],
        IterationBatchPhase::Decode,
        true,
    )?;
    assert!(
        runtime.take_completed_execution().is_none(),
        "failed work must taint the first successful completion"
    );
    iteration(
        &mut runtime,
        &sessions[0],
        &[6],
        IterationBatchPhase::Decode,
        true,
    )?;
    let recovered = runtime
        .take_completed_execution()
        .expect("recovered timing");
    assert_eq!((recovered.prompt_tokens, recovered.decode_tokens), (0, 1));
    Ok(())
}

fn iteration(
    runtime: &mut RuntimeState,
    session: &str,
    tokens: &[i32],
    phase: IterationBatchPhase,
    sample: bool,
) -> Result<()> {
    let sampling = SamplingConfig::default();
    runtime.iteration_batch_sampled(&[RuntimeIterationBatchRequest {
        session_id: session,
        token_ids: tokens,
        positions: &[],
        sampling: Some(&sampling),
        input: None,
        sample_last: sample,
        phase,
    }])?;
    Ok(())
}

fn exercise_lane(runtime: &mut RuntimeState, session: &str) -> Result<()> {
    let prompt = vec![1; 64];
    iteration(
        runtime,
        session,
        &prompt,
        IterationBatchPhase::Prefill,
        false,
    )?;
    assert!(runtime.take_completed_execution().is_none());
    iteration(
        runtime,
        session,
        &prompt,
        IterationBatchPhase::Prefill,
        true,
    )?;
    let completed = runtime.take_completed_execution().expect("prefill fence");
    assert_eq!((completed.prompt_tokens, completed.decode_tokens), (128, 0));
    assert!(!completed.elapsed.is_zero());

    iteration(
        runtime,
        session,
        &prompt,
        IterationBatchPhase::Prefill,
        false,
    )?;
    assert!(runtime.take_completed_execution().is_none());
    iteration(runtime, session, &[2], IterationBatchPhase::Decode, true)?;
    let mixed = runtime.take_completed_execution().expect("decode fence");
    assert_eq!((mixed.prompt_tokens, mixed.decode_tokens), (64, 1));
    // Feeding this completed epoch to the estimator must not train either rate.
    let estimator = crate::frontend::generation::GenerationServiceEstimator::default();
    estimator.observe_execution(
        crate::frontend::generation::GenerationAdmissionWork {
            prompt_tokens: mixed.prompt_tokens,
            decode_tokens: mixed.decode_tokens,
        },
        mixed.elapsed,
    );
    assert_eq!(estimator.estimated_prefill_ms(64), None);

    iteration(
        runtime,
        session,
        &prompt,
        IterationBatchPhase::Prefill,
        false,
    )?;
    assert!(runtime.take_completed_execution().is_none());
    let before_export = runtime.model.execution_epoch().expect("native context");
    assert_ne!(before_export & 1, 0);
    let state = runtime.export_full_state(session)?;
    assert!(!state.is_empty());
    let after_export = runtime.model.execution_epoch().expect("native context");
    assert_ne!(before_export, after_export);
    assert_eq!(after_export & 1, 0);
    iteration(runtime, session, &[3], IterationBatchPhase::Decode, true)?;
    let decode = runtime
        .take_completed_execution()
        .expect("independent decode");
    assert_eq!((decode.prompt_tokens, decode.decode_tokens), (0, 1));
    Ok(())
}
