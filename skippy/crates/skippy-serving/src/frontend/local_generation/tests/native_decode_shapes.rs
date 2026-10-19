//! Compare decode shapes from identical native checkpoints, without scheduling
//! or prefix-cache lookup. Keep exact token assertions.
use super::*;
use crate::runtime_state::RuntimeIterationBatchRequest;
use skippy_runtime::IterationBatchPhase;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_checkpoint_decode_matches_across_batch_shapes() -> Result<()> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "decode-shapes",
        "decode-shapes",
        256,
        (32, 32, 2),
        8,
        512,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let sessions = ["shape-0", "shape-1"];
    let prompts = [(1..=64).collect::<Vec<_>>(), (1..=64).rev().collect()];
    let mut inputs = Vec::new();
    let mut checkpoints = Vec::new();
    for (session, prompt) in sessions.iter().zip(&prompts) {
        let mut predicted = None;
        for chunk in prompt.chunks(32) {
            let output = iterate(
                &mut runtime,
                &[*session],
                &[chunk],
                IterationBatchPhase::Prefill,
            )?;
            predicted = Some(output[0]);
        }
        inputs.push(vec![predicted.expect("nonempty prompt")]);
        checkpoints.push(runtime.export_full_state(session)?);
    }
    let tokens: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
    let batched = iterate(
        &mut runtime,
        &sessions,
        &tokens,
        IterationBatchPhase::Decode,
    )?;
    restore(&mut runtime, &sessions, &prompts, &checkpoints)?;
    let replay = iterate(
        &mut runtime,
        &sessions,
        &tokens,
        IterationBatchPhase::Decode,
    )?;
    assert_eq!(
        batched, replay,
        "restoring identical states must preserve same-shape decode tokens"
    );
    restore(&mut runtime, &sessions, &prompts, &checkpoints)?;
    let mut separate = Vec::new();
    for (session, token) in sessions.iter().zip(&tokens) {
        separate.extend(iterate(
            &mut runtime,
            &[*session],
            &[*token],
            IterationBatchPhase::Decode,
        )?);
    }
    assert_eq!(
        batched, separate,
        "identical native checkpoints, only decode shape changed"
    );
    Ok(())
}

fn restore(
    runtime: &mut RuntimeState,
    sessions: &[&str],
    prompts: &[Vec<i32>],
    checkpoints: &[Vec<u8>],
) -> Result<()> {
    for ((session, prompt), checkpoint) in sessions.iter().zip(prompts).zip(checkpoints) {
        runtime.import_full_state_for_token_count(session, checkpoint, prompt.len() as u64)?;
        runtime.restore_token_history(session, prompt)?;
    }
    Ok(())
}

fn iterate(
    runtime: &mut RuntimeState,
    sessions: &[&str],
    tokens: &[&[i32]],
    phase: IterationBatchPhase,
) -> Result<Vec<i32>> {
    let sampling = SamplingConfig::default();
    let requests: Vec<_> = sessions
        .iter()
        .zip(tokens)
        .map(|(session, tokens)| RuntimeIterationBatchRequest {
            session_id: session,
            token_ids: tokens,
            positions: &[],
            sampling: Some(&sampling),
            input: None,
            sample_last: true,
            phase,
        })
        .collect();
    let output = runtime.iteration_batch_sampled(&requests)?;
    assert_eq!(output.samples.len(), sessions.len());
    let mut predictions = Vec::new();
    for index in 0..sessions.len() {
        let sample = output
            .samples
            .iter()
            .find(|sample| sample.request_index == index)
            .expect("each request must have a sample");
        predictions.push(sample.predicted_token);
    }
    Ok(predictions)
}
