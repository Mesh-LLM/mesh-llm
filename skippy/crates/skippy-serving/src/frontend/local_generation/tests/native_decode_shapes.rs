//! Compare decode shapes from identical native checkpoints, without scheduling
//! or prefix-cache lookup. Keep exact token assertions.
use super::*;
use crate::runtime_state::RuntimeIterationBatchRequest;
use skippy_runtime::IterationBatchPhase;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_prefill_boundaries_preserve_decode_tokens() -> Result<()> {
    // No snapshot restore, cache lookup, HTTP, or concurrent scheduler is used.
    // Every arm starts in a fresh context at physical KV position zero. Only
    // the final prefill chunk boundaries differ, as they can on cache replay.
    let expected = continuation_after_prefill_chunks(&[32, 32])?;
    for chunks in [&[32, 28, 4][..], &[32, 31, 1][..]] {
        let actual = continuation_after_prefill_chunks(chunks)?;
        assert_eq!(
            actual, expected,
            "fresh native contexts; prefill chunks {chunks:?} versus [32, 32]"
        );
    }
    Ok(())
}

fn continuation_after_prefill_chunks(chunks: &[usize]) -> Result<Vec<i32>> {
    assert_eq!(chunks.iter().sum::<usize>(), 64);
    let (backend, _) = recurrent_test_backend_with_lanes(
        "prefill-shapes",
        "prefill-shapes",
        256,
        (32, 32, 2),
        8,
        512,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let prompt: Vec<i32> = (1..=64).collect();
    let mut offset = 0;
    let mut predicted = None;
    for &count in chunks {
        let output = iterate(
            &mut runtime,
            &["prefill-shape"],
            &[&prompt[offset..offset + count]],
            IterationBatchPhase::Prefill,
        )?;
        predicted = Some(output[0]);
        offset += count;
    }
    let mut output = vec![predicted.expect("nonempty prefill")];
    for _ in 0..2 {
        let input = [*output.last().unwrap()];
        output.extend(iterate(
            &mut runtime,
            &["prefill-shape"],
            &[&input],
            IterationBatchPhase::Decode,
        )?);
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
enum FollowupLayout {
    SingleLane,
    ReservedLanes,
    OccupiedLane,
    RestoredPrefix,
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_followup_matches_across_lane_and_restore_shapes() -> Result<()> {
    let mut prompt: Vec<i32> = (1..=64).collect();
    prompt.extend(continuation_after_prefill_chunks(&[32, 32])?);
    prompt.push(7);
    let expected = native_followup(&prompt, &[32, 32], FollowupLayout::SingleLane)?;
    for (chunks, layout) in [
        (&[32, 32][..], FollowupLayout::ReservedLanes),
        (&[32, 28, 4][..], FollowupLayout::ReservedLanes),
        (&[32, 31, 1][..], FollowupLayout::ReservedLanes),
        (&[32, 32][..], FollowupLayout::OccupiedLane),
        (&[32, 32][..], FollowupLayout::RestoredPrefix),
    ] {
        assert_eq!(
            native_followup(&prompt, chunks, layout)?,
            expected,
            "native follow-up: layout={layout:?}, prefix chunks={chunks:?}, prompt={prompt:?}"
        );
    }
    Ok(())
}

fn native_followup(
    prompt: &[i32],
    prefix_chunks: &[usize],
    layout: FollowupLayout,
) -> Result<Vec<i32>> {
    let lanes = if matches!(layout, FollowupLayout::SingleLane) {
        1
    } else {
        2
    };
    let (backend, _) = recurrent_test_backend_with_lanes(
        "followup-shapes",
        "followup-shapes",
        256,
        (32, 32, lanes),
        8,
        lanes as usize * 256,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let mut offset = 0;
    for &count in prefix_chunks {
        iterate(
            &mut runtime,
            &["followup"],
            &[&prompt[offset..offset + count]],
            IterationBatchPhase::Prefill,
        )?;
        offset += count;
    }
    assert_eq!(offset, 64);
    if matches!(
        layout,
        FollowupLayout::OccupiedLane | FollowupLayout::RestoredPrefix
    ) {
        for chunk in (1..=64).rev().collect::<Vec<_>>().chunks(32) {
            iterate(
                &mut runtime,
                &["other"],
                &[chunk],
                IterationBatchPhase::Prefill,
            )?;
        }
    }
    if matches!(layout, FollowupLayout::RestoredPrefix) {
        let checkpoint = runtime.export_full_state("followup")?;
        for token in &prompt[64..67] {
            iterate(
                &mut runtime,
                &["followup"],
                &[&[*token]],
                IterationBatchPhase::Decode,
            )?;
        }
        restore(
            &mut runtime,
            &["followup"],
            &[prompt[..64].to_vec()],
            &[checkpoint],
        )?;
    }
    let mut output = iterate(
        &mut runtime,
        &["followup"],
        &[&prompt[64..]],
        IterationBatchPhase::Prefill,
    )?;
    for _ in 1..8 {
        let token = [*output.last().unwrap()];
        output.extend(iterate(
            &mut runtime,
            &["followup"],
            &[&token],
            IterationBatchPhase::Decode,
        )?);
    }
    Ok(output)
}

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
