//! Compare decode shapes from identical native checkpoints, without scheduling
//! or prefix-cache lookup. Keep exact token assertions.
use super::*;
use crate::runtime_state::RuntimeIterationBatchRequest;
use anyhow::Context;
use skippy_runtime::IterationBatchPhase;

pub(super) fn prefill_without_intermediate_sampling(
    runtime: &mut RuntimeState,
    session: &str,
    tokens: &[i32],
    sample_last: bool,
) -> Result<Option<i32>> {
    let sampling = SamplingConfig::default();
    let output = runtime.iteration_batch_sampled(&[RuntimeIterationBatchRequest {
        session_id: session,
        token_ids: tokens,
        positions: &[],
        sampling: sample_last.then_some(&sampling),
        input: None,
        sample_last,
        phase: IterationBatchPhase::Prefill,
    }])?;
    Ok(output.samples.first().map(|sample| sample.predicted_token))
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_hybrid_iteration_batch_shape_parity() -> Result<()> {
    let (wide, _) =
        recurrent_test_backend("iteration-wide", "iteration-wide", 512, (128, 32), 8, 256)?;
    let (narrow, _) = recurrent_test_backend(
        "iteration-narrow",
        "iteration-narrow",
        512,
        (32, 32),
        8,
        256,
    )?;
    let prompt = (1..=128).collect::<Vec<_>>();
    let mut wide_runtime = wide.runtime.lock().unwrap();
    let mut narrow_runtime = narrow.runtime.lock().unwrap();
    let mut wide_token = None;
    let mut narrow_token = None;
    for (index, chunk) in prompt.chunks(32).enumerate() {
        let sample_last = index == 3;
        wide_token = prefill_without_intermediate_sampling(
            &mut wide_runtime,
            "native-parity",
            chunk,
            sample_last,
        )?;
        narrow_token = prefill_without_intermediate_sampling(
            &mut narrow_runtime,
            "native-parity",
            chunk,
            sample_last,
        )?;
    }
    let mut wide_token = wide_token.context("wide prefill produced no sample")?;
    let mut narrow_token = narrow_token.context("narrow prefill produced no sample")?;
    assert_eq!(
        wide_token, narrow_token,
        "native prefill sample differs across batch sizes"
    );
    for index in 1..8 {
        wide_token = iterate(
            &mut wide_runtime,
            &["native-parity"],
            &[&[wide_token]],
            IterationBatchPhase::Decode,
        )?[0];
        narrow_token = iterate(
            &mut narrow_runtime,
            &["native-parity"],
            &[&[narrow_token]],
            IterationBatchPhase::Decode,
        )?[0];
        assert_eq!(
            wide_token, narrow_token,
            "native decode token {index} differs across batch sizes"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_interleaved_prefills_preserve_concurrent_decode() -> Result<()> {
    let serial = interleaved_prefill_continuation(false, false, None)?;
    let interleaved = interleaved_prefill_continuation(true, false, None)?;
    let serial_batched = interleaved_prefill_continuation(false, true, None)?;
    let interleaved_batched = interleaved_prefill_continuation(true, true, None)?;
    assert_eq!(
        interleaved.0, serial.0,
        "prefill order changed serial decode"
    );
    assert_eq!(
        serial_batched.0[0],
        serial.0[0][..3],
        "batching changed lane 0"
    );
    assert_eq!(serial_batched.0[1], serial.0[1], "batching changed lane 1");
    assert_eq!(
        interleaved_batched.0, serial_batched.0,
        "interleaved prefills changed batched decode"
    );
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_interleaved_prefills_preserve_matched_decode_after_near_tie() -> Result<()> {
    // With quantized CUDA, physical KV placement can change reduction order
    // enough to swap a near-tied top two. Feed the same continuation after the
    // first step so one swapped sample does not turn into an unrelated trace.
    let baseline = interleaved_prefill_continuation(false, false, None)?;
    let fixed = Some([baseline.0[0][1], baseline.0[1][1]]);
    let serial = interleaved_prefill_continuation(false, false, fixed)?;
    let interleaved = interleaved_prefill_continuation(true, false, fixed)?;
    let serial_batched = interleaved_prefill_continuation(false, true, fixed)?;
    let interleaved_batched = interleaved_prefill_continuation(true, true, fixed)?;

    assert_eq!(
        interleaved.0, serial.0,
        "prefill order changed serial decode"
    );
    assert_eq!(
        serial_batched.0[0],
        serial.0[0][..3],
        "batching changed lane 0"
    );
    assert_eq!(serial_batched.0[1], serial.0[1], "batching changed lane 1");
    assert_eq!(interleaved_batched.0[0], serial_batched.0[0]);
    assert_eq!(
        interleaved_batched.0[1][2..],
        serial_batched.0[1][2..],
        "matched continuation changed lane 1 after the first shared decode"
    );
    assert_near_tie_or_match(
        serial_batched.0[1][1],
        interleaved_batched.0[1][1],
        serial_batched
            .1
            .context("serial batched first signal missing")?,
        interleaved_batched
            .1
            .context("interleaved batched first signal missing")?,
    );
    Ok(())
}

fn assert_near_tie_or_match(
    expected: i32,
    actual: i32,
    serial: skippy_runtime::TokenSignal,
    interleaved: skippy_runtime::TokenSignal,
) {
    if expected == actual {
        return;
    }
    assert_eq!((serial.top_token, serial.second_token), (expected, actual));
    assert_eq!(
        (interleaved.top_token, interleaved.second_token),
        (actual, expected)
    );
    assert!(
        serial.margin <= 0.05 && interleaved.margin <= 0.05,
        "first decode was not a near tie: serial={serial:?}, interleaved={interleaved:?}"
    );
    assert!(
        (serial.top_logprob - interleaved.top_logprob).abs() <= 0.05
            && (serial.entropy - interleaved.entropy).abs() <= 0.05,
        "first decode distributions changed beyond near-tie tolerance: serial={serial:?}, interleaved={interleaved:?}"
    );
}

fn interleaved_prefill_continuation(
    interleaved: bool,
    batched_decode: bool,
    fixed_after_first: Option<[i32; 2]>,
) -> Result<(Vec<Vec<i32>>, Option<skippy_runtime::TokenSignal>)> {
    let prompts = [
        (1..=64).collect::<Vec<_>>(),
        (1..=64).rev().collect::<Vec<_>>(),
    ];
    let (backend, _) = recurrent_test_backend_with_lanes(
        "interleaved-prefill-shape",
        "interleaved-prefill-shape",
        256,
        (32, 32, 2),
        8,
        512,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let sessions = ["interleaved-prefill-0", "interleaved-prefill-1"];
    let mut first = [0; 2];
    let chunks = if interleaved {
        [
            (0, 0, 32),
            (0, 32, 30),
            (1, 0, 32),
            (0, 62, 2),
            (1, 32, 30),
            (1, 62, 2),
        ]
    } else {
        [
            (0, 0, 32),
            (0, 32, 30),
            (0, 62, 2),
            (1, 0, 32),
            (1, 32, 30),
            (1, 62, 2),
        ]
    };
    for (lane, offset, count) in chunks {
        if let Some(token) = prefill_without_intermediate_sampling(
            &mut runtime,
            sessions[lane],
            &prompts[lane][offset..offset + count],
            offset == 62,
        )? {
            first[lane] = token;
        }
    }
    if batched_decode {
        let decoded = long_decode(&mut runtime, &sessions, &first, true, fixed_after_first)?;
        return Ok((decoded.tokens, Some(decoded.first_signal)));
    }
    let mut outputs = first.map(|token| vec![token]);
    let mut first_signal = None;
    for lane in 0..2 {
        for _ in 1..8 {
            let input = [if outputs[lane].len() == 1 {
                *outputs[lane].last().unwrap()
            } else {
                fixed_after_first.map_or(*outputs[lane].last().unwrap(), |fixed| fixed[lane])
            }];
            outputs[lane].extend(iterate(
                &mut runtime,
                &[sessions[lane]],
                &[&input],
                IterationBatchPhase::Decode,
            )?);
            if lane == 1 && outputs[lane].len() == 2 {
                first_signal = Some(runtime.last_token_signal(sessions[1])?);
            }
        }
    }
    Ok((outputs.into_iter().collect(), first_signal))
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_prefill_boundaries_preserve_decode_tokens() -> Result<()> {
    // No snapshot restore, cache lookup, HTTP, or concurrent scheduler is used.
    // Every arm starts in a fresh context at physical KV position zero. Only
    // the final prefill chunk boundaries differ, as they can on cache replay.
    let expected = continuation_after_prefill_chunks(&[32, 32])?;
    for chunks in [&[32, 28, 4][..], &[32, 30, 2][..], &[32, 31, 1][..]] {
        let actual = continuation_after_prefill_chunks(chunks)?;
        assert_eq!(
            actual, expected,
            "fresh native contexts; prefill chunks {chunks:?} versus [32, 32]"
        );
    }
    let expected_extended = continuation_after_prefill_chunks(&[32, 32, 4])?;
    for chunks in [&[32, 32, 2, 2][..], &[32, 30, 2, 2, 2][..]] {
        let actual = continuation_after_prefill_chunks(chunks)?;
        assert_eq!(
            actual, expected_extended,
            "fresh native contexts; prefill chunks {chunks:?} versus [32, 32, 4]"
        );
    }
    Ok(())
}

fn continuation_after_prefill_chunks(chunks: &[usize]) -> Result<Vec<i32>> {
    let token_count = chunks.iter().sum::<usize>();
    assert!(token_count == 64 || token_count == 68);
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
    let prompt: Vec<i32> = (1..=token_count as i32).collect();
    let mut offset = 0;
    let mut predicted = None;
    for &count in chunks {
        let output = prefill_without_intermediate_sampling(
            &mut runtime,
            "prefill-shape",
            &prompt[offset..offset + count],
            offset + count == prompt.len(),
        )?;
        if let Some(token) = output {
            predicted = Some(token);
        }
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

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_intermediate_sampling_preserves_prefill_state() -> Result<()> {
    let sampled = continuation_after_intermediate_sampling(true)?;
    let unsampled = continuation_after_intermediate_sampling(false)?;
    assert_eq!(
        sampled, unsampled,
        "sampling at a 64-token prefill boundary changed the 68-token continuation"
    );
    Ok(())
}

fn continuation_after_intermediate_sampling(sample_at_64: bool) -> Result<Vec<i32>> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "intermediate-sampling",
        "intermediate-sampling",
        256,
        (32, 32, 2),
        8,
        512,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let prompt: Vec<i32> = (1..=68).collect();
    let mut first = None;
    for (start, end, sample_last) in [
        (0, 32, false),
        (32, 62, false),
        (62, 64, sample_at_64),
        (64, 66, false),
        (66, 68, true),
    ] {
        if let Some(token) = prefill_without_intermediate_sampling(
            &mut runtime,
            "intermediate-sampling",
            &prompt[start..end],
            sample_last,
        )? {
            first = Some(token);
        }
    }
    let mut output = vec![first.expect("final chunk sampled")];
    for _ in 0..2 {
        let input = [*output.last().unwrap()];
        output.extend(iterate(
            &mut runtime,
            &["intermediate-sampling"],
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
fn native_followup_prefill_boundaries_preserve_tokens() -> Result<()> {
    let prompt = followup_prompt()?;
    let expected = native_followup(&prompt, &[32, 32], FollowupLayout::SingleLane)?;
    for (chunks, layout) in [
        (&[32, 32][..], FollowupLayout::ReservedLanes),
        (&[32, 28, 4][..], FollowupLayout::ReservedLanes),
        (&[32, 31, 1][..], FollowupLayout::ReservedLanes),
    ] {
        assert_eq!(
            native_followup(&prompt, chunks, layout)?,
            expected,
            "native follow-up: layout={layout:?}, prefix chunks={chunks:?}, prompt={prompt:?}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_followup_cache_and_lane_layouts_preserve_tokens() -> Result<()> {
    // Keep chunk boundaries fixed so a numerical shape mismatch cannot prevent
    // the occupied-lane and snapshot-restore controls from running.
    let prompt = followup_prompt()?;
    let expected = native_followup(&prompt, &[32, 32], FollowupLayout::SingleLane)?;
    for layout in [
        FollowupLayout::ReservedLanes,
        FollowupLayout::OccupiedLane,
        FollowupLayout::RestoredPrefix,
    ] {
        assert_eq!(
            native_followup(&prompt, &[32, 32], layout)?,
            expected,
            "native follow-up with fixed prefill chunks: layout={layout:?}"
        );
    }
    Ok(())
}

fn followup_prompt() -> Result<Vec<i32>> {
    let mut prompt: Vec<i32> = (1..=64).collect();
    prompt.extend(continuation_after_prefill_chunks(&[32, 32])?);
    prompt.push(7);
    Ok(prompt)
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
        prefill_without_intermediate_sampling(
            &mut runtime,
            "followup",
            &prompt[offset..offset + count],
            false,
        )?;
        offset += count;
    }
    assert_eq!(offset, 64);
    if matches!(
        layout,
        FollowupLayout::OccupiedLane | FollowupLayout::RestoredPrefix
    ) {
        for chunk in (1..=64).rev().collect::<Vec<_>>().chunks(32) {
            prefill_without_intermediate_sampling(&mut runtime, "other", chunk, false)?;
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

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_long_decode_batches_preserve_both_continuations() -> Result<()> {
    compare_long_decode_batches(32)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_single_token_microbatches_preserve_both_continuations() -> Result<()> {
    compare_long_decode_batches(1)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_shortened_tail_decode_batches_preserve_continuations() -> Result<()> {
    let prompts = [0, 1].map(|lane| {
        let mut prompt: Vec<_> = (1..=176).map(|token| token + 257 * lane).collect();
        prompt.extend_from_slice(&[501, 502, 503, 504]);
        prompt
    });
    compare_long_decode_prompts(32, prompts)
}

#[test]
#[ignore = "requires a full-attention hybrid SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_recurrent_cache_payload_excludes_attention_growth() -> Result<()> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "cache-payload",
        "cache-payload",
        256,
        (32, 32, 2),
        8,
        512,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let mut sizes = Vec::new();
    for _ in 0..2 {
        for _ in 0..2 {
            iterate(
                &mut runtime,
                &["payload"],
                &[&[1; 32]],
                IterationBatchPhase::Prefill,
            )?;
        }
        sizes.push((
            runtime.export_recurrent_cache_state("payload")?.len(),
            runtime.export_full_state("payload")?.len(),
        ));
    }
    assert!(sizes[0].0 > 0, "the fixture must contain recurrent state");
    assert_eq!(
        sizes[0].0, sizes[1].0,
        "resident side state must not grow with full-attention KV"
    );
    assert!(
        sizes[0].1 < sizes[1].1,
        "the full checkpoint must retain growing attention KV"
    );
    assert!(sizes[1].0 < sizes[1].1);
    Ok(())
}

fn compare_long_decode_batches(microbatch: u32) -> Result<()> {
    compare_long_decode_prompts(
        microbatch,
        [(1..=64).collect::<Vec<_>>(), (1..=64).rev().collect()],
    )
}

fn compare_long_decode_prompts(microbatch: u32, prompts: [Vec<i32>; 2]) -> Result<()> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "long-decode-shapes",
        "long-decode-shapes",
        256,
        (32, microbatch, 2),
        8,
        512,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let sessions = ["long-shape-0", "long-shape-1"];
    let mut starts = Vec::new();
    let mut checkpoints = Vec::new();
    for (session, prompt) in sessions.iter().zip(&prompts) {
        let mut predicted = None;
        for chunk in prompt.chunks(32) {
            predicted = Some(
                iterate(
                    &mut runtime,
                    &[*session],
                    &[chunk],
                    IterationBatchPhase::Prefill,
                )?[0],
            );
        }
        starts.push(predicted.expect("nonempty prompt"));
        checkpoints.push(runtime.export_full_state(session)?);
    }
    // Match the serving regression: the first lane stops at three tokens while
    // the second continues to eight. Comparing only the first decode step
    // misses accumulated recurrent-state differences near the end of a turn.
    let together = long_decode(&mut runtime, &sessions, &starts, true, None)?;
    restore(&mut runtime, &sessions, &prompts, &checkpoints)?;
    let replay = long_decode(&mut runtime, &sessions, &starts, true, None)?;
    assert_eq!(
        together.tokens, replay.tokens,
        "same-shape checkpoint replay, ubatch={microbatch}"
    );
    restore(&mut runtime, &sessions, &prompts, &checkpoints)?;
    let separate = long_decode(&mut runtime, &sessions, &starts, false, None)?;
    assert_eq!(
        together.tokens,
        separate.tokens,
        "concurrent versus serial decode, ubatch={microbatch}; first decode signals: together={:?}, separate={:?}; terminal signals: together={:?}, separate={:?}; recurrent byte differences={}",
        together.first_signal,
        separate.first_signal,
        together.terminal_signal,
        separate.terminal_signal,
        together
            .recurrent
            .iter()
            .zip(&separate.recurrent)
            .filter(|(a, b)| a != b)
            .count()
    );
    Ok(())
}

struct LongDecode {
    tokens: Vec<Vec<i32>>,
    first_signal: skippy_runtime::TokenSignal,
    terminal_signal: skippy_runtime::TokenSignal,
    recurrent: Vec<u8>,
}

fn long_decode(
    runtime: &mut RuntimeState,
    sessions: &[&str; 2],
    starts: &[i32],
    together: bool,
    fixed_after_first: Option<[i32; 2]>,
) -> Result<LongDecode> {
    let mut outputs: Vec<_> = starts.iter().map(|&token| vec![token]).collect();
    let mut first_signal = None;
    for step in 1..8 {
        let active: &[usize] = if step < 3 { &[0, 1] } else { &[1] };
        let names: Vec<_> = active.iter().map(|&index| sessions[index]).collect();
        let inputs: Vec<_> = active
            .iter()
            .map(|&index| {
                [if step == 1 {
                    *outputs[index].last().unwrap()
                } else {
                    fixed_after_first.map_or(*outputs[index].last().unwrap(), |fixed| fixed[index])
                }]
            })
            .collect();
        let tokens: Vec<_> = inputs.iter().map(|input| input.as_slice()).collect();
        let next = if together {
            iterate(runtime, &names, &tokens, IterationBatchPhase::Decode)?
        } else {
            let mut next = Vec::new();
            for (name, token) in names.iter().zip(&tokens) {
                next.extend(iterate(
                    runtime,
                    &[*name],
                    &[*token],
                    IterationBatchPhase::Decode,
                )?);
            }
            next
        };
        for (&index, token) in active.iter().zip(next) {
            outputs[index].push(token);
        }
        if step == 1 {
            let signal = runtime.last_token_signal(sessions[1])?;
            assert_eq!(signal.top_token, *outputs[1].last().unwrap());
            first_signal = Some(signal);
        }
    }
    // At the terminal step only lane 1 has run; its signal refers to the last
    // native logit row. Do not read earlier lanes through the last-row API.
    let terminal_signal = runtime.last_token_signal(sessions[1])?;
    assert_eq!(terminal_signal.top_token, *outputs[1].last().unwrap());
    Ok(LongDecode {
        tokens: outputs,
        first_signal: first_signal.expect("decode includes a first step"),
        terminal_signal,
        recurrent: runtime.export_recurrent_cache_state(sessions[1])?,
    })
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

pub(super) fn iterate(
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
