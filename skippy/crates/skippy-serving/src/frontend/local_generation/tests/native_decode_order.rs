//! Verify that changing native batch row order does not change a live hybrid continuation.
use super::native_decode_shapes::{iterate, prefill_without_intermediate_sampling};
use super::*;
use crate::frontend::iteration_scheduler::ScheduledResumeRequest;
use skippy_runtime::IterationBatchPhase;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_four_lane_decode_preserves_state_across_batch_reordering() -> Result<()> {
    let fixed = decode_with_order_changes(false)?;
    let reordered = decode_with_order_changes(true)?;
    for lane in 0..4 {
        assert_eq!(
            reordered[lane], fixed[lane],
            "native c=4 continuation changed after reordering the batch: lane={lane}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn four_lane_planned_handoff_matches_direct_decode() -> Result<()> {
    let direct = decode_with_order_changes(false)?;
    let planned = decode_with_scheduler_handoff()?;
    for lane in 0..4 {
        assert_eq!(
            planned[lane], direct[lane],
            "planned c=4 continuation changed after direct-to-planned handoff: lane={lane}"
        );
    }
    Ok(())
}

fn decode_with_order_changes(reorder: bool) -> Result<[Vec<i32>; 4]> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "batch-order",
        "batch-order",
        1024,
        (128, 32, 4),
        8,
        4096,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let sessions = [
        "batch-order-0",
        "batch-order-1",
        "batch-order-2",
        "batch-order-3",
    ];
    let mut output: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::new());
    for lane in 0..4 {
        let prompt: Vec<_> = (1..=256).map(|token| token + lane as i32 * 257).collect();
        for (index, chunk) in prompt.chunks(32).enumerate() {
            if let Some(token) = prefill_without_intermediate_sampling(
                &mut runtime,
                sessions[lane],
                chunk,
                index == 7,
            )? {
                output[lane].push(token);
            }
        }
        assert_eq!(output[lane].len(), 1);
    }

    for step in 1..8 {
        let order = if reorder && step % 2 == 0 {
            [3, 1, 0, 2]
        } else {
            [0, 1, 2, 3]
        };
        let names: Vec<_> = order.iter().map(|&lane| sessions[lane]).collect();
        let inputs: Vec<_> = order
            .iter()
            .map(|&lane| [*output[lane].last().unwrap()])
            .collect();
        let tokens: Vec<_> = inputs.iter().map(|input| input.as_slice()).collect();
        let next = iterate(&mut runtime, &names, &tokens, IterationBatchPhase::Decode)?;
        for (lane, token) in order.into_iter().zip(next) {
            output[lane].push(token);
        }
    }
    Ok(output)
}

fn decode_with_scheduler_handoff() -> Result<[Vec<i32>; 4]> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "batch-handoff",
        "batch-handoff",
        1024,
        (128, 32, 4),
        8,
        4096,
        None,
    )?;
    let sessions = [
        "batch-handoff-0",
        "batch-handoff-1",
        "batch-handoff-2",
        "batch-handoff-3",
    ];
    let prompts: [Vec<i32>; 4] =
        std::array::from_fn(|lane| (1..=256).map(|token| token + lane as i32 * 257).collect());
    let mut output: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::new());
    {
        let mut runtime = backend.runtime.lock().unwrap();
        for lane in 0..4 {
            for (index, chunk) in prompts[lane].chunks(32).enumerate() {
                if let Some(token) = prefill_without_intermediate_sampling(
                    &mut runtime,
                    sessions[lane],
                    chunk,
                    index == 7,
                )? {
                    output[lane].push(token);
                }
            }
        }
        let inputs: Vec<_> = output.iter().map(|tokens| [tokens[0]]).collect();
        let tokens: Vec<_> = inputs.iter().map(|input| input.as_slice()).collect();
        let next = iterate(
            &mut runtime,
            &sessions,
            &tokens,
            IterationBatchPhase::Decode,
        )?;
        for (lane, token) in next.into_iter().enumerate() {
            output[lane].push(token);
        }
    }
    let resumed = std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|lane| {
                let mut generated = output[lane].clone();
                let sampling = SamplingConfig::default();
                let scheduler = &backend.iteration_scheduler;
                let prompt = &prompts[lane];
                scope.spawn(move || -> Result<Vec<i32>> {
                    scheduler.resume_generation(
                        ScheduledResumeRequest {
                            id: sessions[lane],
                            prompt_tokens: prompt,
                            generated_tokens: &generated.clone(),
                            max_tokens: 8,
                            sampling: Some(&sampling),
                            cancellation: None,
                        },
                        |token| {
                            generated.push(token);
                            Ok(TokenControl::Continue)
                        },
                    )?;
                    Ok(generated)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("planned decoder thread panicked"))
            .collect::<Result<Vec<_>>>()
    })?;
    output = resumed.try_into().expect("four resumed lanes");
    Ok(output)
}
