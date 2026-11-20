use super::super::tests::direct_iteration;
use super::*;
use skippy_runtime::{ActivationDesc, IterationSample};

#[test]
fn canonical_batch_order_keeps_inputs_and_native_outputs_with_their_reply_owners() {
    let mut queue = VecDeque::new();
    let mut replies = Vec::new();
    for (session, marker) in [("c", 3), ("a", 1), ("b", 2)] {
        let (reply, result) = std_mpsc::sync_channel(1);
        let mut request = direct_iteration(session, 1);
        request.token_ids = vec![marker];
        request.positions = vec![marker + 10];
        request.selected_tokens = Some(vec![marker + 100]);
        request.input = Some(frame(u8::try_from(marker).unwrap()));
        request.reply = reply.into();
        queue.push_back(request);
        replies.push((marker, result));
    }

    let batch = take_direct_iteration_batch(&mut queue, 3, 3);
    assert_eq!(
        batch
            .iter()
            .map(|r| r.session_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    for (request, marker) in batch.iter().zip(1..=3) {
        assert_eq!(request.token_ids, [marker]);
        assert_eq!(request.positions, [marker + 10]);
        assert_eq!(
            request.selected_tokens.as_deref(),
            Some([marker + 100].as_slice())
        );
        assert_eq!(
            request.input.as_ref().unwrap().payload,
            [u8::try_from(marker).unwrap()]
        );
    }
    let output = IterationBatchOutput {
        request_outputs: batch.iter().map(|r| r.input.clone().unwrap()).collect(),
        // Native samples need not arrive in request order.
        samples: vec![sample(2, 3), sample(0, 1), sample(1, 2)],
    };
    let timing = BatchTiming {
        wait: vec![1.0, 2.0, 3.0],
        lock_wait_ms: 4.0,
        lock_hold_ms: 5.0,
        kv_used_cells: None,
    };
    send_outputs(
        batch.into_iter().map(|r| (r, None)).collect(),
        output,
        vec![vec![1.0], vec![2.0], vec![3.0]],
        timing,
    );

    for (marker, result) in replies {
        let outcome = result.try_recv().unwrap().unwrap();
        assert_eq!(outcome.predicted, marker);
        assert_eq!(outcome.output.payload, [u8::try_from(marker).unwrap()]);
        assert_eq!(
            outcome.log_probabilities,
            [f32::from(u8::try_from(marker).unwrap())]
        );
        assert_eq!(outcome.batch_wait_ms, f64::from(marker));
        assert_eq!(outcome.batch_size, 3);
        assert_eq!(outcome.runtime_lock_wait_ms, 4.0);
        assert_eq!(outcome.runtime_lock_hold_ms, 5.0);
    }
}

fn sample(request_index: usize, predicted_token: i32) -> IterationSample {
    IterationSample {
        request_index,
        predicted_token,
    }
}

fn frame(marker: u8) -> ActivationFrame {
    ActivationFrame {
        desc: ActivationDesc {
            version: 0,
            producer_stage_index: 0,
            layer_start: 0,
            layer_end: 0,
            token_count: 1,
            sequence_count: 1,
            part_count: 0,
            payload_bytes: 1,
            frontier_identity: Default::default(),
            parts: Default::default(),
        },
        payload: vec![marker],
    }
}
