//! Real scheduler progress, exact checkpoint capture, and cancellation cleanup.
use super::recurrent_fixture::recurrent_test_backend_with_lanes;
use super::*;
use std::sync::mpsc;
use std::time::Instant;

struct PendingPrefill {
    cancellation: skippy_inference_api::CancellationToken,
    worker: Option<
        thread::JoinHandle<
            skippy_inference_api::InferenceResult<
                crate::frontend::generation::GenerationCacheStats,
            >,
        >,
    >,
}

impl Drop for PendingPrefill {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn resident_checkpoint_prefill_yields_to_decode_and_cancels_without_losing_checkpoint() -> Result<()>
{
    let (backend, speculative) = recurrent_test_backend_with_lanes(
        "cooperative-prefill",
        "cooperative-prefill",
        8192,
        32,
        1,
        16384,
        2,
    )?;
    let kv = backend.kv.as_ref().unwrap();
    assert!(
        kv.l3.is_none(),
        "this test must exercise disk-off capability admission"
    );
    assert!(kv.scheduling_policy().cooperative());
    backend.iteration_scheduler.execute_iteration(
        "concurrent-decode",
        &[11, 12, 13],
        &[],
        None,
        false,
        skippy_runtime::IterationBatchPhase::Prefill,
    )?;
    let prompt = [1, 2, 3, 4, 5, 6, 7, 8].repeat(512);
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("cooperative-long"),
        true,
        None,
    );
    let session = ids.session_label.clone();
    let cancel = skippy_inference_api::CancellationToken::new();
    let worker_cancel = cancel.clone();
    let worker_backend = backend.clone();
    let worker_prompt = prompt.clone();
    let (finished_tx, finished_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let sampling = SamplingConfig {
            enabled: false,
            ..Default::default()
        };
        let result = worker_backend.generate_local_tokens(
            LocalGeneration {
                prompt_token_ids: &worker_prompt,
                recurrent_cache_prefix_token_ids: Some(&worker_prompt[..64]),
                max_tokens: 1,
                sampling: &sampling,
                chat_sampling_metadata: None,
                speculative: &speculative,
                native_mtp_enabled: false,
                hook_request: None,
                hook_runtime: None,
                cancellation: Some(&worker_cancel),
                ids: &ids,
            },
            |_| Ok(TokenControl::Continue),
        );
        let _ = finished_tx.send(());
        result
    });
    let mut pending = PendingPrefill {
        cancellation: cancel,
        worker: Some(worker),
    };
    // Observe a real intermediate native position through the same scheduler.
    // A serialized whole-prefill closure cannot satisfy this condition.
    let deadline = Instant::now() + Duration::from_secs(30);
    let observed = loop {
        let session = session.clone();
        let position = backend
            .iteration_scheduler
            .execute_runtime("test-prefill-progress", move |rt| {
                Ok(rt.session_token_count(&session).unwrap_or_default())
            })?;
        if position >= 128 && position < prompt.len() as u64 {
            break true;
        }
        if finished_rx.try_recv().is_ok() || Instant::now() >= deadline {
            break false;
        }
        thread::yield_now();
    };
    let decode = if observed {
        Some(backend.iteration_scheduler.execute_iteration(
            "concurrent-decode",
            &[14],
            &[],
            None,
            true,
            skippy_runtime::IterationBatchPhase::Decode,
        ))
    } else {
        None
    };
    let unfinished_at_decode = !pending.worker.as_ref().unwrap().is_finished();
    pending.cancellation.cancel();
    let result = pending
        .worker
        .take()
        .unwrap()
        .join()
        .map_err(|_| anyhow::anyhow!("prefill worker panicked"))?;
    assert!(
        observed,
        "the long prefill never yielded an intermediate scheduler boundary"
    );
    decode.expect("concurrent decode attempted")?;
    assert!(
        unfinished_at_decode,
        "decode did not complete before long prefill finished"
    );
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("cancellation must terminate the unfinished prefill"),
    };
    assert_eq!(
        serde_json::to_value(error.body())?["error"]["code"],
        "request_cancelled"
    );
    kv.wait_for_exact_state_captures_idle(Duration::from_secs(30))
        .map_err(anyhow::Error::msg)?;
    let check_session = session.clone();
    let remaining =
        backend
            .iteration_scheduler
            .execute_runtime("test-cancel-cleanup", move |rt| {
                let remaining = rt.session_token_count(&check_session).unwrap_or_default();
                rt.drop_session_timed("concurrent-decode")
                    .map_err(crate::frontend::util::openai_backend_error)?;
                Ok(remaining)
            })?;
    assert_eq!(
        remaining, 0,
        "cancelled execution lane still owns prompt state"
    );
    let base = backend.local_kv_message_base(
        &session,
        &InferenceGenerationIds::new_with_trust(
            InferenceCacheHints::default(),
            Some("checkpoint-probe"),
            true,
            None,
        ),
    );
    let checkpoint = kv.prefill_identity(&backend.config, &base, 0, &prompt[..64]);
    assert!(
        kv.retained_exact_identity(&checkpoint.namespace, &prompt[..64]),
        "cancellation lost the captured immutable checkpoint"
    );
    assert_eq!(
        kv.radix.lock().unwrap().stats().resident_pinned_tokens,
        0,
        "checkpoint lease leaked after cancellation"
    );
    Ok(())
}
