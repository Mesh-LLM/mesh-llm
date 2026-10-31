//! Queue, resource and trusted-session admission contracts.
use super::*;
use crate::frontend::generation::ADMISSION_STARVATION_BOUND_TURNS;

fn trusted_ids(session_id: &str) -> InferenceGenerationIds {
    InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some(session_id),
        true,
        None,
    )
}

fn trusted_session_key(session_id: &str) -> String {
    trusted_generation_session_key(&trusted_ids(session_id)).expect("trusted session key")
}

fn admission_controller(
    generation_concurrency: usize,
    generation_queue_limit: usize,
) -> GenerationAdmissionController {
    admission_controller_with_budget(generation_concurrency, generation_queue_limit, 4_096)
}

fn admission_controller_with_budget(
    generation_concurrency: usize,
    generation_queue_limit: usize,
    token_capacity: usize,
) -> GenerationAdmissionController {
    GenerationAdmissionController {
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(
            generation_concurrency,
        )),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit,
        generation_service_estimator: Arc::new(GenerationServiceEstimator::new(
            generation_concurrency,
        )),
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(token_capacity)),
    }
}

fn result_error<T>(result: InferenceResult<T>) -> InferenceError {
    match result {
        Ok(_) => panic!("expected generation admission to fail"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn queued_admission_balances_shared_prefix_families_across_lane_wave() {
    let controller = admission_controller(1, 4);
    let work = GenerationAdmissionWork::new(4, 1);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::from_secs(2),
            work,
        )
        .await
        .expect("active request admission");
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    for (label, prompt) in [
        ("family-a-1", vec![1, 1, 3, 4]),
        ("family-a-2", vec![1, 1, 3, 5]),
        ("family-b-1", vec![2, 2, 3, 4]),
        ("family-b-2", vec![2, 2, 3, 5]),
    ] {
        let controller = controller.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let cancellation = skippy_inference_api::CancellationToken::new();
            let admitted = controller
                .acquire_scheduled_work(
                    &trusted_ids(label),
                    &cancellation,
                    Duration::from_secs(2),
                    work,
                    GenerationAdmissionScheduling::new(
                        Arc::from(prompt),
                        Arc::new(skippy_scheduler::CacheAffinity::default),
                    ),
                )
                .await
                .expect("queued request admission");
            tx.send((label, admitted)).await.unwrap();
        });
    }
    drop(tx);
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 4 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all prompts become scheduler-visible");

    drop(active);
    let (first_label, first) = tokio::time::timeout(Duration::from_millis(250), rx.recv())
        .await
        .expect("first queued admission was promoted")
        .expect("first queued admission");
    drop(first);
    let (second_label, second) = rx.recv().await.expect("second queued admission");
    assert_ne!(
        first_label.split('-').nth(1),
        second_label.split('-').nth(1),
        "one family must not drain the whole lane wave"
    );
    drop(second);
    let (_, third) = rx.recv().await.expect("third queued admission");
    drop(third);
    let (_, fourth) = rx.recv().await.expect("fourth queued admission");
    drop(fourth);
}

#[tokio::test]
async fn capacity_waiter_holds_neither_a_lane_nor_kv_until_atomic_promotion() {
    let controller = admission_controller_with_budget(2, 2, 10);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::ZERO,
            GenerationAdmissionWork::new(7, 0),
        )
        .await
        .expect("first capacity reservation");
    let waiting_controller = controller.clone();
    let waiter = tokio::spawn(async move {
        waiting_controller
            .acquire_work(
                &trusted_ids("waiting"),
                &skippy_inference_api::CancellationToken::new(),
                Duration::ZERO,
                GenerationAdmissionWork::new(7, 0),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("capacity waiter entered the queue");
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 7);

    drop(active);
    let promoted = tokio::time::timeout(Duration::from_millis(100), waiter)
        .await
        .expect("capacity waiter promoted")
        .expect("capacity waiter task completed")
        .expect("capacity waiter admission");
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 7);
    drop(promoted);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn capacity_waiters_drain_serially_after_each_kv_release() {
    let controller = admission_controller_with_budget(2, 4, 10);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::ZERO,
            GenerationAdmissionWork::new(10, 0),
        )
        .await
        .expect("active capacity reservation");
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    for index in 0..4 {
        let controller = controller.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let cancellation = skippy_inference_api::CancellationToken::new();
            let admitted = controller
                .acquire_work(
                    &trusted_ids(&format!("waiting-{index}")),
                    &cancellation,
                    Duration::ZERO,
                    GenerationAdmissionWork::new(10, 0),
                )
                .await
                .expect("queued capacity admission");
            tx.send(admitted).await.unwrap();
        });
    }
    drop(tx);
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 4 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all capacity waiters entered the queue");

    drop(active);
    for _ in 0..4 {
        let admitted = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("next capacity waiter promoted")
            .expect("queued capacity admission");
        assert_eq!(controller.generation_token_budget.active_tokens(), 10);
        drop(admitted);
    }
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn full_pool_waiter_is_admitted_after_bounded_half_pool_bypasses() {
    let controller = admission_controller_with_budget(2, 4, 10);
    let half_pool_work = GenerationAdmissionWork::new(5, 0);
    let full_pool_work = GenerationAdmissionWork::new(10, 0);
    let active = controller
        .acquire_work(
            &trusted_ids("active-half-pool"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::ZERO,
            half_pool_work,
        )
        .await
        .expect("initial half-pool admission");

    let full_pool_controller = controller.clone();
    let full_pool_waiter = tokio::spawn(async move {
        full_pool_controller
            .acquire_work(
                &trusted_ids("full-pool-waiter"),
                &skippy_inference_api::CancellationToken::new(),
                Duration::ZERO,
                full_pool_work,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("full-pool request entered the queue");

    // Keep one half-pool reservation active and repeatedly fill/release the
    // other half. The full-pool request cannot fit during these admissions.
    for wave in 0..ADMISSION_STARVATION_BOUND_TURNS {
        let bypass = tokio::time::timeout(
            Duration::from_secs(1),
            controller.acquire_work(
                &trusted_ids(&format!("half-pool-bypass-{wave}")),
                &skippy_inference_api::CancellationToken::new(),
                Duration::ZERO,
                half_pool_work,
            ),
        )
        .await
        .expect("fitting half-pool request completed before the timeout")
        .expect("fitting half-pool request admitted before the bound");
        assert_eq!(controller.generation_token_budget.active_tokens(), 10);
        drop(bypass);
    }

    // The next fitting arrival reaches the bound and must remain queued while
    // the controller drains capacity toward the older full-pool request.
    let bypass_controller = controller.clone();
    let blocked_bypass = tokio::spawn(async move {
        bypass_controller
            .acquire_work(
                &trusted_ids("half-pool-after-bound"),
                &skippy_inference_api::CancellationToken::new(),
                Duration::ZERO,
                half_pool_work,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("post-bound half-pool request remained queued");
    assert!(!full_pool_waiter.is_finished());
    assert!(!blocked_bypass.is_finished());

    drop(active);
    let admitted_full_pool = tokio::time::timeout(Duration::from_secs(1), full_pool_waiter)
        .await
        .expect("full-pool waiter admitted after capacity drained")
        .expect("full-pool waiter task completed")
        .expect("full-pool admission succeeded");
    assert_eq!(controller.generation_token_budget.active_tokens(), 10);
    assert!(!blocked_bypass.is_finished());

    drop(admitted_full_pool);
    let admitted_bypass = tokio::time::timeout(Duration::from_secs(1), blocked_bypass)
        .await
        .expect("younger half-pool waiter admitted after the senior completed")
        .expect("half-pool waiter task completed")
        .expect("half-pool admission succeeded");
    drop(admitted_bypass);
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn request_larger_than_the_kv_pool_fails_without_queueing_or_taking_a_lane() {
    let controller = admission_controller_with_budget(2, 2, 128);
    let error = result_error(
        controller
            .acquire_work(
                &trusted_ids("too-large"),
                &skippy_inference_api::CancellationToken::new(),
                Duration::ZERO,
                GenerationAdmissionWork::new(129, 0),
            )
            .await,
    );

    assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
    assert!(
        error
            .body()
            .error
            .message
            .contains("runtime pool holds 128")
    );
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn cancelling_a_capacity_waiter_leaks_neither_lane_nor_kv_reservation() {
    let controller = admission_controller_with_budget(2, 2, 128);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::ZERO,
            GenerationAdmissionWork::new(100, 0),
        )
        .await
        .expect("first capacity reservation");
    let cancellation = skippy_inference_api::CancellationToken::new();
    let waiter_cancellation = cancellation.clone();
    let waiting_controller = controller.clone();
    let waiter = tokio::spawn(async move {
        waiting_controller
            .acquire_work(
                &trusted_ids("waiting"),
                &waiter_cancellation,
                Duration::ZERO,
                GenerationAdmissionWork::new(64, 0),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("capacity waiter entered the queue");

    cancellation.cancel();
    let error = result_error(
        tokio::time::timeout(Duration::from_millis(100), waiter)
            .await
            .expect("cancelled capacity waiter returned")
            .expect("capacity waiter task completed"),
    );
    assert!(error.body().error.message.contains("request cancelled"));
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 100);

    drop(active);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn inflated_prediction_waits_for_capacity_and_preserves_queue_bounds() {
    let controller = admission_controller(1, 1);
    let work = GenerationAdmissionWork::new(100, 100);
    controller.generation_service_estimator.observe_completed(
        GenerationAdmissionWork::new(467, 21),
        994.1,
        115_557.901_853,
    );
    let active = controller
        .acquire_work(
            &trusted_ids("agent-1"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::from_secs(1),
            work,
        )
        .await
        .expect("active request admission");

    let timeout = Duration::from_secs(1);
    let predicted = controller
        .generation_service_estimator
        .predicted_wait_ms()
        .unwrap();
    assert!(predicted > timeout.as_secs_f64() * 1_000.0);
    let ids = trusted_ids("agent-2");
    let cancellation = skippy_inference_api::CancellationToken::new();
    let waiting = controller.acquire_work(&ids, &cancellation, timeout, work);
    tokio::pin!(waiting);
    // Poll synchronously: this must queue rather than immediately reject the
    // inflated prediction, without depending on scheduling delays or sleeps.
    assert!(futures_util::poll!(&mut waiting).is_pending());
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 1);
    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert_eq!(controller.generation_token_budget.active_tokens(), 200);
    let overflow = result_error(
        controller
            .acquire_work(
                &trusted_ids("overflow"),
                &skippy_inference_api::CancellationToken::new(),
                timeout,
                work,
            )
            .await,
    );
    assert!(overflow.body().error.message.contains("queue is full"));
    drop(active);
    let admitted = waiting
        .await
        .expect("released lane admits despite inflated prediction");
    assert_eq!(admitted.0.predicted_wait_ms, Some(predicted));
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_token_budget.active_tokens(), 200);
    drop(admitted);
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
    assert_eq!(
        controller.generation_service_estimator.predicted_wait_ms(),
        Some(0.0)
    );
    assert!(
        controller
            .generation_session_locks
            .lock()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn advisory_prediction_does_not_bypass_an_expired_admission_deadline() {
    let controller = admission_controller(1, 1);
    let work = GenerationAdmissionWork::new(100, 100);
    controller
        .generation_service_estimator
        .observe_completed(work, 100.0, 100.0);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::from_secs(1),
            work,
        )
        .await
        .expect("active request admission");
    let error = result_error(
        controller
            .acquire_generation_permit_until(
                Some(Instant::now()),
                Duration::from_secs(1),
                &skippy_inference_api::CancellationToken::new(),
                work,
                GenerationAdmissionScheduling::default(),
            )
            .await,
    );
    assert!(error.body().error.message.contains("timed out waiting"));
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert_eq!(controller.generation_token_budget.active_tokens(), 200);
    drop(active);
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
    assert_eq!(
        controller.generation_service_estimator.predicted_wait_ms(),
        Some(0.0)
    );
}

#[test]
fn session_registry_counts_live_leases_and_cleans_replaced_entries() {
    let registry = Arc::new(Mutex::new(BTreeMap::new()));
    let first = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("first session lease");
    let second = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("second session lease");

    {
        let locks = registry.lock().expect("session registry lock");
        let entry = locks.get("agent-1").expect("shared session entry");
        assert_eq!(entry.users.load(Ordering::Acquire), 2);
    }

    drop(first);
    {
        let locks = registry.lock().expect("session registry lock");
        let entry = locks.get("agent-1").expect("live session entry");
        assert_eq!(entry.users.load(Ordering::Acquire), 1);
    }

    drop(second);
    assert!(registry.lock().expect("session registry lock").is_empty());

    let replacement = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("replacement session lease");
    assert_eq!(registry.lock().expect("session registry lock").len(), 1);
    drop(replacement);
    assert!(registry.lock().expect("session registry lock").is_empty());
}

#[tokio::test]
async fn same_trusted_session_serializes_without_consuming_global_queue_capacity() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let first_cancellation = skippy_inference_api::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let second_controller = controller.clone();
    let second_cancellation = skippy_inference_api::CancellationToken::new();
    let waiter_cancellation = second_cancellation.clone();
    let waiter = tokio::spawn(async move {
        second_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("second turn registered its session wait");

    assert!(!waiter.is_finished());
    assert_eq!(
        controller.generation_queue_depth.load(Ordering::Acquire),
        0,
        "session contention must not consume global queue capacity"
    );

    second_cancellation.cancel();
    let error = result_error(
        tokio::time::timeout(Duration::from_millis(100), waiter)
            .await
            .expect("cancelled session waiter returned")
            .expect("session waiter task completed"),
    );
    assert!(error.body().error.message.contains("request cancelled"));
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    drop(first);
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[tokio::test]
async fn same_trusted_session_acquires_only_after_the_first_turn_releases() {
    let controller = admission_controller(1, 1);
    let first_cancellation = skippy_inference_api::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let second_controller = controller.clone();
    let second = tokio::spawn(async move {
        second_controller
            .acquire(
                &trusted_ids("agent-1"),
                &skippy_inference_api::CancellationToken::new(),
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(!second.is_finished());
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    drop(first);
    let second = tokio::time::timeout(Duration::from_millis(100), second)
        .await
        .expect("second turn acquired after first released")
        .expect("second turn task completed")
        .expect("second session admission");
    assert_eq!(controller.generation_limit.available_permits(), 0);
    drop(second);
    assert_eq!(controller.generation_limit.available_permits(), 1);
}

#[tokio::test]
async fn session_and_global_admission_share_one_absolute_deadline() {
    let controller = admission_controller(1, 1);
    let first_cancellation = skippy_inference_api::CancellationToken::new();
    let (global_permit, session_permit) = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");
    let session_permit = session_permit.expect("trusted session permit");
    let release_session = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(140)).await;
        drop(session_permit);
    });
    let started = Instant::now();

    let error = result_error(
        controller
            .acquire(
                &trusted_ids("agent-1"),
                &skippy_inference_api::CancellationToken::new(),
                Duration::from_millis(200),
            )
            .await,
    );

    assert!(error.body().error.message.contains("timed out waiting"));
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "global-lane waiting must not restart the request admission timeout"
    );
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    release_session
        .await
        .expect("session release task completed");
    drop(global_permit);
}

#[tokio::test]
async fn unrelated_session_is_not_starved_by_a_same_session_waiter() {
    let controller = admission_controller(2, 2);
    let session_key = trusted_session_key("agent-1");
    let first_cancellation = skippy_inference_api::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let duplicate_controller = controller.clone();
    let duplicate_cancellation = skippy_inference_api::CancellationToken::new();
    let waiter_cancellation = duplicate_cancellation.clone();
    let duplicate = tokio::spawn(async move {
        duplicate_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("duplicate turn registered its session wait");

    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    let unrelated = controller
        .acquire(
            &trusted_ids("agent-2"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::from_millis(100),
        )
        .await
        .expect("unrelated session used the free global lane");
    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert!(!duplicate.is_finished());

    duplicate_cancellation.cancel();
    let duplicate_error = result_error(
        tokio::time::timeout(Duration::from_millis(100), duplicate)
            .await
            .expect("duplicate waiter cancelled")
            .expect("duplicate waiter task completed"),
    );
    assert!(
        duplicate_error
            .body()
            .error
            .message
            .contains("request cancelled")
    );
    drop((first, unrelated));
    assert_eq!(controller.generation_limit.available_permits(), 2);
}

#[tokio::test]
async fn same_session_waiter_does_not_reserve_the_only_global_queue_slot() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let duplicate_controller = controller.clone();
    let duplicate_cancellation = skippy_inference_api::CancellationToken::new();
    let waiter_cancellation = duplicate_cancellation.clone();
    let duplicate = tokio::spawn(async move {
        duplicate_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("duplicate turn registered its session wait");
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    let unrelated_controller = controller.clone();
    let unrelated = tokio::spawn(async move {
        unrelated_controller
            .acquire(
                &trusted_ids("agent-2"),
                &skippy_inference_api::CancellationToken::new(),
                Duration::from_secs(1),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("unrelated turn reserved the only global queue slot");
    assert!(!unrelated.is_finished());

    duplicate_cancellation.cancel();
    let duplicate_error = result_error(
        tokio::time::timeout(Duration::from_millis(100), duplicate)
            .await
            .expect("duplicate waiter cancelled")
            .expect("duplicate waiter task completed"),
    );
    assert_eq!(duplicate_error.status().as_u16(), 499);

    drop(first);
    let unrelated = tokio::time::timeout(Duration::from_millis(100), unrelated)
        .await
        .expect("unrelated turn acquired the released lane")
        .expect("unrelated waiter task completed")
        .expect("unrelated session admission");
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    drop(unrelated);
}

#[tokio::test]
async fn different_trusted_sessions_can_hold_generation_lanes_concurrently() {
    let controller = admission_controller(2, 2);
    let first_cancellation = skippy_inference_api::CancellationToken::new();
    let second_cancellation = skippy_inference_api::CancellationToken::new();
    let first_ids = trusted_ids("agent-1");
    let second_ids = trusted_ids("agent-2");

    let (first, second) = tokio::join!(
        controller.acquire(&first_ids, &first_cancellation, Duration::from_secs(1)),
        controller.acquire(&second_ids, &second_cancellation, Duration::from_secs(1)),
    );
    let first = first.expect("first session admission");
    let second = second.expect("second session admission");

    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert_eq!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .len(),
        2
    );

    drop((first, second));
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn blocking_worker_holds_global_and_session_permits_until_work_finishes() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let cancellation = skippy_inference_api::CancellationToken::new();
    let (global_permit, session_permit) = controller
        .acquire(
            &trusted_ids("agent-1"),
            &cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("worker admission");
    let worker_context = InferenceRequestContext::new();
    let worker_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let release_worker = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started = worker_started.clone();
    let release = release_worker.clone();

    let worker = tokio::spawn(run_blocking_generation_worker(
        global_permit,
        worker_context,
        move |_| {
            let _session_permit = session_permit;
            started.store(true, Ordering::Release);
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        },
    ));

    tokio::time::timeout(Duration::from_millis(100), async {
        while !worker_started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("blocking generation worker started");

    assert_eq!(controller.generation_limit.available_permits(), 0);
    let entry = controller
        .generation_session_locks
        .lock()
        .expect("session registry lock")
        .get(&session_key)
        .expect("worker retains session entry")
        .semaphore
        .clone();
    assert_eq!(entry.available_permits(), 0);

    release_worker.store(true, Ordering::Release);
    tokio::time::timeout(Duration::from_millis(100), worker)
        .await
        .expect("blocking generation worker completed")
        .expect("worker task completed")
        .expect("blocking worker joined");
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[test]
fn untrusted_conversation_affinity_bypasses_session_registry() {
    let registry = Arc::new(Mutex::new(BTreeMap::new()));
    let untrusted = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("conversation-7"),
        false,
        None,
    );
    assert!(trusted_generation_session_key(&untrusted).is_none());
    assert!(registry.lock().expect("session registry lock").is_empty());

    let trusted = trusted_ids("agent-7");
    let key = trusted_generation_session_key(&trusted).expect("trusted session key");
    assert_eq!(key, trusted.session_id_string());
    let _permit =
        GenerationSessionPermit::new(registry.clone(), key).expect("trusted session lease");
    assert_eq!(registry.lock().expect("session registry lock").len(), 1);
}
