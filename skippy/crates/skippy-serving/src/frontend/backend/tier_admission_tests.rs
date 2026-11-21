//! Admission by request tier: higher tiers are admitted first, keep their
//! room in the waiting queue and in the KV pool, and are not held behind a
//! lower tier's head-of-line reservation.

use super::*;
use crate::frontend::generation::ADMISSION_STARVATION_BOUND_TURNS;
use skippy_inference_api::{CancellationToken, InferenceRequestClass, InferenceRequestTier};
use skippy_scheduler::PRIORITY_AGING_TURNS;

/// A generous bound for an event the test waits on, so a slow or busy
/// machine does not fail it.
const WAIT: Duration = Duration::from_secs(60);

fn controller(
    generation_concurrency: usize,
    generation_queue_limit: usize,
    token_capacity: usize,
    local_reserve_percent: u8,
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
        generation_token_budget: Arc::new(
            GenerationTokenBudget::new(token_capacity)
                .with_local_reserve_percent(local_reserve_percent),
        ),
    }
}

fn ids(session_id: &str, tier: InferenceRequestTier) -> InferenceGenerationIds {
    InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some(session_id),
        true,
        None,
    )
    .with_request_class(InferenceRequestClass::new(tier, false))
}

async fn admit(
    controller: &GenerationAdmissionController,
    session_id: &str,
    tier: InferenceRequestTier,
    tokens: usize,
) -> InferenceResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
    controller
        .acquire_work(
            &ids(session_id, tier),
            &CancellationToken::new(),
            Duration::ZERO,
            GenerationAdmissionWork::new(tokens, 0),
        )
        .await
}

async fn wait_for_queue_depth(controller: &GenerationAdmissionController, depth: usize) {
    tokio::time::timeout(WAIT, async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != depth {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("waiters reached the queue");
}

#[tokio::test]
async fn the_local_reserve_is_kept_for_local_requests() {
    let controller = controller(4, 4, 10, 50);
    let too_large = admit(&controller, "mesh-large", InferenceRequestTier::Mesh, 6).await;
    let error = too_large
        .err()
        .expect("a mesh request cannot use the reserve");
    assert!(
        error.body().error.message.contains("reserved for local"),
        "{}",
        error.body().error.message
    );

    let mesh = admit(&controller, "mesh", InferenceRequestTier::Mesh, 5)
        .await
        .expect("mesh fits outside the reserve");
    let local = tokio::time::timeout(
        WAIT,
        admit(&controller, "local", InferenceRequestTier::Local, 5),
    )
    .await
    .expect("local is not queued behind mesh")
    .expect("local fits in the reserve");
    assert_eq!(controller.generation_token_budget.active_tokens(), 10);
    drop((mesh, local));
}

#[tokio::test]
async fn a_lower_tier_cannot_fill_the_waiting_room_ahead_of_local() {
    let controller = controller(1, 1, 100, 0);
    let active = admit(&controller, "active", InferenceRequestTier::Local, 1)
        .await
        .expect("the lane is free");

    let waiting_controller = controller.clone();
    let mesh_waiter = tokio::spawn(async move {
        admit(&waiting_controller, "mesh-1", InferenceRequestTier::Mesh, 1).await
    });
    wait_for_queue_depth(&controller, 1).await;
    let full = admit(&controller, "mesh-2", InferenceRequestTier::Mesh, 1).await;
    assert!(full.is_err(), "the mesh share of the room is full");

    let local_controller = controller.clone();
    let local_waiter = tokio::spawn(async move {
        admit(&local_controller, "local", InferenceRequestTier::Local, 1).await
    });
    wait_for_queue_depth(&controller, 2).await;

    // The younger local waiter is admitted first.
    drop(active);
    let local = tokio::time::timeout(WAIT, local_waiter)
        .await
        .expect("local admitted")
        .unwrap()
        .expect("local admission succeeded");
    assert!(!mesh_waiter.is_finished());
    drop(local);
    let mesh = tokio::time::timeout(WAIT, mesh_waiter)
        .await
        .expect("mesh admitted after local")
        .unwrap()
        .expect("mesh admission succeeded");
    drop(mesh);
}

#[tokio::test]
async fn an_unfitting_senior_mesh_waiter_does_not_hold_back_local_requests() {
    // 10 tokens, 5 of them reserved for local requests.
    let controller = controller(4, 4, 10, 50);
    // A long local request keeps the mesh share full.
    let long_local = admit(&controller, "long-local", InferenceRequestTier::Local, 6)
        .await
        .expect("local fits");
    let mesh_controller = controller.clone();
    let mesh_waiter = tokio::spawn(async move {
        admit(&mesh_controller, "mesh", InferenceRequestTier::Mesh, 1).await
    });
    wait_for_queue_depth(&controller, 1).await;

    // Past the starvation bound, local requests that fit are still admitted.
    for wave in 0..ADMISSION_STARVATION_BOUND_TURNS + 4 {
        let local = tokio::time::timeout(
            WAIT,
            admit(
                &controller,
                &format!("local-{wave}"),
                InferenceRequestTier::Local,
                4,
            ),
        )
        .await
        .unwrap_or_else(|_| panic!("local request {wave} was held behind the mesh waiter"))
        .expect("local admission succeeded");
        drop(local);
    }
    assert!(!mesh_waiter.is_finished());

    drop(long_local);
    let mesh = tokio::time::timeout(WAIT, mesh_waiter)
        .await
        .expect("mesh admitted once the pool drains")
        .unwrap()
        .expect("mesh admission succeeded");
    drop(mesh);
}

#[tokio::test]
async fn an_older_mesh_waiter_does_not_take_a_starving_local_waiters_hold() {
    let controller = controller(4, 8, 10, 0);
    let active = admit(&controller, "active", InferenceRequestTier::Local, 6)
        .await
        .expect("the pool is free");
    // The oldest waiter is a mesh request; a large local one queues behind it.
    let mesh_controller = controller.clone();
    let mesh_waiter = tokio::spawn(async move {
        admit(&mesh_controller, "mesh", InferenceRequestTier::Mesh, 5).await
    });
    wait_for_queue_depth(&controller, 1).await;
    let large_controller = controller.clone();
    let large_local = tokio::spawn(async move {
        admit(
            &large_controller,
            "large-local",
            InferenceRequestTier::Local,
            7,
        )
        .await
    });
    wait_for_queue_depth(&controller, 2).await;

    // Smaller local requests pass the large one until it has waited long enough.
    for turn in 0..ADMISSION_STARVATION_BOUND_TURNS {
        let small = tokio::time::timeout(
            WAIT,
            admit(
                &controller,
                &format!("small-{turn}"),
                InferenceRequestTier::Local,
                4,
            ),
        )
        .await
        .unwrap_or_else(|_| panic!("small local request {turn} was held"))
        .expect("small local admission succeeded");
        drop(small);
    }

    // Now the pool drains for the large local request, not for the older
    // mesh one: a small local newcomer waits. Had it been admitted, its
    // tokens would keep the large request out of the drained pool.
    let small_controller = controller.clone();
    let late_small = tokio::spawn(async move {
        admit(
            &small_controller,
            "late-small",
            InferenceRequestTier::Local,
            4,
        )
        .await
    });
    wait_for_queue_depth(&controller, 3).await;
    drop(active);
    let large = tokio::time::timeout(WAIT, large_local)
        .await
        .expect("the large local request is admitted once the pool drains")
        .unwrap()
        .expect("large local admission succeeded");
    drop(large);
    for waiter in [late_small, mesh_waiter] {
        let admitted = tokio::time::timeout(WAIT, waiter)
            .await
            .expect("the remaining waiters are admitted")
            .unwrap()
            .expect("admission succeeded");
        drop(admitted);
    }
}

#[tokio::test]
async fn a_local_request_is_admitted_ahead_of_a_long_mesh_queue() {
    const MESH_WAITERS: usize = 40;
    let controller = controller(1, 64, 1_000, 0);
    let active = admit(&controller, "active", InferenceRequestTier::Local, 1)
        .await
        .expect("the lane is free");
    // Each mesh request holds the lane until the test releases it.
    let (admitted_tx, mut admitted_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut releases = Vec::new();
    for index in 0..MESH_WAITERS {
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        releases.push(Some(release_tx));
        let (mesh_controller, admitted_tx) = (controller.clone(), admitted_tx.clone());
        tokio::spawn(async move {
            let permit = admit(
                &mesh_controller,
                &format!("mesh-{index}"),
                InferenceRequestTier::Mesh,
                1,
            )
            .await;
            // Only a request that holds the lane counts as admitted.
            let _ = admitted_tx.send(
                permit
                    .as_ref()
                    .map(|_| index)
                    .map_err(|error| error.body().error.message.clone()),
            );
            let _ = release_rx.await;
            drop(permit);
        });
    }
    wait_for_queue_depth(&controller, MESH_WAITERS).await;

    // Serve the mesh queue until every remaining mesh waiter is past the
    // starvation bound, keeping the last one admitted on the lane.
    drop(active);
    let bound = usize::try_from(ADMISSION_STARVATION_BOUND_TURNS).unwrap();
    let mut on_lane = None;
    for served in 0..bound {
        let index = tokio::time::timeout(WAIT, admitted_rx.recv())
            .await
            .expect("a mesh request is admitted")
            .unwrap()
            .expect("mesh admission succeeded");
        if served + 1 < bound {
            releases[index].take().unwrap().send(()).unwrap();
        } else {
            on_lane = Some(index);
        }
    }

    // A local request that arrives now goes next, not after the mesh queue.
    let local_controller = controller.clone();
    let local = tokio::spawn(async move {
        admit(&local_controller, "local", InferenceRequestTier::Local, 1).await
    });
    wait_for_queue_depth(&controller, MESH_WAITERS - bound + 1).await;
    releases[on_lane.unwrap()].take().unwrap().send(()).unwrap();
    let local = tokio::time::timeout(WAIT, local)
        .await
        .expect("the local request is admitted next")
        .unwrap()
        .expect("local admission succeeded");
    assert!(
        admitted_rx.try_recv().is_err(),
        "no mesh request went first"
    );
    drop(local);
    for release in releases.into_iter().flatten() {
        let _ = release.send(());
    }
}

/// Which waiter one admission wave went to.
enum Wave<T> {
    Mesh(T),
    Local(T),
}

#[tokio::test]
async fn a_mesh_request_passed_over_by_local_ones_is_admitted_in_bounded_time() {
    let controller = controller(1, 8, 1_000, 0);
    let mut on_lane = admit(&controller, "active", InferenceRequestTier::Local, 1)
        .await
        .expect("the lane is free");
    let mesh_controller = controller.clone();
    let mut mesh = tokio::spawn(async move {
        admit(&mesh_controller, "mesh", InferenceRequestTier::Mesh, 1).await
    });
    wait_for_queue_depth(&controller, 1).await;

    // A steady stream of local requests: one is always waiting with the mesh one.
    let limit = 2 * PRIORITY_AGING_TURNS + 1;
    for passed in 0..=limit {
        let local_controller = controller.clone();
        let mut local = tokio::spawn(async move {
            admit(
                &local_controller,
                &format!("local-{passed}"),
                InferenceRequestTier::Local,
                1,
            )
            .await
        });
        wait_for_queue_depth(&controller, 2).await;
        drop(on_lane);
        let wave = tokio::time::timeout(WAIT, async {
            tokio::select! {
                admitted = &mut mesh => Wave::Mesh(admitted),
                admitted = &mut local => Wave::Local(admitted),
            }
        })
        .await
        .unwrap_or_else(|_| panic!("nobody was admitted in wave {passed}"));
        match wave {
            Wave::Mesh(admitted) => {
                let mesh = admitted.unwrap().expect("mesh admission succeeded");
                assert!(
                    passed >= 2 * PRIORITY_AGING_TURNS,
                    "mesh went early after {passed}"
                );
                drop(mesh);
                let local = tokio::time::timeout(WAIT, &mut local)
                    .await
                    .expect("the waiting local request follows")
                    .unwrap()
                    .expect("local admission succeeded");
                drop(local);
                return;
            }
            Wave::Local(admitted) => {
                on_lane = admitted.unwrap().expect("local admission succeeded");
            }
        }
        assert!(passed < limit, "mesh was never admitted");
    }
}

/// A `mesh` waiter elected while a bigger `local` waiter did not fit loses
/// the lane once a release makes room for the `local` one.
#[tokio::test]
async fn a_local_waiter_that_fits_after_a_release_goes_before_an_elected_mesh_one() {
    let controller = controller(1, 8, 10, 0);
    let active = admit(&controller, "active", InferenceRequestTier::Local, 6)
        .await
        .expect("the lane is free");
    let mesh_controller = controller.clone();
    let mesh_waiter = tokio::spawn(async move {
        admit(&mesh_controller, "mesh", InferenceRequestTier::Mesh, 2).await
    });
    // The mesh request queues first, so it is waiter 0.
    wait_for_queue_depth(&controller, 1).await;
    let local_controller = controller.clone();
    let local_waiter = tokio::spawn(async move {
        admit(&local_controller, "local", InferenceRequestTier::Local, 5).await
    });
    // Both wait, and the mesh waiter (id 0) is elected: only it fits the
    // tokens left beside the active request.
    let queue = controller.generation_limit.admission_queue();
    tokio::time::timeout(WAIT, async {
        while queue.election_for_test() != (2, Some(0)) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the mesh waiter was elected with both waiting");

    drop(active);
    let local = tokio::time::timeout(WAIT, local_waiter)
        .await
        .expect("local admitted")
        .unwrap()
        .expect("local admission succeeded");
    assert!(!mesh_waiter.is_finished());
    drop(local);
    let mesh = tokio::time::timeout(WAIT, mesh_waiter)
        .await
        .expect("mesh admitted after local")
        .unwrap()
        .expect("mesh admission succeeded");
    drop(mesh);
}
