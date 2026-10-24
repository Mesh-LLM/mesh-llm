use super::*;
use futures_util::poll;

#[tokio::test]
async fn aggregate_capacity_is_not_multiplied_by_memberships() {
    let admission = Admission::new(1, 2).unwrap();
    let a = admission.register(1);
    let b = admission.register(1);
    let running = admission
        .acquire(a, CancellationToken::new())
        .await
        .unwrap();
    let mut waiting = Box::pin(admission.acquire(b, CancellationToken::new()));
    assert!(poll!(waiting.as_mut()).is_pending());
    assert_eq!(admission.active(), 1);
    assert_eq!(admission.status(b).queued, 1);
    let extra = admission
        .acquire(b, CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(extra.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    let worker_copy = running.clone();
    drop(running);
    assert!(poll!(waiting.as_mut()).is_pending());
    drop(worker_copy);
    let _next = waiting.await.unwrap();
    assert_eq!(admission.active(), 1);
}

#[tokio::test]
async fn pending_memberships_take_round_robin_turns() {
    let admission = Admission::new(1, 8).unwrap();
    let a = admission.register(8);
    let b = admission.register(8);
    let running = admission
        .acquire(a, CancellationToken::new())
        .await
        .unwrap();
    let mut a_first = Box::pin(admission.acquire(a, CancellationToken::new()));
    let mut a_second = Box::pin(admission.acquire(a, CancellationToken::new()));
    let mut b_first = Box::pin(admission.acquire(b, CancellationToken::new()));
    assert!(poll!(a_first.as_mut()).is_pending());
    assert!(poll!(a_second.as_mut()).is_pending());
    assert!(poll!(b_first.as_mut()).is_pending());
    drop(running);
    let first = a_first.await.unwrap();
    assert!(poll!(a_second.as_mut()).is_pending());
    assert!(poll!(b_first.as_mut()).is_pending());
    drop(first);
    let second = b_first.await.unwrap();
    assert!(poll!(a_second.as_mut()).is_pending());
    drop(second);
    drop(a_second.await.unwrap());
    assert_eq!(admission.active(), 0);
}

#[tokio::test]
async fn dropped_and_cancelled_waiters_immediately_release_queue_budget() {
    let admission = Admission::new(1, 1).unwrap();
    let a = admission.register(1);
    let _running = admission
        .acquire(a, CancellationToken::new())
        .await
        .unwrap();
    let mut dropped = Box::pin(admission.acquire(a, CancellationToken::new()));
    assert!(poll!(dropped.as_mut()).is_pending());
    drop(dropped);
    assert_eq!(admission.status(a).queued, 0);
    let cancel = CancellationToken::new();
    let mut cancelled = Box::pin(admission.acquire(a, cancel.clone()));
    assert!(poll!(cancelled.as_mut()).is_pending());
    cancel.cancel();
    assert!(cancelled.await.is_err());
    assert_eq!(admission.status(a).queued, 0);
    assert_eq!(admission.active(), 1);
}

#[tokio::test]
async fn detach_cancels_only_its_member_and_holds_capacity_until_worker_exits() {
    let admission = Admission::new(2, 2).unwrap();
    let a = admission.register(2);
    let b = admission.register(2);
    let a_cancel = CancellationToken::new();
    let b_cancel = CancellationToken::new();
    let a_work = admission.acquire(a, a_cancel.clone()).await.unwrap();
    let b_work = admission.acquire(b, b_cancel.clone()).await.unwrap();
    let mut a_queue = Box::pin(admission.acquire(a, CancellationToken::new()));
    assert!(poll!(a_queue.as_mut()).is_pending());
    admission.retire(a);
    assert!(a_cancel.is_cancelled());
    assert!(!b_cancel.is_cancelled());
    assert!(a_queue.await.is_err());
    assert_eq!(admission.active(), 2);
    admission.resume(a);
    assert!(
        admission
            .acquire(a, CancellationToken::new())
            .await
            .is_err()
    );
    drop(a_work);
    assert!(!admission.state.lock().unwrap().members.contains_key(&a));
    assert_eq!(admission.active(), 1);
    drop(b_work);
    assert_eq!(admission.active(), 0);
}

#[tokio::test]
async fn pause_drains_in_flight_but_rejects_waiters_and_can_resume() {
    let admission = Admission::new(1, 1).unwrap();
    let a = admission.register(1);
    let cancel = CancellationToken::new();
    let work = admission.acquire(a, cancel.clone()).await.unwrap();
    let mut queued = Box::pin(admission.acquire(a, CancellationToken::new()));
    assert!(poll!(queued.as_mut()).is_pending());
    admission.pause(a, false);
    assert!(!cancel.is_cancelled());
    assert!(queued.await.is_err());
    assert!(
        admission
            .acquire(a, CancellationToken::new())
            .await
            .is_err()
    );
    drop(work);
    admission.resume(a);
    drop(
        admission
            .acquire(a, CancellationToken::new())
            .await
            .unwrap(),
    );
}

#[tokio::test]
async fn zero_queue_allows_immediate_work_but_never_waits() {
    let admission = Admission::new(1, 0).unwrap();
    let member = admission.register(0);
    let work = admission
        .acquire(member, CancellationToken::new())
        .await
        .unwrap();
    assert!(
        admission
            .acquire(member, CancellationToken::new())
            .await
            .is_err()
    );
    drop(work);
    drop(
        admission
            .acquire(member, CancellationToken::new())
            .await
            .unwrap(),
    );
}

#[tokio::test]
async fn aggregate_queue_limit_applies_even_with_spare_member_budgets() {
    let admission = Admission::new(1, 1).unwrap();
    let a = admission.register(8);
    let b = admission.register(8);
    let _running = admission
        .acquire(a, CancellationToken::new())
        .await
        .unwrap();
    let mut waiting = Box::pin(admission.acquire(b, CancellationToken::new()));
    assert!(poll!(waiting.as_mut()).is_pending());
    assert!(
        admission
            .acquire(a, CancellationToken::new())
            .await
            .is_err()
    );
}
