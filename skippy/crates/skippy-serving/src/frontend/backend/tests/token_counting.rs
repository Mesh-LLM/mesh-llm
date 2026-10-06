use super::*;

#[tokio::test]
async fn token_counting_obeys_full_admission_before_prompt_preparation() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let mut backend = hooks_test_backend(None);
    backend.generation_queue_limit = 0;
    let controller = GenerationAdmissionController::for_backend(&backend);
    let active = controller
        .acquire_work(
            &trusted_ids("active-generation"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::from_secs(1),
            GenerationAdmissionWork::new(1, 0),
        )
        .await
        .expect("occupy the only generation lane");

    // Ordinary work is rejected under this exact admission state.
    let rejected = controller
        .acquire_work(
            &trusted_ids("excess-generation"),
            &skippy_inference_api::CancellationToken::new(),
            Duration::from_secs(1),
            GenerationAdmissionWork::new(1, 0),
        )
        .await;
    assert_eq!(result_error(rejected).status().as_u16(), 429);

    // Make entering native execution observable without requiring a GGUF.
    // Admission needs no runtime lock, so a correctly rejected request never
    // encounters this sentinel. It also prevents use of the fixture's dummy FFI.
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = backend.runtime.lock().unwrap();
        panic!("intentional runtime poison for admission regression");
    }));
    assert!(poisoned.is_err());

    let app = skippy_inference_api::router(Arc::new(backend));
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        app.oneshot(
            Request::post("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "hooks-test-model",
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        ),
    )
    .await
    .expect("overload must be rejected promptly")
    .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
        .await
        .unwrap();
    drop(active);

    assert_eq!(
        status.as_u16(),
        429,
        "Token counting bypassed full admission and reached prompt preparation: {}",
        String::from_utf8_lossy(&body)
    );
}

#[tokio::test]
async fn token_count_admission_survives_dropped_blocking_handle() {
    let mut backend = hooks_test_backend(None);
    backend.generation_queue_limit = 0;
    let admission = backend.acquire_token_count_admission().await.unwrap();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let job = tokio::task::spawn_blocking(move || {
        let admission = admission;
        started_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(admission);
        let _ = done_tx.send(());
    });
    started_rx.await.unwrap();
    drop(job);
    let rejected = backend.acquire_token_count_admission().await;
    // Release before asserting, so a failed assertion cannot strand the worker.
    release_tx.send(()).unwrap();
    done_rx.await.unwrap();
    match rejected {
        Err(error) => assert_eq!(error.status().as_u16(), 429),
        Ok(_) => panic!("dropping a blocking handle released admission early"),
    }
    assert!(backend.acquire_token_count_admission().await.is_ok());
}
