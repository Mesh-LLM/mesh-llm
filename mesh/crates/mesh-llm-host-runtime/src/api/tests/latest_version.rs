#[tokio::test]
async fn test_console_latest_version_check_sends_no_request_when_turned_off() {
    let state = build_test_mesh_api().await;
    state.set_update_check(false).await;

    record_latest_version(state.clone(), || async {
        panic!("the release check must not run");
    })
    .await;

    assert_eq!(state.inner.lock().await.latest_version, None);
}

#[tokio::test]
async fn test_console_latest_version_check_records_a_newer_release() {
    let state = build_test_mesh_api().await;

    record_latest_version(state.clone(), || async { Some("999.0.0".to_string()) }).await;

    assert_eq!(
        state.inner.lock().await.latest_version.as_deref(),
        Some("999.0.0")
    );
}
