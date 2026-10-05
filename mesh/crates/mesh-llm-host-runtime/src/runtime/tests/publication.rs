use super::*;

#[test]
fn publish_state_updates_map_to_api_states() {
    assert_eq!(
        publication_state_from_update(nostr::PublishStateUpdate::Public),
        api::PublicationState::Public
    );
    assert_eq!(
        publication_state_from_update(nostr::PublishStateUpdate::PublishFailed),
        api::PublicationState::PublishFailed
    );
}

#[tokio::test]
async fn publication_bridge_keeps_private_until_a_real_publish_outcome_arrives() {
    let state = build_test_mesh_api().await;
    let (status_tx, status_rx) = tokio::sync::watch::channel(None);
    bridge_publication_state(state.clone(), status_rx);

    assert_eq!(state.publication_state().await.as_str(), "private");

    status_tx
        .send(Some(nostr::PublishStateUpdate::Public))
        .unwrap();
    wait_for_condition(Duration::from_secs(2), || {
        let state = state.clone();
        async move { state.publication_state().await.as_str() == "public" }
    })
    .await;

    status_tx
        .send(Some(nostr::PublishStateUpdate::PublishFailed))
        .unwrap();
    wait_for_condition(Duration::from_secs(2), || {
        let state = state.clone();
        async move { state.publication_state().await.as_str() == "publish_failed" }
    })
    .await;
}
