use mesh_llm_membership::announcements::{apply_transitive_ann, peer_meaningfully_changed};

async fn self_announcement(node: &Node) -> PeerAnnouncement {
    node.collect_announcements()
        .await
        .into_iter()
        .find(|announcement| announcement.addr.id == node.id())
        .expect("self announcement must be present")
}

#[tokio::test]
pub(crate) async fn test_default_node_does_not_advertise_download_refusal() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    let announcement = self_announcement(&node).await;

    assert!(!announcement.refuses_peer_model_downloads);
    assert_eq!(
        crate::protocol::local_ann_to_proto_ann(&announcement).refuses_peer_model_downloads,
        None,
        "a default node must send the same bytes as before the field existed"
    );
}

#[tokio::test]
pub(crate) async fn test_node_refusing_peer_downloads_advertises_it() {
    let mut node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    node.peer_stage_source_fetch = crate::inference::skippy::StageSourceFetch::LocalOnly;

    let announcement = self_announcement(&node).await;

    assert!(announcement.refuses_peer_model_downloads);
    assert_eq!(
        crate::protocol::local_ann_to_proto_ann(&announcement).refuses_peer_model_downloads,
        Some(true)
    );
}

#[test]
pub(crate) fn test_meaningfully_changed_peer_model_download_refusal() {
    let old_peer = test_peer(Some(100));
    let mut new_peer = test_peer(Some(100));
    new_peer.refuses_peer_model_downloads = !old_peer.refuses_peer_model_downloads;

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_apply_transitive_ann_preserves_direct_download_refusal() {
    let mut existing = test_peer(Some(100));
    existing.refuses_peer_model_downloads = true;
    // An older relay drops the field, which decodes as "accepts downloads".
    let old_relay_ann = test_announcement(Some(100));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &old_relay_ann,
        test_endpoint_id(0xee),
        1_800_000_000_000,
    );

    assert!(existing.refuses_peer_model_downloads);
}

#[tokio::test]
pub(crate) async fn test_direct_announcement_sets_and_clears_download_refusal() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x4c);
    let addr = test_addr(0x4c);
    let mut ann = test_announcement(Some(100));
    ann.refuses_peer_model_downloads = true;

    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    {
        let state = node.state.lock().await;
        let peer = state.peers.get(&peer_id).expect("peer should be tracked");
        assert!(peer.refuses_peer_model_downloads);
    }

    ann.refuses_peer_model_downloads = false;
    node.add_peer(peer_id, addr, &ann, None).await;
    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert!(!peer.refuses_peer_model_downloads);
}

#[tokio::test]
pub(crate) async fn test_relayed_download_refusal_reaches_a_new_transitive_peer() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x4d);
    let addr = test_addr(0x4d);
    let mut ann = test_announcement(Some(100));
    ann.addr = addr.clone();
    ann.refuses_peer_model_downloads = true;

    node.update_transitive_peer(peer_id, &addr, &ann, test_endpoint_id(0xed))
        .await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert!(peer.refuses_peer_model_downloads);
}
