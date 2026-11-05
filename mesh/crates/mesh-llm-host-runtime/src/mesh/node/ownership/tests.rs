use super::*;
use crate::crypto::OwnershipStatus;
use crate::mesh::NodeRole;

#[test]
fn certificate_paths_preserve_scoped_and_ordinary_storage() {
    let directory = tempfile::tempdir().unwrap();
    assert_eq!(
        owner_attestation_path(Some(directory.path())).unwrap(),
        directory.path().join("node-ownership.json")
    );
    assert_eq!(
        owner_attestation_path(None).unwrap(),
        crate::crypto::default_node_ownership_path().unwrap()
    );
}

/// A node whose 7-day certificate enters its 36-hour renewal window re-signs
/// it for a full lifetime, so peers checking the expiry keep admitting it.
#[tokio::test]
async fn a_running_node_renews_its_ownership_certificate_inside_the_window() {
    use crate::crypto::{
        DEFAULT_NODE_CERT_LIFETIME_SECS, TrustPolicy, TrustStore, sign_node_ownership,
        verify_node_ownership,
    };
    const HOUR_MS: u64 = 60 * 60 * 1000;
    let lifetime_ms = DEFAULT_NODE_CERT_LIFETIME_SECS * 1000;
    let owner = crate::crypto::OwnerKeypair::from_bytes(&[0xc1; 32], &[0xc2; 32]).unwrap();
    let mut node = Node::new_for_tests(NodeRole::Worker)
        .await
        .expect("test node");
    node.owner_keypair = Some(owner.clone());
    let start = crate::mesh::current_time_unix_ms();
    let first = sign_node_ownership(
        &owner,
        node.id().as_bytes(),
        start + lifetime_ms,
        Some("rig".to_string()),
        Some("host-a".to_string()),
    )
    .expect("sign the startup certificate");
    *node.owner_attestation.lock().await = Some(first.clone());

    let two_days_left = start + lifetime_ms - 48 * HOUR_MS;
    assert!(
        node.renew_owner_attestation_if_due(two_days_left)
            .await
            .expect("check outside the window")
            .is_none(),
        "48 hours left is outside the 36-hour window"
    );
    assert_eq!(node.owner_attestation.lock().await.as_ref(), Some(&first));

    let one_day_left = start + lifetime_ms - 24 * HOUR_MS;
    let renewed = node
        .renew_owner_attestation_if_due(one_day_left)
        .await
        .expect("renew inside the window")
        .expect("24 hours left is inside the window");
    assert_eq!(renewed.claim.expires_at_unix_ms, one_day_left + lifetime_ms);
    assert_ne!(renewed.claim.cert_id, first.claim.cert_id);
    assert_eq!(renewed.claim.node_label.as_deref(), Some("rig"));
    assert_eq!(renewed.claim.hostname_hint.as_deref(), Some("host-a"));
    assert_eq!(node.owner_attestation.lock().await.as_ref(), Some(&renewed));

    let summary = node.owner_summary().await;
    assert_eq!(summary.status, OwnershipStatus::Verified);
    assert_eq!(
        summary.cert_id.as_deref(),
        Some(renewed.claim.cert_id.as_str())
    );
    assert_eq!(summary.expires_at_unix_ms, Some(one_day_left + lifetime_ms));

    // What a peer sees once the startup certificate has expired.
    let past_first_expiry = start + lifetime_ms + HOUR_MS;
    let peer_view = |ownership| {
        verify_node_ownership(
            Some(ownership),
            node.id().as_bytes(),
            &TrustStore::default(),
            TrustPolicy::Off,
            past_first_expiry,
        )
        .status
    };
    assert_eq!(peer_view(&first), OwnershipStatus::Expired);
    assert_eq!(peer_view(&renewed), OwnershipStatus::Verified);
}

#[tokio::test]
async fn a_node_without_an_owner_key_has_no_certificate_to_renew() {
    let node = Node::new_for_tests(NodeRole::Worker)
        .await
        .expect("test node");
    let far_future = crate::mesh::current_time_unix_ms() + 365 * 24 * 60 * 60 * 1000;
    assert!(
        node.renew_owner_attestation_if_due(far_future)
            .await
            .expect("nothing to do")
            .is_none()
    );
}

/// A failed write is retried even though the renewed certificate is no longer due.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_save_of_the_renewed_certificate_is_retried_on_the_next_heartbeat() {
    use crate::crypto::sign_node_ownership;
    const DAY_MS: u64 = 24 * 60 * 60 * 1000;
    let owner = crate::crypto::OwnerKeypair::from_bytes(&[0xd1; 32], &[0xd2; 32]).unwrap();
    let dir = tempfile::tempdir().expect("temp dir");
    // A file where the ownership file's directory should be: writes fail.
    let blocker = dir.path().join("identity");
    std::fs::write(&blocker, b"not a directory").expect("blocker");
    let ownership_file = blocker.join("node-ownership.json");
    let mut node = Node::new_for_tests(NodeRole::Worker)
        .await
        .expect("test node");
    node.owner_keypair = Some(owner.clone());
    node.owner_attestation_path = Some(ownership_file.clone());
    let due = sign_node_ownership(
        &owner,
        node.id().as_bytes(),
        crate::mesh::current_time_unix_ms() + DAY_MS,
        None,
        None,
    )
    .expect("sign a certificate a day from expiry");
    *node.owner_attestation.lock().await = Some(due);

    node.run_heartbeat_cycle(&mut Default::default()).await;
    let renewed = node
        .owner_attestation
        .lock()
        .await
        .clone()
        .expect("the node keeps a certificate");
    assert!(renewed.claim.expires_at_unix_ms > crate::mesh::current_time_unix_ms() + 6 * DAY_MS);
    assert!(!ownership_file.exists(), "the write failed");

    // The pending save must succeed without another certificate renewal.
    std::fs::remove_file(&blocker).expect("unblock");
    node.run_heartbeat_cycle(&mut Default::default()).await;
    let saved: SignedNodeOwnership = serde_json::from_str(
        &std::fs::read_to_string(&ownership_file).expect("written on the next heartbeat"),
    )
    .expect("a certificate");
    assert_eq!(saved, renewed);
    assert!(node.unsaved_owner_attestation.lock().await.is_none());

    // Once saved, a heartbeat writes nothing more.
    std::fs::remove_file(&ownership_file).expect("remove");
    node.run_heartbeat_cycle(&mut Default::default()).await;
    assert!(!ownership_file.exists());
}
