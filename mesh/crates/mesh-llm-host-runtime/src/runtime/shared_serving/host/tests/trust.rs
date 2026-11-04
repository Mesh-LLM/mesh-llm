use super::*;
use crate::crypto::{
    OwnerKeypair, TrustPolicy, TrustStore, save_trust_store, sign_node_ownership,
    verify_node_ownership,
};

/// Install a short-lived, verified ownership attestation on a test endpoint.
async fn attest(node: &Node, owner: &OwnerKeypair) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let attestation =
        sign_node_ownership(owner, node.id().as_bytes(), now + 60_000, None, None).unwrap();
    let summary = verify_node_ownership(
        Some(&attestation),
        node.id().as_bytes(),
        &TrustStore::default(),
        TrustPolicy::Off,
        now,
    );
    assert!(summary.verified);
    *node.owner_attestation.lock().await = Some(attestation);
    *node.owner_summary.lock().await = summary;
}

#[tokio::test]
/// A trusted owner in one mesh must not gain another mesh's admission or host control.
async fn allowlist_and_owner_control_authority_remain_membership_local() {
    let mut fixture = Fixture::new().await;
    fixture.host.shutdown().await.unwrap();
    let owner = OwnerKeypair::from_bytes(&[0x41; 32], &[0x42; 32]).unwrap();
    for seed in &fixture.seeds {
        attest(seed, &owner).await;
    }
    let mut config = fixture.host.config.clone();
    config.state_dir = fixture.directory.path().join("strict-host");
    for member in &mut config.memberships {
        let mut trust = TrustStore {
            policy: TrustPolicy::Allowlist,
            ..Default::default()
        };
        // The same signed owner is allowed only in alpha. Loading alpha's
        // trust must never admit it into beta through a shared global store.
        if member.name == "alpha" {
            trust.add_trusted_owner(owner.owner_id(), None);
        }
        let path = fixture
            .directory
            .path()
            .join(format!("{}.trust", member.name));
        save_trust_store(&path, &trust).unwrap();
        member.trust_store = Some(path);
    }
    config.validate().unwrap();
    let lock = lock_state(&config).unwrap();
    fixture.host = SharedServingHost::assemble(
        config,
        fixture.host.descriptor.clone(),
        fixture.backend.clone(),
        None,
        lock,
    )
    .unwrap();
    fixture.host.reconnect("alpha").await.unwrap();
    assert!(fixture.host.reconnect("beta").await.is_err());
    assert!(
        fixture
            .request(0, "alpha")
            .await
            .starts_with("HTTP/1.1 200")
    );
    let status = fixture.host.status().await;
    assert!(matches!(
        status.memberships[0].phase,
        MembershipPhase::Connected
    ));
    assert!(matches!(
        status.memberships[1].phase,
        MembershipPhase::Failed
    ));
    let record = fixture.host.record("alpha").unwrap();
    let attachment = record.attachment.lock().await;
    let node = &attachment.as_ref().unwrap().node;
    attest(node, &owner).await;
    assert_eq!(node.local_verified_owner_id().await, Some(owner.owner_id()));
    node.maybe_start_control_listener(node.endpoint_secret_key.clone(), None, None)
        .await
        .unwrap();
    assert!(node.control_listener.lock().await.is_none());
    drop(attachment);
    fixture.close().await;
}
