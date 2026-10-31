//! Originator startup must not bypass a persisted strict genesis policy.

use super::*;

#[test]
#[serial_test::serial]
fn membership_creation_preserves_the_primary_nodes_strict_genesis() {
    let temp = tempfile::tempdir().expect("temp home");
    let _home = HomeGuard::set(temp.path());
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let mut origin = make_test_node(super::super::super::NodeRole::Worker)
            .await
            .expect("origin");
        origin.owner_keypair = Some(requirement_policy_owner());
        origin.local_mesh_requirements.require_signed_gossip = true;
        let mesh_id = origin
            .initialize_mesh_identity_as_originator(None, None)
            .await
            .expect("create strict mesh");
        let path = temp.path().join(".mesh-llm/mesh-genesis-policy.json");
        let persisted = std::fs::read(&path).expect("persisted policy");
        let owners = vec![requirement_policy_owner().owner_id()];
        let (membership, _channels) = origin
            .start_membership_node(crate::mesh::node::MembershipNodeConfig {
                mesh_name: "independent-team",
                secret_key: SecretKey::generate(),
                relay: RelayConfig {
                    urls: &[],
                    auths: &HashMap::new(),
                    policy: RelayPolicy::ExplicitlyDisabled,
                },
                quic_port: None,
                trusted_owners: &owners,
                enumerate_host: false,
            })
            .await
            .expect("a membership must not read its host's genesis policy");
        assert!(!membership.requires_signed_gossip().await);
        assert!(origin.requires_signed_gossip().await);
        assert_eq!(origin.mesh_id().await.as_deref(), Some(mesh_id.as_str()));
        assert_ne!(
            membership.mesh_id().await.as_deref(),
            Some(mesh_id.as_str())
        );
        assert_eq!(
            crate::mesh::node::membership_token_endpoint(&membership.invite_token().await)
                .expect("membership keeps its unsigned invitation"),
            membership.id()
        );
        assert_eq!(std::fs::read(&path).unwrap(), persisted);
        membership.close_endpoint().await;
        origin.close_endpoint().await;
    });
}

#[test]
#[serial_test::serial]
fn strict_genesis_restart_preserves_policy_and_rejects_downgrade() {
    let temp = tempfile::tempdir().expect("temp home");
    let _home = HomeGuard::set(temp.path());
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let mut origin = make_test_node(super::super::super::NodeRole::Worker)
            .await
            .expect("origin");
        origin.owner_keypair = Some(requirement_policy_owner());
        origin.local_mesh_requirements.require_signed_gossip = true;
        let mesh_id = origin
            .initialize_mesh_identity_as_originator(None, None)
            .await
            .expect("create strict mesh");
        let path = temp.path().join(".mesh-llm/mesh-genesis-policy.json");
        let persisted = std::fs::read(&path).expect("persisted policy");
        origin.close_endpoint().await;

        let mut restarted = make_test_node(super::super::super::NodeRole::Worker)
            .await
            .expect("restart");
        restarted.owner_keypair = Some(requirement_policy_owner());
        // Reproduces the normal startup path when the creation flag/config
        // disappears. Calling only load_or_create_signed_genesis_policy would
        // miss the unrestricted-originator early return that caused this bug.
        let error = restarted
            .initialize_mesh_identity_as_originator(None, None)
            .await
            .expect_err("strict identity cannot become an unrestricted mesh");
        assert!(error.to_string().contains("requires signed gossip"));
        assert!(restarted.requirement_mesh_state.lock().await.is_none());
        assert_eq!(std::fs::read(&path).unwrap(), persisted);

        restarted.local_mesh_requirements.require_signed_gossip = true;
        assert_eq!(
            restarted
                .initialize_mesh_identity_as_originator(None, None)
                .await
                .expect("restart with original requirement"),
            mesh_id
        );
        assert!(restarted.requires_signed_gossip().await);
        assert_eq!(std::fs::read(&path).unwrap(), persisted);
        restarted.close_endpoint().await;
    });
}

#[test]
#[serial_test::serial]
fn unrestricted_originator_without_genesis_keeps_legacy_identity() {
    let temp = tempfile::tempdir().expect("temp home");
    let _home = HomeGuard::set(temp.path());
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let node = make_test_node(super::super::super::NodeRole::Worker)
            .await
            .expect("origin");
        let first = node
            .initialize_mesh_identity_as_originator(None, None)
            .await
            .expect("legacy creation");
        assert_eq!(
            node.initialize_mesh_identity_as_originator(None, None)
                .await
                .expect("legacy restart"),
            first
        );
        assert!(!node.requires_signed_gossip().await);
        assert!(
            !temp
                .path()
                .join(".mesh-llm/mesh-genesis-policy.json")
                .exists()
        );
        node.close_endpoint().await;
    });
}
