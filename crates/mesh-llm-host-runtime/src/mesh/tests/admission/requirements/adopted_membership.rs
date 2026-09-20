use super::*;

pub(crate) fn assert_expired_bootstrap_token_requires_matching_adopted_membership() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let temp = tempfile::tempdir().expect("temp home");
        let membership_file = temp.path().join("membership.json");
        let owner = requirement_policy_owner();
        let policy = requirement_policy_without_release_attestation();
        let signed_policy =
            crate::SignedMeshGenesisPolicy::sign(policy.clone(), &owner).expect("signed policy");
        let mut node = make_test_node_with_requirements(
            super::super::super::NodeRole::Worker,
            policy.requirements.clone(),
        )
        .await
        .expect("joiner node");
        node.adopted_membership_file = Some(membership_file.to_path_buf());
        let expired = crate::SignedBootstrapToken::sign(
            vec![
                serde_json::to_vec(&node.endpoint_addr_for_advertisement())
                    .expect("serializable addr"),
            ],
            &signed_policy,
            Some(current_time_unix_ms().saturating_sub(1)),
            &owner,
        )
        .expect("expired token can be constructed");

        assert_eq!(
            node.validate_bootstrap_token(&expired).await,
            Err(crate::MeshRequirementRejectReason::BootstrapTokenExpired),
            "a fresh node must not use an expired bearer token"
        );

        let adopted = crate::mesh::node::RequirementAwareMeshState {
            mesh_id: expired.mesh_id.clone(),
            policy_hash: expired.policy_hash.clone(),
            policy: policy.clone(),
            signed_policy: Some(signed_policy.clone()),
            bootstrap_token: Some(expired.clone()),
        };
        let saved_addr = node.endpoint_addr_for_advertisement();
        crate::mesh::node_requirements::persist_adopted_mesh_membership(
            &membership_file,
            &adopted,
            vec![saved_addr.clone()],
        )
        .expect("persist adopted membership");
        node.validate_bootstrap_token(&expired)
            .await
            .expect("matching adopted membership may reuse the dial target");

        assert_direct_join_restores_policy(&membership_file, &policy, &expired, &adopted).await;

        crate::mesh::node_requirements::persist_adopted_mesh_membership(
            &membership_file,
            &adopted,
            vec![saved_addr.clone()],
        )
        .expect("restore saved-target fixture");
        let mut restored = make_test_node_with_requirements(
            super::super::super::NodeRole::Worker,
            policy.requirements.clone(),
        )
        .await
        .expect("restart node");
        restored.adopted_membership_file = Some(membership_file.to_path_buf());
        let encoded_expired = super::super::super::encode_signed_bootstrap_token(&expired);
        restored
            .restore_adopted_mesh_membership(std::slice::from_ref(&encoded_expired))
            .await;
        let restored_state = restored
            .active_mesh_policy_state()
            .await
            .expect("matching token restores persisted membership");
        assert_eq!(restored_state.mesh_id, expired.mesh_id);
        assert_eq!(
            restored.join_targets.lock().await.as_slice(),
            std::slice::from_ref(&saved_addr),
            "startup restore seeds persisted dial targets"
        );

        let other_owner = crate::crypto::OwnerKeypair::generate();
        let other_policy = crate::MeshGenesisPolicy::new(
            other_owner.owner_id(),
            policy.created_at_unix_ms,
            policy.requirements.clone(),
        )
        .expect("other policy");
        let other_signed = crate::SignedMeshGenesisPolicy::sign(other_policy, &other_owner)
            .expect("other signed policy");
        let mismatched = crate::SignedBootstrapToken::sign(
            expired.serialized_addrs.clone(),
            &other_signed,
            Some(current_time_unix_ms().saturating_sub(1)),
            &other_owner,
        )
        .expect("mismatched expired token");
        assert_eq!(
            node.validate_bootstrap_token(&mismatched).await,
            Err(crate::MeshRequirementRejectReason::MeshPolicyMismatch)
        );
        let fresh_other = crate::SignedBootstrapToken::sign(
            mismatched.serialized_addrs.clone(),
            &other_signed,
            Some(current_time_unix_ms().saturating_add(60_000)),
            &other_owner,
        )
        .expect("fresh other-mesh token");
        let fresh_other_encoded = super::super::super::encode_signed_bootstrap_token(&fresh_other);
        let mut switching = make_test_node_with_requirements(
            super::super::super::NodeRole::Worker,
            policy.requirements.clone(),
        )
        .await
        .expect("switching node");
        switching.adopted_membership_file = Some(membership_file.to_path_buf());
        switching
            .restore_adopted_mesh_membership(std::slice::from_ref(&fresh_other_encoded))
            .await;
        assert!(
            switching.active_mesh_policy_state().await.is_none(),
            "a persisted mesh must not block a fresh invite for another mesh"
        );
        switching
            .validate_bootstrap_token(&fresh_other)
            .await
            .expect("fresh other-mesh token remains valid");

        let mut tampered = expired;
        tampered.signature[0] ^= 1;
        let encoded = super::super::super::encode_signed_bootstrap_token(&tampered);
        assert!(!switching.restore_adopted_mesh_membership(&[encoded]).await);
        assert!(switching.active_mesh_policy_state().await.is_none());
        assert!(switching.join_targets.lock().await.is_empty());
        assert_eq!(
            node.validate_bootstrap_token(&tampered).await,
            Err(crate::MeshRequirementRejectReason::BootstrapTokenInvalid)
        );
    });
}

pub(crate) fn assert_fresh_single_invite_with_persisted_membership_joins_new_mesh() {
    // Regression for the corrected tray flow: "Join with an invite" replaces the
    // previous selection, so the engine receives a SINGLE fresh invite for a new
    // mesh (B) while a previously adopted mesh (A) is still persisted on disk.
    // Persisted membership A must not pin the node — neither the eager
    // pre-attempt restore nor the actual prepare/join boundary may leave A
    // installed — and the fresh invite for B must be the mesh that gets
    // installed. This is checked through prepare_join_target (via join), not just
    // validate_bootstrap_token, and for both restricted and unrestricted local
    // requirements because install_requirement_mesh_state_transition rejects a
    // differing installed policy regardless of local restrictions.
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        for local_requirements in [
            requirement_policy_without_release_attestation().requirements,
            crate::MeshRequirements::unrestricted(),
        ] {
            let temp = tempfile::tempdir().expect("temp home");
            let membership_file = temp.path().join("membership.json");

            // Persist adopted membership for the OLD mesh (A).
            let owner = requirement_policy_owner();
            let policy = requirement_policy_without_release_attestation();
            let signed_policy = crate::SignedMeshGenesisPolicy::sign(policy.clone(), &owner)
                .expect("signed policy");
            let seed_node = make_test_node_with_requirements(
                super::super::super::NodeRole::Worker,
                policy.requirements.clone(),
            )
            .await
            .expect("seed node");
            let expired_old = crate::SignedBootstrapToken::sign(
                vec![
                    serde_json::to_vec(&seed_node.endpoint_addr_for_advertisement())
                        .expect("serializable addr"),
                ],
                &signed_policy,
                Some(current_time_unix_ms().saturating_sub(1)),
                &owner,
            )
            .expect("expired old token");
            let adopted = crate::mesh::node::RequirementAwareMeshState {
                mesh_id: expired_old.mesh_id.clone(),
                policy_hash: expired_old.policy_hash.clone(),
                policy: policy.clone(),
                signed_policy: Some(signed_policy.clone()),
                bootstrap_token: Some(expired_old.clone()),
            };
            crate::mesh::node_requirements::persist_adopted_mesh_membership(
            &membership_file,
                &adopted,
                vec![seed_node.endpoint_addr_for_advertisement()],
            )
            .expect("persist adopted membership");

            // Fresh, currently-valid single invite for a DIFFERENT mesh (B).
            let other_owner = crate::crypto::OwnerKeypair::generate();
            let other_policy = crate::MeshGenesisPolicy::new(
                other_owner.owner_id(),
                policy.created_at_unix_ms,
                policy.requirements.clone(),
            )
            .expect("other policy");
            let other_signed = crate::SignedMeshGenesisPolicy::sign(other_policy, &other_owner)
                .expect("other signed policy");
            let fresh_new = crate::SignedBootstrapToken::sign(
                expired_old.serialized_addrs.clone(),
                &other_signed,
                Some(current_time_unix_ms().saturating_add(60_000)),
                &other_owner,
            )
            .expect("fresh new-mesh token");
            let fresh_new_encoded = super::super::super::encode_signed_bootstrap_token(&fresh_new);

            let mut node = make_test_node_with_requirements(
                super::super::super::NodeRole::Worker,
                local_requirements.clone(),
            )
            .await
            .expect("joiner node");
        node.adopted_membership_file = Some(membership_file.to_path_buf());

            // The eager pre-attempt restore over the single fresh invite must not
            // adopt A (the invite does not match the persisted membership).
            assert!(
                !node
                    .restore_adopted_mesh_membership(std::slice::from_ref(&fresh_new_encoded))
                    .await,
                "a single fresh invite for another mesh must not eagerly restore the old mesh"
            );
            assert!(
                node.active_mesh_policy_state().await.is_none(),
                "persisted membership must not be installed before the fresh invite is attempted"
            );

            // Drive the real join/prepare boundary. Poll only through preparation
            // and the first pending dial so state is installed without requiring a
            // live peer.
            let attempt = node.join(&fresh_new_encoded);
            tokio::pin!(attempt);
            let _ = futures_util::poll!(attempt);

            let active = node
                .active_mesh_policy_state()
                .await
                .expect("prepare_join_target must install the fresh mesh");
            assert_eq!(
                active.mesh_id, fresh_new.mesh_id,
                "the fresh single invite (mesh B) must be the installed mesh, not persisted A"
            );
        }
    });
}

async fn assert_direct_join_restores_policy(
    membership_file: &std::path::Path,
    policy: &crate::MeshGenesisPolicy,
    token: &crate::SignedBootstrapToken,
    adopted: &crate::mesh::node::RequirementAwareMeshState,
) {
    for retry in [false, true] {
        let mut node = make_test_node_with_requirements(
            super::super::super::NodeRole::Worker,
            policy.requirements.clone(),
        )
        .await
        .expect("direct joiner");
        node.adopted_membership_file = Some(membership_file.to_path_buf());
        crate::mesh::node_requirements::persist_adopted_mesh_membership(membership_file, adopted, vec![])
            .expect("reset persisted fixture for each independent join");
        let encoded = super::super::super::encode_signed_bootstrap_token(token);
        let attempt = async {
            if retry {
                node.join_with_retry(&encoded).await
            } else {
                node.join(&encoded).await
            }
        };
        // Poll only through preparation and the first pending dial. Do not allow
        // unrelated background gossip to change the persisted fixture between cases.
        tokio::pin!(attempt);
        let result = futures_util::poll!(attempt);
        assert_eq!(
            node.active_mesh_policy_state()
                .await
                .unwrap_or_else(|| panic!("policy before dial: retry={retry}, result={result:?}"))
                .mesh_id,
            token.mesh_id
        );
    }
}

#[test]
fn accepted_peer_is_persisted_before_promotion() {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let temp = tempfile::tempdir().unwrap();
        let membership_file = temp.path().join("membership.json");
        let policy = requirement_policy_without_release_attestation();
        let signed =
            crate::SignedMeshGenesisPolicy::sign(policy.clone(), &requirement_policy_owner())
                .unwrap();
        let host = make_test_node(super::super::super::NodeRole::Worker)
            .await
            .unwrap();
        let mut joiner = make_test_node(super::super::super::NodeRole::Worker)
            .await
            .unwrap();
        joiner.adopted_membership_file = Some(membership_file.to_path_buf());
        for node in [&host, &joiner] {
            node.install_requirement_aware_mesh_state(
                policy.policy_derived_mesh_id().unwrap(),
                policy.canonical_hash_hex().unwrap(),
                policy.clone(),
                Some(signed.clone()),
                None,
            )
            .await
            .unwrap();
        }
        let ann = host
            .collect_announcements()
            .await
            .into_iter()
            .find(|ann| ann.addr.id == host.id())
            .unwrap();
        assert!(joiner.peers().await.is_empty());
        joiner
            .validate_direct_peer_requirements(host.id(), &ann, Some(NODE_PROTOCOL_GENERATION))
            .await
            .unwrap();
        assert!(
            joiner.peers().await.is_empty(),
            "validation precedes promotion"
        );
        let bytes = std::fs::read(
            &membership_file,
        )
        .unwrap();
        let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let peers: Vec<EndpointAddr> = serde_json::from_value(saved["peer_addrs"].clone()).unwrap();
        assert_eq!(peers, vec![ann.addr]);
    });
}
