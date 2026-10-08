use super::*;
use mesh_llm_identity::plugin_delegation::{
    DelegationVerification, PluginSigningScope, verify_plugin_delegation,
};
use mesh_llm_identity::sign_node_ownership;
fn bundle_status(
    node: &[u8; 32],
    trust: &TrustStore,
    ownership: Option<&SignedNodeOwnership>,
    now_unix_ms: u64,
) -> IdentityEvidenceStatus {
    PluginIdentitySnapshot {
        node_endpoint_id: node,
        ownership,
        trust_store: trust,
        now_unix_ms,
    }
    .read_identity_bundle(&PluginIdentityGrants {
        read_identity_bundle: true,
        ..Default::default()
    })
    .unwrap()
    .status
}

#[test]
fn denies_ordinary_plugin_and_constrains_issuance_to_verified_host_registration() {
    let owner = OwnerKeypair::generate();
    let node = [1; 32];
    let now = chrono::Utc::now().timestamp_millis().unsigned_abs();
    let certificate = sign_node_ownership(&owner, &node, now + 60_000, None, None).unwrap();
    let trust = TrustStore::default();
    let snapshot = PluginIdentitySnapshot {
        node_endpoint_id: &node,
        ownership: Some(&certificate),
        trust_store: &trust,
        now_unix_ms: now,
    };
    let registration = PluginSigningBinding {
        plugin_id: "observer".into(),
        artifact_sha256: hex::encode([2; 32]),
        signing_public_key: hex::encode(OwnerKeypair::generate().verifying_key().as_bytes()),
        scope: PluginSigningScope::OpenAiExchangeEvidence,
    };
    let request = DelegatePluginSigningKeyRequest {
        lifetime_ms: 120_000,
        signing_public_key: registration.signing_public_key.clone(),
        scope: registration.scope,
    };
    let grants = PluginIdentityGrants {
        read_identity_bundle: true,
        delegate_signing_key: true,
        max_delegation_lifetime_ms: 120_000,
    };
    let too_short = DelegatePluginSigningKeyRequest {
        lifetime_ms: 999,
        ..request.clone()
    };
    assert!(
        snapshot
            .delegation_claim(&grants, &registration, &owner, &too_short)
            .is_err()
    );
    assert!(
        snapshot
            .read_identity_bundle(&PluginIdentityGrants::default())
            .is_err()
    );
    assert!(
        snapshot
            .delegate_plugin_signing_key(
                &PluginIdentityGrants::default(),
                &registration,
                &owner,
                &request
            )
            .is_err()
    );
    let bundle = snapshot.read_identity_bundle(&grants).unwrap();
    assert_eq!(bundle.status, IdentityEvidenceStatus::Verified);
    assert_eq!(bundle.evidence_kind, "software_identity");
    let delegation = snapshot
        .delegate_plugin_signing_key(&grants, &registration, &owner, &request)
        .unwrap();
    assert_eq!(
        delegation.claim.expires_at_unix_ms,
        certificate.claim.expires_at_unix_ms
    );
    let context = DelegationVerification {
        owner_id: &certificate.claim.owner_id,
        node_endpoint_id: &certificate.claim.node_endpoint_id,
        node_certificate_id: &certificate.claim.cert_id,
        plugin: &registration,
        identity_status: bundle.status,
        revoked_delegation_ids: &[],
        now_unix_ms: now,
    };
    assert_eq!(
        verify_plugin_delegation(Some(&delegation), &context),
        IdentityEvidenceStatus::Verified
    );
    assert!(
        snapshot
            .delegate_plugin_signing_key(
                &grants,
                &registration,
                &OwnerKeypair::generate(),
                &request
            )
            .is_err()
    );
    let oversized = DelegatePluginSigningKeyRequest {
        lifetime_ms: MAX_DELEGATION_LIFETIME_MS + 1,
        ..request.clone()
    };
    assert!(
        snapshot
            .delegate_plugin_signing_key(&grants, &registration, &owner, &oversized)
            .is_err()
    );
}

#[test]
fn public_bundle_distinguishes_missing_unverified_expired_and_revoked() {
    let owner = OwnerKeypair::generate();
    let node = [9; 32];
    let now = chrono::Utc::now().timestamp_millis().unsigned_abs();
    let certificate = sign_node_ownership(&owner, &node, now + 60_000, None, None).unwrap();
    let mut trust = TrustStore::default();
    assert_eq!(
        bundle_status(&node, &trust, None, now),
        IdentityEvidenceStatus::Missing
    );
    trust.policy = mesh_llm_identity::TrustPolicy::Allowlist;
    assert_eq!(
        bundle_status(&node, &trust, Some(&certificate), now),
        IdentityEvidenceStatus::Unverified
    );
    trust.policy = mesh_llm_identity::TrustPolicy::Off;
    assert_eq!(
        bundle_status(&node, &trust, Some(&certificate), now + 60_000),
        IdentityEvidenceStatus::Expired
    );
    trust.revoke_node_cert(certificate.claim.cert_id.clone(), None);
    assert_eq!(
        bundle_status(&node, &trust, Some(&certificate), now),
        IdentityEvidenceStatus::Revoked
    );
    let mut invalid = certificate;
    invalid.signature = hex::encode([0; 64]);
    assert_eq!(
        bundle_status(&node, &trust, Some(&invalid), now),
        IdentityEvidenceStatus::Invalid
    );
}

#[test]
fn request_shape_is_rejected_before_artifact_work() {
    for params in [
        serde_json::json!({"lifetime_ms":999,"signing_public_key":"00".repeat(32),"scope":"mesh.openai.exchange.evidence.sign.v1"}),
        serde_json::json!({"lifetime_ms":1000,"signing_public_key":"bad","scope":"mesh.openai.exchange.evidence.sign.v1"}),
        serde_json::json!({"lifetime_ms":1000,"signing_public_key":"00".repeat(32),"scope":"owner.sign.arbitrary"}),
        serde_json::json!({"lifetime_ms":1000,"signing_public_key":"00".repeat(32),"scope":"mesh.openai.exchange.evidence.sign.v1","claim":{}}),
    ] {
        assert!(
            validate_identity_input(&super::super::proto::RpcRequest {
                method: "DelegatePluginSigningKey".into(),
                params_json: params.to_string()
            })
            .is_err()
        );
    }
}

#[test]
fn early_signing_key_gate_matches_canonical_ed25519_validation() {
    let valid = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    let invalid_point = (0u8..=255)
        .map(|byte| [byte; 32])
        .find(|bytes| ed25519_dalek::VerifyingKey::from_bytes(bytes).is_err())
        .expect("non-point encoding");
    for (key, expected) in [
        (valid.to_owned(), true),
        (valid.to_uppercase(), false),
        (hex::encode([0; 32]), false),
        (format!("01{}", "00".repeat(31)), false),
        (hex::encode(invalid_point), false),
        (valid[..62].to_owned(), false),
    ] {
        let request = super::super::proto::RpcRequest {
            method: "DelegatePluginSigningKey".into(),
            params_json: serde_json::json!({"lifetime_ms":1000,"signing_public_key":key,"scope":"mesh.openai.exchange.evidence.sign.v1"}).to_string(),
        };
        let result = validate_identity_input(&request);
        assert_eq!(result.is_ok(), expected);
        if let Err(error) = result {
            assert!(
                error
                    .to_string()
                    .contains("64 lowercase hexadecimal characters")
            );
        }
    }
}

#[tokio::test]
async fn callback_renewal_gate_is_per_authenticated_plugin_and_precedes_artifact_work() {
    let manager = super::super::PluginManager::for_test_summaries(Vec::new());
    manager.set_test_exchange_callback_active("active-observer", true);
    let node = Box::pin(crate::mesh::Node::new_for_tests(
        crate::mesh::NodeRole::Worker,
    ))
    .await
    .unwrap();
    node.set_plugin_manager(manager.clone()).await;
    let delegate=super::super::proto::RpcRequest {
        method:"DelegatePluginSigningKey".into(),
        params_json:serde_json::json!({"lifetime_ms":60_000,"signing_public_key":hex::encode(OwnerKeypair::generate().verifying_key().as_bytes()),"scope":"mesh.openai.exchange.evidence.sign.v1"}).to_string(),
    };
    let error = handle_request(&node, "active-observer", delegate.clone())
        .await
        .unwrap_err();
    assert!(error.message.contains("outside lifecycle callbacks"));
    // This fixture has no artifact registration. Its different error proves
    // unrelated background renewal and public reads pass the callback gate.
    let other = handle_request(&node, "setup-observer", delegate.clone())
        .await
        .unwrap_err();
    assert!(
        other
            .message
            .contains("authenticated plugin registration unavailable")
    );
    let read = handle_request(
        &node,
        "active-observer",
        super::super::proto::RpcRequest {
            method: "ReadIdentityBundle".into(),
            params_json: "{}".into(),
        },
    )
    .await
    .unwrap_err();
    assert!(
        read.message
            .contains("authenticated plugin registration unavailable")
    );
    manager.set_test_exchange_callback_active("active-observer", false);
    let setup = handle_request(&node, "active-observer", delegate)
        .await
        .unwrap_err();
    assert!(
        setup
            .message
            .contains("authenticated plugin registration unavailable")
    );
    node.take_plugin_manager().await;
    node.endpoint.close().await;
}
