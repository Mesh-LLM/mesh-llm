//! Installable process identity RPC conformance over authenticated plugin transport.

use super::lifecycle_live_tests::LiveHost;
use mesh_llm_identity::plugin_delegation::{
    DelegationVerification, IdentityEvidenceStatus, PluginSigningBinding, PluginSigningDelegation,
    PluginSigningScope, verify_plugin_delegation,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const NAME: &str = "openai-exchange-observer";

async fn probe(host: &LiveHost, method: &str, params: Value) -> Value {
    let result = Box::pin(host.manager.invoke_operation(
        NAME,
        "identity_probe",
        &json!({"method":method,"params":params}).to_string(),
    ))
    .await
    .unwrap();
    assert!(!result.is_error, "{}", result.content_json);
    serde_json::from_str(&result.content_json).unwrap()
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_identity_services_require_host_grants_and_bind_live_delegation() {
    let ordinary = LiveHost::start(false, false).await;
    assert!(
        probe(&ordinary, "ReadIdentityBundle", json!({}))
            .await
            .get("error")
            .is_some()
    );
    assert!(
        probe(&ordinary, "DelegatePluginSigningKey", json!({}))
            .await
            .get("error")
            .is_some()
    );
    ordinary.stop().await;

    let host = LiveHost::start_with_identity(false, false, true, true).await;
    let bundle = probe(&host, "ReadIdentityBundle", json!({})).await["response"].clone();
    assert_eq!(bundle["status"], "verified");
    assert_eq!(bundle["evidence_kind"], "software_identity");
    assert_eq!(bundle["signing_delegations"]["status"], "missing");
    let artifact_sha256 = hex::encode(Sha256::digest(
        std::fs::read(host.installed_metadata.executable_path()).unwrap(),
    ));
    assert_eq!(bundle["plugin_artifact_sha256"], artifact_sha256);
    let issued = probe(&host, "DelegatePluginSigningKey", json!({})).await;
    let delegation: PluginSigningDelegation =
        serde_json::from_value(issued["response"].clone()).unwrap();
    let cached = probe(&host, "DelegatePluginSigningKey", json!({})).await;
    assert_eq!(
        cached["response"], issued["response"],
        "repeated setup RPC created a fresh owner signature"
    );
    let binding = PluginSigningBinding {
        plugin_id: NAME.into(),
        artifact_sha256,
        signing_public_key: issued["plugin_signing_public_key"].as_str().unwrap().into(),
        scope: PluginSigningScope::OpenAiExchangeEvidence,
    };
    let node_id = hex::encode(host.node.id().as_bytes());
    let owner_id = host.node.owner_keypair.as_ref().unwrap().owner_id();
    let certificate_id = bundle["node_ownership"]["claim"]["cert_id"]
        .as_str()
        .unwrap();
    let context = DelegationVerification {
        owner_id: &owner_id,
        node_endpoint_id: &node_id,
        node_certificate_id: certificate_id,
        plugin: &binding,
        identity_status: IdentityEvidenceStatus::Verified,
        revoked_delegation_ids: &[],
        now_unix_ms: chrono::Utc::now().timestamp_millis().unsigned_abs(),
    };
    assert_eq!(
        verify_plugin_delegation(Some(&delegation), &context),
        IdentityEvidenceStatus::Verified
    );
    assert!(
        probe(
            &host,
            "DelegatePluginSigningKey",
            json!({"lifetime_ms":60_001})
        )
        .await
        .get("error")
        .is_some()
    );
    assert!(
        probe(
            &host,
            "DelegatePluginSigningKey",
            json!({"scope":"arbitrary.owner.sign"})
        )
        .await
        .get("error")
        .is_some()
    );
    assert!(
        probe(
            &host,
            "DelegatePluginSigningKey",
            json!({"claim":{"owner_id":"attacker"}})
        )
        .await
        .get("error")
        .is_some()
    );
    let executable = host.installed_metadata.executable_path();
    let replacement = host.root.path().join("replacement-artifact");
    std::fs::write(&replacement, b"changed installed artifact").unwrap();
    std::fs::rename(&replacement, &executable).unwrap();
    let changed = probe(&host, "ReadIdentityBundle", json!({})).await["response"].clone();
    assert_eq!(changed["plugin_artifact_status"], "invalid");
    assert_eq!(
        changed["plugin_artifact_sha256"],
        hex::encode(Sha256::digest(b"changed installed artifact"))
    );
    assert_eq!(changed["signing_delegations"]["status"], "revoked");
    assert!(
        changed["signing_delegations"]["revoked_delegation_ids"]
            .as_array()
            .unwrap()
            .contains(&json!(delegation.claim.delegation_id))
    );
    assert!(
        probe(&host, "DelegatePluginSigningKey", json!({}))
            .await
            .get("error")
            .is_some()
    );
    let mut grant = host.manager.effective_exchange_grant(NAME).unwrap();
    grant.delegate_signing_key = false;
    grant.signing_scopes.clear();
    grant.max_delegation_ttl_secs = 0;
    let config = mesh_llm_config::MeshConfig {
        plugins: vec![
            serde_json::from_value(json!({"name":NAME,"openai_exchange_grant":grant})).unwrap(),
        ],
        ..Default::default()
    };
    host.manager.apply_exchange_grants(&config).await;
    assert!(
        probe(&host, "DelegatePluginSigningKey", json!({}))
            .await
            .get("error")
            .is_some()
    );
    let revoked = probe(&host, "ReadIdentityBundle", json!({})).await["response"].clone();
    assert_eq!(revoked["status"], "verified");
    assert_eq!(revoked["signing_delegations"]["status"], "revoked");
    let revoked_ids: Vec<String> =
        serde_json::from_value(revoked["signing_delegations"]["revoked_delegation_ids"].clone())
            .unwrap();
    let revoked_context = DelegationVerification {
        revoked_delegation_ids: &revoked_ids,
        ..context
    };
    assert_eq!(
        verify_plugin_delegation(Some(&delegation), &revoked_context),
        IdentityEvidenceStatus::Revoked
    );
    host.stop().await;
}
