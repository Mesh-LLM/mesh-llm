//! Permissioned public identity and fixed-shape plugin signing delegation.
//!
//! Callers must resolve registration from the authenticated plugin connection.
//! Renewal happens during plugin setup, never during an inference request.

use anyhow::{Result, bail};
use mesh_llm_identity::plugin_delegation::{
    IdentityEvidenceStatus, MAX_DELEGATION_LIFETIME_MS, PLUGIN_DELEGATION_VERSION,
    PluginSigningBinding, PluginSigningDelegationClaim, PluginSigningScope, sign_plugin_delegation,
};
use mesh_llm_identity::{
    OwnerKeypair, OwnershipStatus, SignedNodeOwnership, TrustStore, verify_node_ownership,
};
use serde::{Deserialize, Serialize};

/// Public data only. This is software identity evidence, not execution attestation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicIdentityBundle {
    pub node_endpoint_id: String,
    pub status: IdentityEvidenceStatus,
    pub node_ownership: Option<SignedNodeOwnership>,
    pub evidence_kind: String,
    pub owner_sign_public_key: Option<String>,
    pub host_version: String,
    pub release_attestation: Option<crate::ReleaseBuildAttestation>,
    pub release_attestation_summary: Option<crate::ReleaseAttestationSummary>,
    pub plugin_id: Option<String>,
    pub plugin_artifact_sha256: Option<String>,
    pub plugin_artifact_status: IdentityEvidenceStatus,
    pub plugin_artifact_metadata:
        Option<super::identity_registration::PublicPluginArtifactMetadata>,
    pub signing_delegations: super::identity_registry::PluginDelegationState,
}

/// Grants recorded by the host for an authenticated installed plugin.
#[derive(Debug, Clone, Default)]
pub struct PluginIdentityGrants {
    pub read_identity_bundle: bool,
    pub delegate_signing_key: bool,
    pub max_delegation_lifetime_ms: u64,
}

/// Plugins supply their public key, a registered scope, and a duration. Owner,
/// node, certificate and artifact claims are resolved by the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelegatePluginSigningKeyRequest {
    pub lifetime_ms: u64,
    pub signing_public_key: String,
    pub scope: PluginSigningScope,
}

/// An identity snapshot refreshed outside inference. It contains no keys.
pub struct PluginIdentitySnapshot<'a> {
    pub node_endpoint_id: &'a [u8; 32],
    pub ownership: Option<&'a SignedNodeOwnership>,
    pub trust_store: &'a TrustStore,
    pub now_unix_ms: u64,
}

impl PluginIdentitySnapshot<'_> {
    pub fn status(&self) -> IdentityEvidenceStatus {
        match verify_node_ownership(
            self.ownership,
            self.node_endpoint_id,
            self.trust_store,
            self.trust_store.policy,
            self.now_unix_ms,
        )
        .status
        {
            OwnershipStatus::Verified => IdentityEvidenceStatus::Verified,
            OwnershipStatus::Unsigned => IdentityEvidenceStatus::Missing,
            OwnershipStatus::Expired => IdentityEvidenceStatus::Expired,
            OwnershipStatus::RevokedOwner
            | OwnershipStatus::RevokedCert
            | OwnershipStatus::RevokedNodeId => IdentityEvidenceStatus::Revoked,
            OwnershipStatus::UntrustedOwner => IdentityEvidenceStatus::Unverified,
            _ => IdentityEvidenceStatus::Invalid,
        }
    }

    pub fn read_identity_bundle(
        &self,
        grants: &PluginIdentityGrants,
    ) -> Result<PublicIdentityBundle> {
        if !grants.read_identity_bundle {
            bail!("ReadIdentityBundle requires an explicit host grant");
        }
        Ok(PublicIdentityBundle {
            node_endpoint_id: hex::encode(self.node_endpoint_id),
            status: self.status(),
            node_ownership: self.ownership.cloned(),
            evidence_kind: "software_identity".into(),
            owner_sign_public_key: self
                .ownership
                .map(|ownership| ownership.claim.owner_sign_public_key.clone()),
            host_version: env!("CARGO_PKG_VERSION").into(),
            release_attestation: None,
            release_attestation_summary: None,
            plugin_id: None,
            plugin_artifact_sha256: None,
            plugin_artifact_status: IdentityEvidenceStatus::Missing,
            plugin_artifact_metadata: None,
            signing_delegations: Default::default(),
        })
    }

    /// Owner keys are provided only after non-interactive setup succeeds. This
    /// helper never loads a keystore, prompts, or returns private key material.
    #[cfg(test)]
    pub fn delegate_plugin_signing_key(
        &self,
        grants: &PluginIdentityGrants,
        registration: &PluginSigningBinding,
        owner: &OwnerKeypair,
        request: &DelegatePluginSigningKeyRequest,
    ) -> Result<mesh_llm_identity::plugin_delegation::PluginSigningDelegation> {
        Ok(sign_plugin_delegation(
            owner,
            self.delegation_claim(grants, registration, owner, request)?,
        )?)
    }

    fn delegation_claim(
        &self,
        grants: &PluginIdentityGrants,
        registration: &PluginSigningBinding,
        owner: &OwnerKeypair,
        request: &DelegatePluginSigningKeyRequest,
    ) -> Result<PluginSigningDelegationClaim> {
        if !grants.delegate_signing_key {
            bail!("DelegatePluginSigningKey requires an explicit host grant");
        }
        if self.status() != IdentityEvidenceStatus::Verified {
            bail!("DelegatePluginSigningKey requires verified current node ownership");
        }
        let ownership = self
            .ownership
            .ok_or_else(|| anyhow::anyhow!("node ownership missing"))?;
        if owner.owner_id() != ownership.claim.owner_id {
            bail!("owner signing key differs from current node ownership");
        }
        if request.lifetime_ms < 1000
            || request.lifetime_ms > grants.max_delegation_lifetime_ms
            || request.lifetime_ms > MAX_DELEGATION_LIFETIME_MS
        {
            bail!("delegation lifetime exceeds the host grant or protocol limit");
        }
        let requested_expiry = self
            .now_unix_ms
            .checked_add(request.lifetime_ms)
            .ok_or_else(|| anyhow::anyhow!("delegation expiry overflow"))?;
        let expires_at_unix_ms = requested_expiry.min(ownership.claim.expires_at_unix_ms);
        if expires_at_unix_ms.saturating_sub(self.now_unix_ms) < 1000 {
            bail!("node ownership expires before the minimum delegation lifetime");
        }
        Ok(PluginSigningDelegationClaim {
            version: PLUGIN_DELEGATION_VERSION,
            delegation_id: uuid::Uuid::new_v4().to_string(),
            owner_id: ownership.claim.owner_id.clone(),
            owner_sign_public_key: ownership.claim.owner_sign_public_key.clone(),
            node_endpoint_id: hex::encode(self.node_endpoint_id),
            node_certificate_id: ownership.claim.cert_id.clone(),
            plugin: registration.clone(),
            issued_at_unix_ms: self.now_unix_ms,
            expires_at_unix_ms,
        })
    }
}

pub(crate) fn is_identity_method(method: &str) -> bool {
    matches!(method, "ReadIdentityBundle" | "DelegatePluginSigningKey")
}

/// This broker receives the connection's authenticated name, never the envelope
/// plugin_id. Identity data is read from the running node at each setup call.
pub(crate) async fn handle_request(
    node: &crate::mesh::Node,
    plugin_id: &str,
    request: super::proto::RpcRequest,
) -> Result<super::proto::RpcResponse, super::proto::ErrorResponse> {
    let result = match node.plugin_manager().await {
        Some(manager) => {
            let admitted = admit_identity_service(&manager, plugin_id).await;
            if let Err(error) = admitted {
                return Err(identity_service_error(error));
            }
            let mut revision = manager.exchange_grant_revision(plugin_id);
            let result=tokio::time::timeout(super::identity_registration::IDENTITY_EXECUTION_DEADLINE,async {tokio::select! {
                biased;
                _ = revision.changed() => Err(anyhow::anyhow!("identity service grant changed during invocation")),
                result = handle_identity_request(node, plugin_id, request) => result,
            }}).await.unwrap_or_else(|_|Err(anyhow::anyhow!("identity service execution deadline exceeded")));
            manager
                .inner
                .identity_registry
                .lock()
                .await
                .release_service(plugin_id);
            result
        }
        None => Err(anyhow::anyhow!("plugin manager unavailable")),
    };
    result.map_err(identity_service_error)
}
async fn admit_identity_service(manager: &super::PluginManager, plugin_id: &str) -> Result<()> {
    // Signing and revocation hold the shared registry briefly. Queue setup calls
    // through those critical sections without losing per-plugin admission limits.
    let mut registry = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        manager.inner.identity_registry.lock(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("identity service admission deadline exceeded; retry setup"))?;
    registry.admit_service(
        plugin_id,
        chrono::Utc::now().timestamp_millis().unsigned_abs(),
    )
}

fn registration_error_requires_revocation(error: &anyhow::Error) -> bool {
    !super::identity_registration::artifact_inspection_is_transient(error)
}

fn identity_service_error(error: anyhow::Error) -> super::proto::ErrorResponse {
    super::proto::ErrorResponse {
        code: rmcp::model::ErrorCode::INVALID_REQUEST.0,
        message: error.to_string(),
        data_json: String::new(),
    }
}

async fn handle_identity_request(
    node: &crate::mesh::Node,
    plugin_id: &str,
    request: super::proto::RpcRequest,
) -> Result<super::proto::RpcResponse> {
    let manager = node
        .plugin_manager()
        .await
        .ok_or_else(|| anyhow::anyhow!("plugin manager unavailable"))?;
    validate_identity_input(&request)?;
    if request.method == "DelegatePluginSigningKey"
        && manager.plugin_exchange_callback_active(plugin_id).await
    {
        bail!("DelegatePluginSigningKey must renew outside lifecycle callbacks");
    }
    let (grants, artifact_sha256, artifact_status) =
        match manager.identity_grants_and_artifact(plugin_id).await {
            Ok(binding) => binding,
            Err(error) => {
                if registration_error_requires_revocation(&error) {
                    manager.revoke_plugin_delegations(plugin_id).await;
                }
                return Err(error);
            }
        };
    let ownership = node.owner_attestation.lock().await.clone();
    let trust = node.trust_store.lock().await.clone();
    let endpoint_id = node.id();
    let endpoint = endpoint_id.as_bytes();
    let snapshot = PluginIdentitySnapshot {
        node_endpoint_id: endpoint,
        ownership: ownership.as_ref(),
        trust_store: &trust,
        now_unix_ms: chrono::Utc::now().timestamp_millis().unsigned_abs(),
    };
    manager.inner.identity_registry.lock().await.invalidate(
        plugin_id,
        ownership.as_ref().map(|cert| cert.claim.owner_id.as_str()),
        &hex::encode(endpoint),
        ownership.as_ref().map(|cert| cert.claim.cert_id.as_str()),
        artifact_sha256.as_deref().unwrap_or(""),
    );
    if !grants.delegate_signing_key
        || snapshot.status() != IdentityEvidenceStatus::Verified
        || artifact_status != IdentityEvidenceStatus::Verified
    {
        manager.revoke_plugin_delegations(plugin_id).await;
    }
    let result_json = match request.method.as_str() {
        "ReadIdentityBundle" => {
            let params: serde_json::Value = serde_json::from_str(&request.params_json)?;
            if !params.is_null() && params != serde_json::json!({}) {
                bail!("ReadIdentityBundle accepts no claims or parameters");
            }
            let mut bundle = snapshot.read_identity_bundle(&grants)?;
            bundle.release_attestation = node.release_attestation.lock().await.clone();
            bundle.release_attestation_summary =
                Some(node.release_attestation_summary.lock().await.clone());
            bundle.plugin_id = Some(plugin_id.to_owned());
            bundle.plugin_artifact_sha256 = artifact_sha256;
            bundle.plugin_artifact_status = artifact_status;
            bundle.plugin_artifact_metadata = Some(manager.identity_plugin_metadata(plugin_id)?);
            bundle.signing_delegations = manager
                .inner
                .identity_registry
                .lock()
                .await
                .state(plugin_id, snapshot.now_unix_ms);
            serde_json::to_string(&bundle)?
        }
        "DelegatePluginSigningKey" => {
            if artifact_status != IdentityEvidenceStatus::Verified {
                bail!(
                    "installed plugin artifact changed or unavailable; restart and reauthorize delegation"
                );
            }
            if manager.plugin_exchange_callback_active(plugin_id).await {
                bail!("DelegatePluginSigningKey must renew outside lifecycle callbacks");
            }
            let params: DelegatePluginSigningKeyRequest =
                serde_json::from_str(&request.params_json)?;
            let registration = PluginSigningBinding {
                plugin_id: plugin_id.to_owned(),
                artifact_sha256: artifact_sha256
                    .ok_or_else(|| anyhow::anyhow!("plugin artifact unavailable"))?,
                signing_public_key: params.signing_public_key.clone(),
                scope: params.scope,
            };
            let owner = node.owner_keypair.as_ref().ok_or_else(|| {
                anyhow::anyhow!("owner signing key unavailable; unlock outside inference")
            })?;
            let claim = snapshot.delegation_claim(&grants, &registration, owner, &params)?;
            let mut registry = manager.inner.identity_registry.lock().await;
            let delegation = if let Some(cached) = registry.preflight_issuance(&claim)? {
                cached
            } else {
                let delegation = sign_plugin_delegation(owner, claim)?;
                registry.register(delegation.clone())?;
                delegation
            };
            serde_json::to_string(&delegation)?
        }
        _ => bail!("unsupported identity service"),
    };
    Ok(super::proto::RpcResponse { result_json })
}

fn validate_identity_input(request: &super::proto::RpcRequest) -> Result<()> {
    match request.method.as_str() {
        "ReadIdentityBundle" => {
            let params: serde_json::Value = serde_json::from_str(&request.params_json)?;
            if !params.is_null() && params != serde_json::json!({}) {
                bail!("ReadIdentityBundle accepts no claims or parameters");
            }
        }
        "DelegatePluginSigningKey" => {
            let params: DelegatePluginSigningKeyRequest =
                serde_json::from_str(&request.params_json)?;
            if params.signing_public_key.len() != 64
                || hex::decode(&params.signing_public_key).is_err()
            {
                bail!("delegation signing key must contain 32 hexadecimal public-key bytes");
            }
            if !(1000..=MAX_DELEGATION_LIFETIME_MS).contains(&params.lifetime_ms) {
                bail!("delegation lifetime outside protocol bounds");
            }
        }
        _ => bail!("unsupported identity service"),
    }
    Ok(())
}

#[cfg(test)]
#[path = "identity_services_tests.rs"]
mod tests;

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[tokio::test]
    async fn admission_waits_for_shared_registry_and_retains_per_plugin_limits() {
        let manager = super::super::PluginManager::for_test_summaries(Vec::new());
        let mut held = manager.inner.identity_registry.lock().await;
        held.admit_service("other-plugin", 1000).unwrap();
        let admission = admit_identity_service(&manager, "observer");
        tokio::pin!(admission);
        tokio::select! {
            result = &mut admission => panic!("contended admission returned early: {result:?}"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {},
        }
        drop(held);
        admission.await.unwrap();
        assert!(
            admit_identity_service(&manager, "observer")
                .await
                .unwrap_err()
                .to_string()
                .contains("already active")
        );
        admit_identity_service(&manager, "healthy").await.unwrap();
        manager
            .inner
            .identity_registry
            .lock()
            .await
            .release_service("observer");
        admit_identity_service(&manager, "observer").await.unwrap();
    }

    #[test]
    fn transient_artifact_timeout_does_not_revoke_current_delegations() {
        let timeout: anyhow::Error =
            super::super::identity_registration::ArtifactInspectionTimeout.into();
        assert!(!registration_error_requires_revocation(&timeout));
        assert!(registration_error_requires_revocation(&anyhow::anyhow!(
            "identity grant unavailable"
        )));
    }

    #[tokio::test]
    async fn admission_deadline_does_not_consume_plugin_quota() {
        let manager = super::super::PluginManager::for_test_summaries(Vec::new());
        let held = manager.inner.identity_registry.lock().await;
        assert!(
            admit_identity_service(&manager, "observer")
                .await
                .unwrap_err()
                .to_string()
                .contains("admission deadline")
        );
        drop(held);
        admit_identity_service(&manager, "observer").await.unwrap();
    }
}
