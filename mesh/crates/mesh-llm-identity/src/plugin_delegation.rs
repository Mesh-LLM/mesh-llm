//! Software identity claims authorizing one installed plugin key to sign evidence.
//!
//! These claims do not attest to hardware or prove that inference executed.

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::{CryptoError, OwnerKeypair, owner_id_from_verifying_key};

pub const PLUGIN_DELEGATION_VERSION: u32 = 1;
pub const MAX_DELEGATION_LIFETIME_MS: u64 = 24 * 60 * 60 * 1000;
const DOMAIN: &[u8] = b"mesh-llm-plugin-signing-delegation-v1:";

/// The only currently supported signing authority. No arbitrary owner signing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginSigningScope {
    #[serde(rename = "mesh.openai.exchange.evidence.sign.v1")]
    OpenAiExchangeEvidence,
}

/// Host-owned identity of the installed artifact and its registered public key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginSigningBinding {
    pub plugin_id: String,
    pub artifact_sha256: String,
    pub signing_public_key: String,
    pub scope: PluginSigningScope,
}

/// Fixed schema signed by the owner, bound to a particular node certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginSigningDelegationClaim {
    pub version: u32,
    pub delegation_id: String,
    pub owner_id: String,
    pub owner_sign_public_key: String,
    pub node_endpoint_id: String,
    pub node_certificate_id: String,
    pub plugin: PluginSigningBinding,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginSigningDelegation {
    pub claim: PluginSigningDelegationClaim,
    pub signature: String,
}

/// Evidence states remain distinct, including an unavailable verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityEvidenceStatus {
    Missing,
    Unverified,
    Verified,
    Expired,
    Revoked,
    Invalid,
}

/// Current authenticated state supplied by the host or an independent verifier.
pub struct DelegationVerification<'a> {
    pub owner_id: &'a str,
    pub node_endpoint_id: &'a str,
    pub node_certificate_id: &'a str,
    pub plugin: &'a PluginSigningBinding,
    pub identity_status: IdentityEvidenceStatus,
    pub revoked_delegation_ids: &'a [String],
    pub now_unix_ms: u64,
}

/// Stable UTF-8 JSON in declaration order with a domain prefix. Unknown fields
/// are rejected during decoding, and each string is encoded by serde_json.
pub fn canonical_delegation_bytes(
    claim: &PluginSigningDelegationClaim,
) -> Result<Vec<u8>, CryptoError> {
    let mut bytes = DOMAIN.to_vec();
    bytes.extend(serde_json::to_vec(claim)?);
    Ok(bytes)
}

fn valid_hex<const N: usize>(value: &str) -> Option<[u8; N]> {
    if value.len() != N * 2
        || value
            .bytes()
            .any(|b| !b.is_ascii_digit() && !(b'a'..=b'f').contains(&b))
    {
        return None;
    }
    hex::decode(value).ok()?.try_into().ok()
}

/// Require the canonical lowercase encoding of a non-weak Ed25519 public key.
pub fn is_valid_plugin_signing_key(value: &str) -> bool {
    valid_hex::<32>(value)
        .and_then(|key| VerifyingKey::from_bytes(&key).ok())
        .is_some_and(|key| !key.is_weak())
}

fn valid_claim(claim: &PluginSigningDelegationClaim) -> bool {
    claim.version == PLUGIN_DELEGATION_VERSION
        && !claim.delegation_id.is_empty()
        && claim.delegation_id.len() <= 128
        && !claim.node_certificate_id.is_empty()
        && claim.node_certificate_id.len() <= 128
        && !claim.plugin.plugin_id.is_empty()
        && claim.plugin.plugin_id.len() <= 128
        && valid_hex::<32>(&claim.owner_id).is_some()
        && valid_hex::<32>(&claim.node_endpoint_id).is_some()
        && valid_hex::<32>(&claim.plugin.artifact_sha256).is_some()
        && is_valid_plugin_signing_key(&claim.plugin.signing_public_key)
        && claim.expires_at_unix_ms > claim.issued_at_unix_ms
        && claim.expires_at_unix_ms - claim.issued_at_unix_ms <= MAX_DELEGATION_LIFETIME_MS
}

/// Sign a validated fixed claim. Host services construct this claim from their
/// own registration and certificate; they never accept it from plugin input.
pub fn sign_plugin_delegation(
    owner: &OwnerKeypair,
    claim: PluginSigningDelegationClaim,
) -> Result<PluginSigningDelegation, CryptoError> {
    if !valid_claim(&claim)
        || claim.owner_id != owner.owner_id()
        || claim.owner_sign_public_key != hex::encode(owner.verifying_key().as_bytes())
    {
        return Err(CryptoError::InvalidKeyMaterial {
            reason: "invalid plugin signing delegation claim".into(),
        });
    }
    let signature = hex::encode(owner.sign_bytes(&canonical_delegation_bytes(&claim)?));
    Ok(PluginSigningDelegation { claim, signature })
}

/// Verify the signature and all current identity, artifact, key, scope and
/// revocation bindings. A valid historical signature cannot authorize a changed
/// installation or renewed node certificate.
pub fn verify_plugin_delegation(
    delegation: Option<&PluginSigningDelegation>,
    context: &DelegationVerification<'_>,
) -> IdentityEvidenceStatus {
    let Some(delegation) = delegation else {
        return IdentityEvidenceStatus::Missing;
    };
    let claim = &delegation.claim;
    if !valid_claim(claim)
        || claim.owner_id != context.owner_id
        || claim.node_endpoint_id != context.node_endpoint_id
        || claim.node_certificate_id != context.node_certificate_id
        || &claim.plugin != context.plugin
        || claim.issued_at_unix_ms > context.now_unix_ms
        || !valid_signature(delegation)
    {
        return IdentityEvidenceStatus::Invalid;
    }
    if context
        .revoked_delegation_ids
        .contains(&claim.delegation_id)
        || context.identity_status == IdentityEvidenceStatus::Revoked
    {
        return IdentityEvidenceStatus::Revoked;
    }
    if claim.expires_at_unix_ms <= context.now_unix_ms
        || context.identity_status == IdentityEvidenceStatus::Expired
    {
        return IdentityEvidenceStatus::Expired;
    }
    match context.identity_status {
        IdentityEvidenceStatus::Verified => IdentityEvidenceStatus::Verified,
        IdentityEvidenceStatus::Missing | IdentityEvidenceStatus::Unverified => {
            IdentityEvidenceStatus::Unverified
        }
        status => status,
    }
}

fn valid_signature(delegation: &PluginSigningDelegation) -> bool {
    let Some(key) = valid_hex::<32>(&delegation.claim.owner_sign_public_key)
        .and_then(|key| VerifyingKey::from_bytes(&key).ok())
    else {
        return false;
    };
    if owner_id_from_verifying_key(&key) != delegation.claim.owner_id {
        return false;
    }
    let Some(signature) = valid_hex::<64>(&delegation.signature) else {
        return false;
    };
    let Ok(bytes) = canonical_delegation_bytes(&delegation.claim) else {
        return false;
    };
    key.verify_strict(&bytes, &Signature::from_bytes(&signature))
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PluginSigningDelegation {
        let owner = OwnerKeypair::from_bytes(&[1; 32], &[2; 32]).unwrap();
        let plugin_key = OwnerKeypair::from_bytes(&[3; 32], &[4; 32]).unwrap();
        sign_plugin_delegation(
            &owner,
            PluginSigningDelegationClaim {
                version: 1,
                delegation_id: "test-delegation".into(),
                owner_id: owner.owner_id(),
                owner_sign_public_key: hex::encode(owner.verifying_key().as_bytes()),
                node_endpoint_id: hex::encode([5; 32]),
                node_certificate_id: "test-cert".into(),
                plugin: PluginSigningBinding {
                    plugin_id: "observer".into(),
                    artifact_sha256: hex::encode([6; 32]),
                    signing_public_key: hex::encode(plugin_key.verifying_key().as_bytes()),
                    scope: PluginSigningScope::OpenAiExchangeEvidence,
                },
                issued_at_unix_ms: 100,
                expires_at_unix_ms: 1000,
            },
        )
        .unwrap()
    }

    fn context(delegation: &PluginSigningDelegation) -> DelegationVerification<'_> {
        DelegationVerification {
            owner_id: &delegation.claim.owner_id,
            node_endpoint_id: &delegation.claim.node_endpoint_id,
            node_certificate_id: &delegation.claim.node_certificate_id,
            plugin: &delegation.claim.plugin,
            identity_status: IdentityEvidenceStatus::Verified,
            revoked_delegation_ids: &[],
            now_unix_ms: 200,
        }
    }

    #[test]
    fn verifies_independent_signature_and_rejects_claim_tampering() {
        use sha2::{Digest, Sha256};
        let delegation = fixture();
        let bytes = canonical_delegation_bytes(&delegation.claim).unwrap();
        assert!(bytes.starts_with(DOMAIN));
        // Generated independently with Node.js crypto using PKCS#8 seed 0x01.
        assert_eq!(
            hex::encode(Sha256::digest(&bytes)),
            "668a517a32ab2494d360b18b97a94804b09ba4777a7aee9babaae769d4ffcb64"
        );
        assert_eq!(
            delegation.signature,
            "78e682975c1f7dc841f4999561c51b048dcf0a92a354ca7a31a47a01b95a03861e157f0980d93e087faa1dc7d223e7ed5399eb153af7cda5cf13d2138ccb290a"
        );
        let mut decoded: PluginSigningDelegation =
            serde_json::from_slice(&serde_json::to_vec(&delegation).unwrap()).unwrap();
        assert_eq!(
            verify_plugin_delegation(Some(&decoded), &context(&delegation)),
            IdentityEvidenceStatus::Verified
        );
        decoded.claim.plugin.plugin_id = "attacker".into();
        assert_eq!(
            verify_plugin_delegation(Some(&decoded), &context(&decoded)),
            IdentityEvidenceStatus::Invalid
        );
    }

    #[test]
    fn binds_current_installation_and_node_certificate() {
        let delegation = fixture();
        let mut binding = delegation.claim.plugin.clone();
        binding.artifact_sha256 = hex::encode([7; 32]);
        let mut ctx = context(&delegation);
        ctx.plugin = &binding;
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Invalid
        );
        ctx.plugin = &delegation.claim.plugin;
        ctx.node_endpoint_id = "changed-node";
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Invalid
        );
        ctx.node_endpoint_id = &delegation.claim.node_endpoint_id;
        ctx.owner_id = "changed-owner";
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Invalid
        );
        ctx.owner_id = &delegation.claim.owner_id;
        let mut changed_key = delegation.claim.plugin.clone();
        changed_key.signing_public_key =
            hex::encode(OwnerKeypair::generate().verifying_key().as_bytes());
        ctx.plugin = &changed_key;
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Invalid
        );
        ctx.plugin = &delegation.claim.plugin;
        ctx.node_certificate_id = "renewed";
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Invalid
        );
    }

    #[test]
    fn preserves_evidence_status_and_revocation() {
        let delegation = fixture();
        let mut ctx = context(&delegation);
        assert_eq!(
            verify_plugin_delegation(None, &ctx),
            IdentityEvidenceStatus::Missing
        );
        ctx.identity_status = IdentityEvidenceStatus::Unverified;
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Unverified
        );
        ctx.identity_status = IdentityEvidenceStatus::Verified;
        ctx.now_unix_ms = 1000;
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Expired
        );
        let revoked = vec![delegation.claim.delegation_id.clone()];
        ctx.revoked_delegation_ids = &revoked;
        assert_eq!(
            verify_plugin_delegation(Some(&delegation), &ctx),
            IdentityEvidenceStatus::Revoked
        );
    }

    #[test]
    fn refuses_arbitrary_claims_and_unbounded_lifetimes() {
        let owner = OwnerKeypair::from_bytes(&[1; 32], &[2; 32]).unwrap();
        let mut claim = fixture().claim;
        claim.expires_at_unix_ms = claim.issued_at_unix_ms + MAX_DELEGATION_LIFETIME_MS + 1;
        assert!(sign_plugin_delegation(&owner, claim).is_err());
        let mut weak_key_claim = fixture().claim;
        weak_key_claim.plugin.signing_public_key = hex::encode([0; 32]);
        assert!(sign_plugin_delegation(&owner, weak_key_claim).is_err());
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["claim"]["arbitrary_bytes"] = "secret".into();
        assert!(serde_json::from_value::<PluginSigningDelegation>(value).is_err());
    }
}
