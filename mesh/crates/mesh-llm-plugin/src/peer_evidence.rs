//! Versioned, currency-agnostic verification of opaque provider evidence.
use crate::{ManifestEntry, proto};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CONTRACT_VERSION: u32 = 1;
pub const HOST_CAPABILITY: &str = "peer_evidence.v1";
pub const MAX_EVIDENCE_BYTES: usize = 1024;

/// Declare support for verification through an operation in the same manifest.
pub fn peer_evidence_verifier(
    namespace: impl Into<String>,
    handler: impl Into<String>,
) -> proto::PeerEvidenceVerifierManifest {
    proto::PeerEvidenceVerifierManifest {
        contract_version: CONTRACT_VERSION,
        namespace: namespace.into(),
        handler: handler.into(),
    }
}
impl From<proto::PeerEvidenceVerifierManifest> for ManifestEntry {
    fn from(value: proto::PeerEvidenceVerifierManifest) -> Self {
        Self::PeerEvidenceVerifier(value)
    }
}

/// Validate the versioned verifier declaration and its operation handler.
pub fn validate_manifest(
    manifest: &proto::PluginManifest,
) -> Result<Option<&proto::PeerEvidenceVerifierManifest>> {
    let Some(declaration) = manifest.peer_evidence_verifier.as_ref() else {
        return Ok(None);
    };
    ensure!(
        declaration.contract_version == CONTRACT_VERSION,
        "unsupported peer evidence verification contract {}",
        declaration.contract_version
    );
    ensure!(
        !declaration.namespace.is_empty() && declaration.namespace.len() <= 64,
        "invalid peer evidence namespace"
    );
    ensure!(
        !declaration.handler.is_empty() && declaration.handler.len() <= 64,
        "invalid peer evidence handler"
    );
    ensure!(
        manifest
            .operations
            .iter()
            .filter(|operation| operation.name == declaration.handler)
            .count()
            == 1,
        "peer evidence handler must name exactly one declared operation"
    );
    Ok(Some(declaration))
}

/// The identity is authenticated by Mesh; evidence comes from its signed record.
#[derive(Clone, Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PeerEvidenceVerificationRequest {
    pub mesh_id: String,
    pub node_id: String,
    /// Missing evidence means no published proof; the verifier decides eligibility.
    #[serde(default)]
    pub evidence: Option<Value>,
}

/// Decisions expire at the plugin's Unix-second deadline unless they depend
/// only on fixed policy, identity and evidence and opt into restart validity.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PeerEvidenceVerificationResult {
    pub accepted: bool,
    #[serde(default)]
    pub valid_until: u64,
    /// Cache without time expiry, until eviction or this verifier restarts.
    /// Use only for verdicts that do not depend on mutable external state.
    #[serde(default)]
    pub cache_until_restart: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    #[test]
    fn restart_validity_requires_an_explicit_opt_in() {
        let finite: PeerEvidenceVerificationResult =
            serde_json::from_value(serde_json::json!({"accepted":true,"valid_until":123})).unwrap();
        assert!(!finite.cache_until_restart);
        let policy: PeerEvidenceVerificationResult = serde_json::from_value(
            serde_json::json!({"accepted":false,"cache_until_restart":true}),
        )
        .unwrap();
        assert!(policy.cache_until_restart);
    }
    fn manifest() -> proto::PluginManifest {
        crate::plugin_manifest![
            peer_evidence_verifier("test.evidence.v1", "verify"),
            crate::operation::<PeerEvidenceVerificationRequest>("verify", "Verify opaque evidence")
        ]
    }
    #[test]
    fn missing_evidence_is_explicit_and_round_trips() {
        for evidence in [None, Some(serde_json::json!({"bond":"reference"}))] {
            let request = PeerEvidenceVerificationRequest {
                mesh_id: "mesh".into(),
                node_id: "peer".into(),
                evidence: evidence.clone(),
            };
            let decoded: PeerEvidenceVerificationRequest =
                serde_json::from_value(serde_json::to_value(request).unwrap()).unwrap();
            assert_eq!(decoded.evidence, evidence);
        }
        let missing: PeerEvidenceVerificationRequest =
            serde_json::from_value(serde_json::json!({"mesh_id":"mesh","node_id":"peer"})).unwrap();
        assert!(missing.evidence.is_none());
    }

    #[test]
    fn declaration_is_additive_and_packaged_with_its_namespace_and_handler() {
        let old = proto::PluginManifest::default();
        assert!(
            proto::PluginManifest::decode(old.encode_to_vec().as_slice())
                .unwrap()
                .peer_evidence_verifier
                .is_none()
        );
        let declared = manifest();
        let decoded = proto::PluginManifest::decode(declared.encode_to_vec().as_slice()).unwrap();
        assert_eq!(
            validate_manifest(&decoded).unwrap().unwrap().namespace,
            "test.evidence.v1"
        );
        let packaged: Value =
            serde_json::from_str(&crate::package_manifest_json(&declared).unwrap()).unwrap();
        assert_eq!(packaged["peer_evidence_verifier"]["contract_version"], 1);
        assert_eq!(packaged["peer_evidence_verifier"]["handler"], "verify");
        assert_eq!(
            packaged["peer_evidence_verifier"]["namespace"],
            "test.evidence.v1"
        );
    }
    #[test]
    fn missing_handlers_unsupported_versions_and_invalid_names_fail_closed() {
        let mut value = manifest();
        value.operations.clear();
        assert!(validate_manifest(&value).is_err());
        let mut value = manifest();
        value
            .peer_evidence_verifier
            .as_mut()
            .unwrap()
            .contract_version = 2;
        assert!(validate_manifest(&value).is_err());
        for invalid in [String::new(), "x".repeat(65)] {
            let mut value = manifest();
            value.peer_evidence_verifier.as_mut().unwrap().namespace = invalid.clone();
            assert!(validate_manifest(&value).is_err());
            let mut value = manifest();
            value.peer_evidence_verifier.as_mut().unwrap().handler = invalid;
            assert!(validate_manifest(&value).is_err());
        }
        let mut value = manifest();
        value.operations.push(value.operations[0].clone());
        assert!(validate_manifest(&value).is_err());
    }
}
