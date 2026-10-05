use std::path::PathBuf;
use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const RELEASE_BUILD_ATTESTATION_VERSION: u32 = 1;
pub const RELEASE_SIGNER_TRUST_STORE_VERSION: u32 = 1;
const RELEASE_BUILD_ATTESTATION_DOMAIN_TAG: &[u8] = b"mesh-llm-release-attestation-v1:";
const ED25519_SIGNATURE_ALGORITHM: &str = "ed25519";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseBuildAttestation {
    pub version: u32,
    pub node_version: String,
    pub build_id: String,
    pub commit: String,
    pub target_triple: String,
    pub supported_protocol_generation_min: Option<u32>,
    pub supported_protocol_generation_max: Option<u32>,
    pub artifact_digest: Option<String>,
    pub signer_key_id: String,
    pub signature_algorithm: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddedReleaseAttestation {
    pub version: u32,
    pub signer_key_id: String,
    pub signature_algorithm: String,
    pub claims: ReleaseAttestationClaims,
    pub signed_payload_hex: String,
    pub signature_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseAttestationClaims {
    pub version: u32,
    pub node_version: String,
    pub build_id: String,
    pub commit: String,
    pub target_triple: String,
    pub supported_protocol_generation_min: Option<u32>,
    pub supported_protocol_generation_max: Option<u32>,
    pub artifact_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signer_key_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issued_at_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrustedReleaseSigner {
    pub signer_key_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseSignerTrustStore {
    pub version: u32,
    #[serde(default)]
    pub trusted_signers: Vec<TrustedReleaseSigner>,
}

impl Default for ReleaseSignerTrustStore {
    fn default() -> Self {
        Self {
            version: RELEASE_SIGNER_TRUST_STORE_VERSION,
            trusted_signers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseAttestationStatus {
    Valid,
    #[default]
    Missing,
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ReleaseAttestationSummary {
    pub status: ReleaseAttestationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signer_key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_triple: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issued_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_protocol_generation_min: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_protocol_generation_max: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub verified: bool,
}

#[derive(Debug, Clone)]
pub struct VerifiedEmbeddedReleaseAttestation {
    pub attestation: ReleaseBuildAttestation,
    pub summary: ReleaseAttestationSummary,
}

#[derive(Debug, Clone)]
pub struct LoadedEmbeddedReleaseAttestation {
    pub binary_path: PathBuf,
    pub summary: ReleaseAttestationSummary,
    pub attestation: Option<ReleaseBuildAttestation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseAttestationError {
    InvalidShape(&'static str),
    InvalidSignerKeyId,
    InvalidSignature,
    UnsupportedVersion(u32),
    UnsupportedSignatureAlgorithm(String),
    InvalidProtocolBounds,
    MissingRequiredSignedField(&'static str),
    Expired,
    UntrustedSigner,
    Footer(String),
    Io(String),
    Json(String),
}

impl fmt::Display for ReleaseAttestationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(reason) => write!(f, "invalid release attestation shape: {reason}"),
            Self::InvalidSignerKeyId => write!(f, "release signer key id is invalid"),
            Self::InvalidSignature => {
                write!(f, "release attestation signature verification failed")
            }
            Self::UnsupportedVersion(version) => {
                write!(f, "release attestation version {version} is unsupported")
            }
            Self::UnsupportedSignatureAlgorithm(algorithm) => write!(
                f,
                "release attestation signature algorithm {algorithm} is unsupported"
            ),
            Self::InvalidProtocolBounds => {
                write!(
                    f,
                    "release attestation protocol generation bounds are invalid"
                )
            }
            Self::MissingRequiredSignedField(field) => {
                write!(
                    f,
                    "release attestation signed payload is missing required field {field}"
                )
            }
            Self::Expired => write!(f, "release attestation has expired"),
            Self::UntrustedSigner => write!(f, "release attestation signer is not trusted"),
            Self::Footer(message) => write!(f, "{message}"),
            Self::Io(message) => write!(f, "{message}"),
            Self::Json(message) => write!(f, "{message}"),
        }
    }
}

impl Error for ReleaseAttestationError {}

pub fn release_signer_key_id(verifying_key: &ed25519_dalek::VerifyingKey) -> String {
    format!("ed25519:{}", hex::encode(verifying_key.as_bytes()))
}

pub fn parse_release_signer_public_key(
    signer_key_id: &str,
) -> Result<ed25519_dalek::VerifyingKey, ReleaseAttestationError> {
    let encoded = signer_key_id
        .trim()
        .strip_prefix("ed25519:")
        .ok_or(ReleaseAttestationError::InvalidSignerKeyId)?;
    let bytes = hex::decode(encoded).map_err(|_| ReleaseAttestationError::InvalidSignerKeyId)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| ReleaseAttestationError::InvalidSignerKeyId)?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes)
        .map_err(|_| ReleaseAttestationError::InvalidSignerKeyId)
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_optional_string(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            buf.push(1);
            write_string(buf, value);
        }
        None => buf.push(0),
    }
}

fn write_optional_u32(buf: &mut Vec<u8>, value: Option<u32>) {
    match value {
        Some(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_le_bytes());
        }
        None => buf.push(0),
    }
}

impl ReleaseBuildAttestation {
    pub fn to_proto(&self) -> mesh_llm_protocol::proto::node::ReleaseBuildAttestation {
        mesh_llm_protocol::proto::node::ReleaseBuildAttestation {
            version: self.version,
            node_version: self.node_version.clone(),
            build_id: self.build_id.clone(),
            commit: self.commit.clone(),
            target_triple: self.target_triple.clone(),
            supported_protocol_generation_min: self.supported_protocol_generation_min,
            supported_protocol_generation_max: self.supported_protocol_generation_max,
            artifact_digest: self.artifact_digest.clone(),
            signer_key_id: self.signer_key_id.clone(),
            signature_algorithm: self.signature_algorithm.clone(),
            signature: self.signature.clone(),
        }
    }

    pub fn from_proto(value: &mesh_llm_protocol::proto::node::ReleaseBuildAttestation) -> Self {
        Self {
            version: value.version,
            node_version: value.node_version.clone(),
            build_id: value.build_id.clone(),
            commit: value.commit.clone(),
            target_triple: value.target_triple.clone(),
            supported_protocol_generation_min: value.supported_protocol_generation_min,
            supported_protocol_generation_max: value.supported_protocol_generation_max,
            artifact_digest: value.artifact_digest.clone(),
            signer_key_id: value.signer_key_id.clone(),
            signature_algorithm: value.signature_algorithm.clone(),
            signature: value.signature.clone(),
        }
    }

    pub fn validate(&self) -> Result<(), ReleaseAttestationError> {
        if self.version != RELEASE_BUILD_ATTESTATION_VERSION {
            return Err(ReleaseAttestationError::UnsupportedVersion(self.version));
        }
        if self.node_version.trim().is_empty()
            || self.build_id.trim().is_empty()
            || self.commit.trim().is_empty()
            || self.target_triple.trim().is_empty()
            || self.signer_key_id.trim().is_empty()
            || self.signature.is_empty()
        {
            return Err(ReleaseAttestationError::InvalidShape(
                "missing required release attestation fields",
            ));
        }
        if self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM {
            return Err(ReleaseAttestationError::UnsupportedSignatureAlgorithm(
                self.signature_algorithm.clone(),
            ));
        }
        if let (Some(min), Some(max)) = (
            self.supported_protocol_generation_min,
            self.supported_protocol_generation_max,
        ) && min > max
        {
            return Err(ReleaseAttestationError::InvalidProtocolBounds);
        }
        parse_release_signer_public_key(&self.signer_key_id)?;
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ReleaseAttestationError> {
        self.validate()?;
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(RELEASE_BUILD_ATTESTATION_DOMAIN_TAG);
        buf.extend_from_slice(&self.version.to_le_bytes());
        write_string(&mut buf, self.node_version.trim());
        write_string(&mut buf, self.build_id.trim());
        write_string(&mut buf, self.commit.trim());
        write_string(&mut buf, self.target_triple.trim());
        write_optional_u32(&mut buf, self.supported_protocol_generation_min);
        write_optional_u32(&mut buf, self.supported_protocol_generation_max);
        write_optional_string(&mut buf, self.artifact_digest.as_deref());
        write_string(&mut buf, self.signer_key_id.trim());
        write_string(&mut buf, self.signature_algorithm.trim());
        Ok(buf)
    }

    pub fn canonical_hash_hex(&self) -> Result<String, ReleaseAttestationError> {
        Ok(hex::encode(Sha256::digest(self.canonical_bytes()?)))
    }

    pub fn verify(&self) -> Result<(), ReleaseAttestationError> {
        self.validate()?;
        if self.signature.len() != 64 {
            return Err(ReleaseAttestationError::InvalidSignature);
        }
        let signer_public_key = parse_release_signer_public_key(self.signer_key_id.trim())?;
        let signature = ed25519_dalek::Signature::from_bytes(
            &self
                .signature
                .as_slice()
                .try_into()
                .map_err(|_| ReleaseAttestationError::InvalidSignature)?,
        );
        signer_public_key
            .verify_strict(&self.canonical_bytes()?, &signature)
            .map_err(|_| ReleaseAttestationError::InvalidSignature)
    }
}

impl EmbeddedReleaseAttestation {
    pub fn signed_payload_bytes(&self) -> Result<Vec<u8>, ReleaseAttestationError> {
        hex::decode(self.signed_payload_hex.trim())
            .map_err(|error| ReleaseAttestationError::Json(error.to_string()))
    }

    pub fn signature_bytes(&self) -> Result<[u8; 64], ReleaseAttestationError> {
        let bytes = hex::decode(self.signature_hex.trim())
            .map_err(|error| ReleaseAttestationError::Json(error.to_string()))?;
        bytes
            .try_into()
            .map_err(|_| ReleaseAttestationError::InvalidSignature)
    }

    pub fn validate(&self) -> Result<(), ReleaseAttestationError> {
        if self.version != RELEASE_BUILD_ATTESTATION_VERSION {
            return Err(ReleaseAttestationError::UnsupportedVersion(self.version));
        }
        if self.signer_key_id.trim().is_empty()
            || self.signed_payload_hex.trim().is_empty()
            || self.signature_hex.trim().is_empty()
        {
            return Err(ReleaseAttestationError::InvalidShape(
                "embedded release attestation is missing required fields",
            ));
        }
        if self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM {
            return Err(ReleaseAttestationError::UnsupportedSignatureAlgorithm(
                self.signature_algorithm.clone(),
            ));
        }
        let _ = parse_release_signer_public_key(&self.signer_key_id)?;
        let _ = self.signature_bytes()?;
        let _ = self.signed_payload_bytes()?;
        self.claims.validate_against_signer(&self.signer_key_id)?;
        Ok(())
    }

    pub fn verify_claims(&self) -> Result<ReleaseAttestationClaims, ReleaseAttestationError> {
        self.validate()?;
        let signer_public_key = parse_release_signer_public_key(&self.signer_key_id)?;
        let signature = ed25519_dalek::Signature::from_bytes(&self.signature_bytes()?);
        let signed_payload_bytes = self.signed_payload_bytes()?;
        let attestation = self.claims.clone().into_release_build_attestation(
            self.signer_key_id.clone(),
            self.signature_bytes()?.to_vec(),
        );
        if signed_payload_bytes != attestation.canonical_bytes()? {
            return Err(ReleaseAttestationError::InvalidShape(
                "embedded release attestation signed payload does not match claims",
            ));
        }
        signer_public_key
            .verify_strict(&signed_payload_bytes, &signature)
            .map_err(|_| ReleaseAttestationError::InvalidSignature)?;
        Ok(self.claims.clone())
    }
}

impl ReleaseAttestationClaims {
    pub fn validate_against_signer(
        &self,
        envelope_signer_key_id: &str,
    ) -> Result<(), ReleaseAttestationError> {
        if self.version != RELEASE_BUILD_ATTESTATION_VERSION {
            return Err(ReleaseAttestationError::UnsupportedVersion(self.version));
        }
        if self.node_version.trim().is_empty() {
            return Err(ReleaseAttestationError::MissingRequiredSignedField(
                "node_version",
            ));
        }
        if self.build_id.trim().is_empty() {
            return Err(ReleaseAttestationError::MissingRequiredSignedField(
                "build_id",
            ));
        }
        if self.commit.trim().is_empty() {
            return Err(ReleaseAttestationError::MissingRequiredSignedField(
                "commit",
            ));
        }
        if self.target_triple.trim().is_empty() {
            return Err(ReleaseAttestationError::MissingRequiredSignedField(
                "target_triple",
            ));
        }
        if self.artifact_digest.trim().is_empty() {
            return Err(ReleaseAttestationError::MissingRequiredSignedField(
                "artifact_digest",
            ));
        }
        if !self.artifact_digest.starts_with("sha256:") {
            return Err(ReleaseAttestationError::InvalidShape(
                "artifact digest must start with sha256:",
            ));
        }
        if let Some(signer_key_id) = self.signer_key_id.as_deref()
            && signer_key_id.trim() != envelope_signer_key_id.trim()
        {
            return Err(ReleaseAttestationError::InvalidShape(
                "signed payload signer_key_id does not match envelope signer_key_id",
            ));
        }
        if let (Some(min), Some(max)) = (
            self.supported_protocol_generation_min,
            self.supported_protocol_generation_max,
        ) && min > max
        {
            return Err(ReleaseAttestationError::InvalidProtocolBounds);
        }
        Ok(())
    }

    pub fn into_release_build_attestation(
        self,
        signer_key_id: String,
        signature: Vec<u8>,
    ) -> ReleaseBuildAttestation {
        ReleaseBuildAttestation {
            version: self.version,
            node_version: self.node_version,
            build_id: self.build_id,
            commit: self.commit,
            target_triple: self.target_triple,
            supported_protocol_generation_min: self.supported_protocol_generation_min,
            supported_protocol_generation_max: self.supported_protocol_generation_max,
            artifact_digest: Some(self.artifact_digest),
            signer_key_id,
            signature_algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            signature,
        }
    }

    pub fn summary(
        &self,
        signer_key_id: String,
        status: ReleaseAttestationStatus,
        verified: bool,
        error: Option<String>,
    ) -> ReleaseAttestationSummary {
        ReleaseAttestationSummary {
            status,
            signer_key_id: Some(signer_key_id),
            node_version: Some(self.node_version.clone()),
            build_id: Some(self.build_id.clone()),
            commit: Some(self.commit.clone()),
            target_triple: Some(self.target_triple.clone()),
            artifact_digest: Some(self.artifact_digest.clone()),
            issued_at_unix_ms: self.issued_at_unix_ms,
            expires_at_unix_ms: self.expires_at_unix_ms,
            supported_protocol_generation_min: self.supported_protocol_generation_min,
            supported_protocol_generation_max: self.supported_protocol_generation_max,
            error,
            verified,
        }
    }
}

impl ReleaseSignerTrustStore {
    pub fn merged_with_trusted_signers(mut self, signer_ids: &[String]) -> Self {
        for signer_key_id in signer_ids {
            self.add_trusted_signer(signer_key_id.clone(), None);
        }
        self
    }

    pub fn add_trusted_signer(&mut self, signer_key_id: String, label: Option<String>) {
        if let Some(existing) = self
            .trusted_signers
            .iter_mut()
            .find(|entry| entry.signer_key_id == signer_key_id)
        {
            if label.is_some() {
                existing.label = label;
            }
            return;
        }
        self.trusted_signers.push(TrustedReleaseSigner {
            signer_key_id,
            label,
        });
        self.trusted_signers
            .sort_by(|a, b| a.signer_key_id.cmp(&b.signer_key_id));
    }
}

pub fn verify_release_attestation(
    attestation: Option<&ReleaseBuildAttestation>,
    trust_store: &ReleaseSignerTrustStore,
) -> ReleaseAttestationSummary {
    let Some(attestation) = attestation else {
        return ReleaseAttestationSummary::default();
    };
    let mut summary = ReleaseAttestationSummary {
        status: ReleaseAttestationStatus::Invalid,
        signer_key_id: Some(attestation.signer_key_id.clone()),
        node_version: Some(attestation.node_version.clone()),
        build_id: Some(attestation.build_id.clone()),
        commit: Some(attestation.commit.clone()),
        target_triple: Some(attestation.target_triple.clone()),
        artifact_digest: attestation.artifact_digest.clone(),
        supported_protocol_generation_min: attestation.supported_protocol_generation_min,
        supported_protocol_generation_max: attestation.supported_protocol_generation_max,
        ..ReleaseAttestationSummary::default()
    };
    if let Err(error) = attestation.verify() {
        summary.error = Some(error.to_string());
        return summary;
    }
    if !trust_store.trusted_signers.is_empty()
        && !trust_store
            .trusted_signers
            .iter()
            .any(|entry| entry.signer_key_id == attestation.signer_key_id)
    {
        summary.error = Some(ReleaseAttestationError::UntrustedSigner.to_string());
        return summary;
    }
    summary.status = ReleaseAttestationStatus::Valid;
    summary.verified = true;
    summary
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use mesh_llm_identity::OwnerKeypair;

    pub(crate) fn test_release_signing_key(seed: u8) -> ed25519_dalek::SigningKey {
        ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
    }

    fn test_claims(signer_key_id: Option<String>) -> ReleaseAttestationClaims {
        ReleaseAttestationClaims {
            version: RELEASE_BUILD_ATTESTATION_VERSION,
            node_version: env!("CARGO_PKG_VERSION").to_string(),
            build_id: "test-build".into(),
            commit: "deadbeef".into(),
            target_triple: "x86_64-apple-darwin".into(),
            supported_protocol_generation_min: Some(1),
            supported_protocol_generation_max: Some(1),
            artifact_digest: "sha256:placeholder".into(),
            signer_key_id,
            issued_at_unix_ms: Some(1_717_171_717_000),
            expires_at_unix_ms: Some(1_817_171_717_000),
        }
    }

    fn signed_canonical_attestation(
        signing_key: &ed25519_dalek::SigningKey,
    ) -> ReleaseBuildAttestation {
        let signer_key_id = release_signer_key_id(&signing_key.verifying_key());
        let mut attestation = ReleaseBuildAttestation {
            version: RELEASE_BUILD_ATTESTATION_VERSION,
            node_version: env!("CARGO_PKG_VERSION").to_string(),
            build_id: "test-build".into(),
            commit: "deadbeef".into(),
            target_triple: "x86_64-apple-darwin".into(),
            supported_protocol_generation_min: Some(1),
            supported_protocol_generation_max: Some(1),
            artifact_digest: Some("sha256:test".into()),
            signer_key_id,
            signature_algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            signature: vec![0; 64],
        };
        attestation.signature = ed25519_dalek::Signer::sign(
            signing_key,
            &attestation
                .canonical_bytes()
                .expect("canonical attestation bytes"),
        )
        .to_bytes()
        .to_vec();
        attestation
    }

    fn signed_json_claim_attestation(
        signing_key: &ed25519_dalek::SigningKey,
    ) -> ReleaseBuildAttestation {
        let signer_key_id = release_signer_key_id(&signing_key.verifying_key());
        let claims = test_claims(Some(signer_key_id.clone()));
        let signed_payload_bytes = serde_json::to_vec(&claims).expect("claims json");
        let signature = ed25519_dalek::Signer::sign(signing_key, &signed_payload_bytes);
        claims.into_release_build_attestation(signer_key_id, signature.to_bytes().to_vec())
    }

    #[test]
    fn release_attestation_verifies_trusted_release_signer() {
        let signing_key = test_release_signing_key(7);
        let signer_key_id = release_signer_key_id(&signing_key.verifying_key());
        let attestation = signed_canonical_attestation(&signing_key);
        let trust_store = ReleaseSignerTrustStore::default()
            .merged_with_trusted_signers(std::slice::from_ref(&signer_key_id));

        let summary = verify_release_attestation(Some(&attestation), &trust_store);

        assert_eq!(summary.status, ReleaseAttestationStatus::Valid);
        assert!(summary.verified);
        assert_eq!(
            summary.signer_key_id.as_deref(),
            Some(signer_key_id.as_str())
        );
    }

    #[test]
    fn release_attestation_does_not_accept_owner_key_material() {
        let signing_key = test_release_signing_key(7);
        let owner = OwnerKeypair::generate();
        let attestation = signed_canonical_attestation(&signing_key);
        let trust_store = ReleaseSignerTrustStore::default()
            .merged_with_trusted_signers(std::slice::from_ref(&owner.owner_id()));

        let summary = verify_release_attestation(Some(&attestation), &trust_store);

        assert_eq!(summary.status, ReleaseAttestationStatus::Invalid);
        assert!(!summary.verified);
        assert_eq!(
            summary.error.as_deref(),
            Some("release attestation signer is not trusted")
        );
    }

    #[test]
    fn release_attestation_rejects_json_claim_signature() {
        let signing_key = test_release_signing_key(9);
        let attestation = signed_json_claim_attestation(&signing_key);

        let error = attestation
            .verify()
            .expect_err("json claim signatures are not canonical release attestations");

        assert_eq!(error, ReleaseAttestationError::InvalidSignature);
    }

    #[test]
    fn embedded_release_attestation_rejects_claim_payload_mismatch() {
        let signing_key = test_release_signing_key(10);
        let signer_key_id = release_signer_key_id(&signing_key.verifying_key());
        let claims = test_claims(Some(signer_key_id.clone()));
        let attestation = claims
            .clone()
            .into_release_build_attestation(signer_key_id.clone(), vec![0; 64]);
        let signed_payload_bytes = attestation
            .canonical_bytes()
            .expect("canonical attestation bytes");
        let signature = ed25519_dalek::Signer::sign(&signing_key, &signed_payload_bytes);
        let mut mismatched_claims = claims;
        mismatched_claims.build_id = "different-build".into();
        let embedded = EmbeddedReleaseAttestation {
            version: RELEASE_BUILD_ATTESTATION_VERSION,
            signer_key_id,
            signature_algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            claims: mismatched_claims,
            signed_payload_hex: hex::encode(&signed_payload_bytes),
            signature_hex: hex::encode(signature.to_bytes()),
        };

        let error = embedded
            .verify_claims()
            .expect_err("claims must match the signed canonical payload exactly");

        assert_eq!(
            error,
            ReleaseAttestationError::InvalidShape(
                "embedded release attestation signed payload does not match claims"
            )
        );
    }
}
