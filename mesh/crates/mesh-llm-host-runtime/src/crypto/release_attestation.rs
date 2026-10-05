//! Release-signature verification and trust, split across crates.
//!
//! The dependency-neutral contracts — the signed/attestation data types, their
//! canonical serialization, validation, and ed25519 verification, plus the
//! trust-store data model and `verify_release_attestation` — now live in
//! `mesh-llm-membership` (see `mesh_llm_membership::release_attestation`).
//!
//! This host module keeps the file/binary-surface concerns that belong to the
//! host runtime: the trust-store path and disk persistence, and the embedded
//! release-footer loader (which reads a real binary and relies on
//! `mesh_llm_system`'s footer verifier).

pub use mesh_llm_membership::release_attestation::{
    EmbeddedReleaseAttestation, LoadedEmbeddedReleaseAttestation, ReleaseAttestationClaims,
    ReleaseAttestationError, ReleaseAttestationStatus, ReleaseAttestationSummary,
    ReleaseBuildAttestation, ReleaseSignerTrustStore, TrustedReleaseSigner,
    VerifiedEmbeddedReleaseAttestation, parse_release_signer_public_key, release_signer_key_id,
    verify_release_attestation,
};

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use mesh_llm_system::embedded_release_footer::{
    EmbeddedReleaseFooterStatus, EmbeddedReleasePayloadSummary, EmbeddedReleasePayloadVerifier,
    verify_embedded_release_footer,
};

use super::{CryptoError, write_keystore_bytes_atomically};

pub use mesh_llm_membership::release_attestation::RELEASE_SIGNER_TRUST_STORE_VERSION;

fn mesh_dir() -> Result<PathBuf, CryptoError> {
    let home = dirs::home_dir().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine home directory",
        ))
    })?;
    Ok(home.join(".mesh-llm"))
}

pub fn default_release_signer_trust_store_path() -> Result<PathBuf, CryptoError> {
    Ok(mesh_dir()?.join("trusted-release-signers.json"))
}

pub fn load_release_signer_trust_store(
    path: &Path,
) -> Result<ReleaseSignerTrustStore, CryptoError> {
    if !path.exists() {
        return Ok(ReleaseSignerTrustStore::default());
    }
    let raw = std::fs::read_to_string(path)?;
    let store: ReleaseSignerTrustStore = serde_json::from_str(&raw)?;
    if store.version != RELEASE_SIGNER_TRUST_STORE_VERSION {
        return Err(CryptoError::UnsupportedVersion {
            version: store.version,
        });
    }
    Ok(store)
}

pub fn save_release_signer_trust_store(
    path: &Path,
    store: &ReleaseSignerTrustStore,
) -> Result<(), CryptoError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(store)?;
    write_keystore_bytes_atomically(path, &bytes)?;
    Ok(())
}

struct EmbeddedReleasePayloadCapture<'a> {
    trust_store: &'a ReleaseSignerTrustStore,
    now_unix_ms: u64,
    verified: RefCell<Option<VerifiedEmbeddedReleaseAttestation>>,
}

impl EmbeddedReleasePayloadVerifier for EmbeddedReleasePayloadCapture<'_> {
    type Error = ReleaseAttestationError;

    fn verify_payload(
        &self,
        payload_bytes: &[u8],
    ) -> Result<EmbeddedReleasePayloadSummary, Self::Error> {
        let embedded: EmbeddedReleaseAttestation = serde_json::from_slice(payload_bytes)
            .map_err(|error| ReleaseAttestationError::Json(error.to_string()))?;
        let claims = embedded.verify_claims()?;
        if claims
            .expires_at_unix_ms
            .is_some_and(|expires_at| self.now_unix_ms > expires_at)
        {
            return Err(ReleaseAttestationError::Expired);
        }
        if !self.trust_store.trusted_signers.is_empty()
            && !self
                .trust_store
                .trusted_signers
                .iter()
                .any(|entry| entry.signer_key_id == embedded.signer_key_id)
        {
            return Err(ReleaseAttestationError::UntrustedSigner);
        }
        let signature = embedded.signature_bytes()?.to_vec();
        let attestation = claims
            .clone()
            .into_release_build_attestation(embedded.signer_key_id.clone(), signature);
        let summary = claims.summary(
            embedded.signer_key_id,
            ReleaseAttestationStatus::Valid,
            true,
            None,
        );
        self.verified
            .replace(Some(VerifiedEmbeddedReleaseAttestation {
                attestation,
                summary,
            }));
        Ok(EmbeddedReleasePayloadSummary {
            artifact_digest: claims.artifact_digest,
        })
    }
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn load_embedded_release_attestation_for_binary(
    binary_path: &Path,
    trust_store: &ReleaseSignerTrustStore,
) -> Result<LoadedEmbeddedReleaseAttestation, ReleaseAttestationError> {
    let binary_bytes = std::fs::read(binary_path).map_err(|error| {
        ReleaseAttestationError::Io(format!(
            "failed to read release attestation binary {}: {error}",
            binary_path.display()
        ))
    })?;
    let verifier = EmbeddedReleasePayloadCapture {
        trust_store,
        now_unix_ms: current_time_unix_ms(),
        verified: RefCell::new(None),
    };
    let verification = verify_embedded_release_footer(&binary_bytes, &verifier);
    let loaded = match verification.status {
        EmbeddedReleaseFooterStatus::Missing => LoadedEmbeddedReleaseAttestation {
            binary_path: binary_path.to_path_buf(),
            summary: ReleaseAttestationSummary::default(),
            attestation: None,
        },
        EmbeddedReleaseFooterStatus::Valid => {
            let verified = verifier.verified.into_inner().ok_or_else(|| {
                ReleaseAttestationError::Footer(
                    "verified footer payload was not captured".to_string(),
                )
            })?;
            LoadedEmbeddedReleaseAttestation {
                binary_path: binary_path.to_path_buf(),
                summary: verified.summary,
                attestation: Some(verified.attestation),
            }
        }
        EmbeddedReleaseFooterStatus::Invalid => LoadedEmbeddedReleaseAttestation {
            binary_path: binary_path.to_path_buf(),
            summary: ReleaseAttestationSummary {
                status: ReleaseAttestationStatus::Invalid,
                error: verification.error,
                ..ReleaseAttestationSummary::default()
            },
            attestation: None,
        },
    };
    Ok(loaded)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use mesh_llm_membership::release_attestation::{
        EmbeddedReleaseAttestation, RELEASE_BUILD_ATTESTATION_VERSION,
    };
    use mesh_llm_system::embedded_release_footer::stamp_embedded_release_payload;
    use sha2::{Digest, Sha256};

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

    pub(crate) fn stamped_binary_bytes(signing_key: &ed25519_dalek::SigningKey) -> Vec<u8> {
        let base_bytes = b"mesh-llm-test-binary".to_vec();
        let artifact_digest = format!("sha256:{}", hex::encode(Sha256::digest(&base_bytes)));
        let signer_key_id = release_signer_key_id(&signing_key.verifying_key());
        let mut claims = test_claims(Some(signer_key_id.clone()));
        claims.artifact_digest = artifact_digest;
        let mut attestation = claims
            .clone()
            .into_release_build_attestation(signer_key_id.clone(), vec![0; 64]);
        let signed_payload_bytes = attestation
            .canonical_bytes()
            .expect("canonical attestation bytes");
        let signature = ed25519_dalek::Signer::sign(signing_key, &signed_payload_bytes);
        attestation.signature = signature.to_bytes().to_vec();
        let embedded = EmbeddedReleaseAttestation {
            version: RELEASE_BUILD_ATTESTATION_VERSION,
            signer_key_id,
            signature_algorithm: "ed25519".to_string(),
            claims,
            signed_payload_hex: hex::encode(&signed_payload_bytes),
            signature_hex: hex::encode(signature.to_bytes()),
        };
        stamp_embedded_release_payload(
            &base_bytes,
            &serde_json::to_vec(&embedded).expect("embedded json"),
        )
        .expect("stamp binary")
    }

    #[test]
    fn embedded_release_attestation_loader_reports_valid_summary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let binary_path = dir.path().join("mesh-llm");
        let signing_key = test_release_signing_key(8);
        std::fs::write(&binary_path, stamped_binary_bytes(&signing_key)).expect("write binary");

        let loaded = load_embedded_release_attestation_for_binary(
            &binary_path,
            &ReleaseSignerTrustStore::default(),
        )
        .expect("load embedded attestation");

        assert_eq!(loaded.summary.status, ReleaseAttestationStatus::Valid);
        assert!(loaded.summary.verified);
        let attestation = loaded.attestation.expect("embedded attestation");
        attestation
            .verify()
            .expect("embedded attestation should verify as canonical protocol attestation");
    }
}
