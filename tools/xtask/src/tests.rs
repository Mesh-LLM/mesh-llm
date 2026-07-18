use crate::attestation::{
    EmbeddedReleaseAttestation, InspectArgs, ReleaseSigningPrivateKeyFile,
    ReleaseSigningPublicKeyFile, inspect_release_attestation_summary, stamp_release_attestation,
};
use crate::command::{DynResult, unique_temp_dir, write_json_file};
use ed25519_dalek::SigningKey;
use mesh_llm_system::embedded_release_footer::read_embedded_release_footer;
use std::fs;
use std::path::{Path, PathBuf};

fn test_signing_key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn write_test_keypair(dir: &Path, seed: u8) -> DynResult<(PathBuf, PathBuf)> {
    let signing_key = test_signing_key(seed);
    let private_key_path = dir.join("release-key");
    let public_key_path = dir.join("release-key.pub");
    write_json_file(
        &private_key_path,
        &ReleaseSigningPrivateKeyFile::from_signing_key(&signing_key),
    )?;
    write_json_file(
        &public_key_path,
        &ReleaseSigningPublicKeyFile::from_verifying_key(&signing_key.verifying_key()),
    )?;
    Ok((private_key_path, public_key_path))
}

fn make_temp_dir(label: &str) -> DynResult<PathBuf> {
    let dir = unique_temp_dir(label);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[test]
fn release_signing_key_files_round_trip() -> DynResult<()> {
    let signing_key = test_signing_key(7);
    let private_key_file = ReleaseSigningPrivateKeyFile::from_signing_key(&signing_key);
    let public_key_file =
        ReleaseSigningPublicKeyFile::from_verifying_key(&signing_key.verifying_key());

    let loaded_signing_key = private_key_file.signing_key()?;
    let loaded_public_key = public_key_file.verifying_key()?;
    assert_eq!(loaded_signing_key.as_bytes(), signing_key.as_bytes());
    assert_eq!(
        loaded_public_key.as_bytes(),
        signing_key.verifying_key().as_bytes()
    );
    assert_eq!(
        private_key_file.signer_key_id,
        public_key_file.signer_key_id
    );
    Ok(())
}

#[test]
fn release_attestation_inspect_reports_missing_without_public_key() -> DynResult<()> {
    let dir = make_temp_dir("xtask-attestation-missing")?;
    let binary_path = dir.join("mesh-llm");
    fs::write(&binary_path, b"plain release binary")?;

    let summary = inspect_release_attestation_summary(&InspectArgs {
        binary: Some(binary_path),
        public_key_file: None,
        json: true,
    })?;
    assert_eq!(summary.status, "missing");
    assert_eq!(summary.version, None);
    assert_eq!(summary.signer_key_id, None);
    assert_eq!(summary.artifact_digest, None);
    assert_eq!(summary.error, None);
    Ok(())
}

#[test]
fn release_attestation_stamp_and_inspect_round_trip() -> DynResult<()> {
    let dir = make_temp_dir("xtask-attestation-valid")?;
    let binary_path = dir.join("mesh-llm");
    fs::write(&binary_path, b"release-binary-v1")?;
    let (private_key_path, public_key_path) = write_test_keypair(&dir, 11)?;

    stamp_release_attestation(&[
        "--binary".to_string(),
        binary_path.display().to_string(),
        "--signing-key-file".to_string(),
        private_key_path.display().to_string(),
        "--node-version".to_string(),
        "9.9.9".to_string(),
        "--build-id".to_string(),
        "build-123".to_string(),
        "--commit".to_string(),
        "abcdef".to_string(),
        "--target-triple".to_string(),
        "x86_64-unknown-linux-gnu".to_string(),
    ])?;

    let summary = inspect_release_attestation_summary(&InspectArgs {
        binary: Some(binary_path.clone()),
        public_key_file: Some(public_key_path),
        json: true,
    })?;
    assert_eq!(summary.status, "valid");
    assert_eq!(summary.version, Some(1));
    assert!(
        summary
            .signer_key_id
            .as_deref()
            .is_some_and(|value| value.starts_with("ed25519:"))
    );
    assert!(
        summary
            .artifact_digest
            .as_deref()
            .is_some_and(|value| value.starts_with("sha256:"))
    );

    let binary_bytes = fs::read(binary_path)?;
    let footer = read_embedded_release_footer(&binary_bytes)?
        .expect("stamped binary should contain embedded footer");
    let embedded: EmbeddedReleaseAttestation = serde_json::from_slice(footer.payload_bytes)?;
    let claims = embedded.claims()?;
    let canonical_payload =
        claims.canonical_bytes(&embedded.signer_key_id, &embedded.signature_algorithm)?;
    assert_eq!(
        hex::decode(&embedded.signed_payload_hex)?,
        canonical_payload
    );
    assert_eq!(claims.node_version, "9.9.9");
    assert_eq!(claims.build_id, "build-123");
    assert_eq!(claims.commit, "abcdef");
    Ok(())
}

#[test]
fn release_attestation_inspect_reports_invalid_after_tamper() -> DynResult<()> {
    let dir = make_temp_dir("xtask-attestation-invalid")?;
    let binary_path = dir.join("mesh-llm");
    fs::write(&binary_path, b"release-binary-v1")?;
    let (private_key_path, public_key_path) = write_test_keypair(&dir, 13)?;

    stamp_release_attestation(&[
        "--binary".to_string(),
        binary_path.display().to_string(),
        "--signing-key-file".to_string(),
        private_key_path.display().to_string(),
    ])?;

    let mut tampered = fs::read(&binary_path)?;
    tampered[0] ^= 0x01;
    fs::write(&binary_path, tampered)?;

    let summary = inspect_release_attestation_summary(&InspectArgs {
        binary: Some(binary_path),
        public_key_file: Some(public_key_path),
        json: true,
    })?;
    assert_eq!(summary.status, "invalid");
    assert!(
        summary
            .error
            .as_deref()
            .is_some_and(|error| error.contains("artifact digest mismatch"))
    );
    assert!(summary.artifact_digest.is_some());
    Ok(())
}
