//! Detached Minisign authentication for native-runtime release catalogs.

use crate::manifest::normalize_sha256;
use anyhow::{Context, Result, bail};
use minisign_verify::{PublicKey, Signature};
use skippy_native_runtime::{NativeRuntimeArtifact, NativeRuntimeReleaseManifest};
use std::path::{Path, PathBuf};

pub(crate) const CATALOG_SIGNATURE_SUFFIX: &str = ".minisig";

/// Authenticate the exact catalog bytes before a JSON parser can inspect them.
pub(crate) fn verify_catalog_bytes(
    catalog_bytes: &[u8],
    checksum_text: &str,
    signature_text: &str,
    trusted_public_key: Option<&str>,
) -> Result<()> {
    crate::manifest::verify_release_manifest_checksum(catalog_bytes, checksum_text)?;
    let trusted_public_key = trusted_public_key
        .filter(|key| !key.trim().is_empty())
        .context("strict native runtime catalog verification requires a trusted Minisign public key supplied independently of the catalog")?;
    let public_key = PublicKey::decode(trusted_public_key)
        .context("decode trusted native runtime catalog Minisign public key")?;
    let signature = Signature::decode(signature_text)
        .context("parse detached native runtime catalog Minisign signature")?;
    public_key
        .verify(catalog_bytes, &signature, false)
        .context("verify detached native runtime catalog Minisign signature")
}

/// Enforce the signed catalog contract after authentication and parsing.
pub(crate) fn validate_authenticated_catalog(
    manifest: &NativeRuntimeReleaseManifest,
    requested_release: &str,
) -> Result<()> {
    if manifest.release_version != requested_release {
        bail!(
            "authenticated native runtime catalog release {} does not match requested release {requested_release}",
            manifest.release_version
        );
    }
    for artifact in &manifest.artifacts {
        validate_authenticated_artifact(artifact, &manifest.release_version)?;
    }
    Ok(())
}

fn validate_authenticated_artifact(
    artifact: &NativeRuntimeArtifact,
    catalog_release: &str,
) -> Result<()> {
    if artifact.release_version.as_deref() != Some(catalog_release) {
        bail!(
            "authenticated native runtime artifact {} release does not match catalog release {catalog_release}",
            artifact.native_runtime_id()
        );
    }
    let archive_sha256 = artifact.sha256.as_deref().with_context(|| {
        format!(
            "authenticated native runtime artifact {} is missing its archive sha256",
            artifact.native_runtime_id()
        )
    })?;
    normalize_sha256(archive_sha256).with_context(|| {
        format!(
            "authenticated native runtime artifact {} has an invalid archive sha256",
            artifact.native_runtime_id()
        )
    })?;
    if artifact.files.is_empty() {
        bail!(
            "authenticated native runtime artifact {} does not declare file hashes",
            artifact.native_runtime_id()
        );
    }
    for library in &artifact.libraries {
        if !artifact.files.contains_key(library) {
            bail!(
                "authenticated native runtime artifact {} library {library} is missing its file hash",
                artifact.native_runtime_id()
            );
        }
    }
    Ok(())
}

pub(crate) fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(suffix);
    PathBuf::from(sidecar)
}

pub(crate) fn sidecar_url(url: &str, suffix: &str) -> String {
    let url = url.split_once('#').map_or(url, |(base, _)| base);
    match url.split_once('?') {
        Some((base, query)) => format!("{base}{suffix}?{query}"),
        None => format!("{url}{suffix}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisign::KeyPair;
    use sha2::{Digest, Sha256};
    use std::io::Cursor;

    #[test]
    fn exact_bytes_are_authenticated_and_legacy_signatures_are_not_enabled() {
        let key_pair = KeyPair::generate_unencrypted_keypair().unwrap();
        let bytes = b"{\"schema_version\":2}\n";
        let checksum = hex::encode(Sha256::digest(bytes));
        let signature = minisign::sign(
            Some(&key_pair.pk),
            &key_pair.sk,
            Cursor::new(bytes),
            None,
            None,
        )
        .unwrap()
        .into_string();
        let public_key = key_pair.pk.to_box().unwrap().into_string();

        verify_catalog_bytes(bytes, &checksum, &signature, Some(&public_key)).unwrap();
        let error = verify_catalog_bytes(
            b"{\"schema_version\":2} \n",
            &hex::encode(Sha256::digest(b"{\"schema_version\":2} \n")),
            &signature,
            Some(&public_key),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("Minisign signature"),
            "{error:#}"
        );
    }

    #[test]
    fn sidecars_append_before_queries_and_drop_fragments() {
        assert_eq!(
            sidecar_url("https://host/catalog.json?token=1#fragment", ".minisig"),
            "https://host/catalog.json.minisig?token=1"
        );
        assert_eq!(
            sidecar_path(Path::new("catalog.json"), CATALOG_SIGNATURE_SUFFIX),
            PathBuf::from("catalog.json.minisig")
        );
    }
}
