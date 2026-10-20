//! Detached Minisign signing for the final native-runtime release catalog.

use crate::repository::check_report::CheckReport;
use minisign::{PublicKey, PublicKeyBox, SecretKeyBox};
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::Path;

const SECRET_KEY_ENV: &str = "MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_SECRET_KEY";
const PASSWORD_ENV: &str = "MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_PASSWORD";

pub(super) fn run(args: &[String]) -> CheckReport {
    match sign_from_environment(args) {
        Ok(()) => CheckReport::default(),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}

fn sign_from_environment(args: &[String]) -> Result<(), String> {
    let [catalog, signature_out, trusted_public_key] = args else {
        return Err(
            "usage: product runtime-catalog-sign <native-runtimes.json> <native-runtimes.json.minisig> <trusted-public-key-file>"
                .into(),
        );
    };
    let secret_key = std::env::var(SECRET_KEY_ENV).unwrap_or_default();
    let password = std::env::var(PASSWORD_ENV).unwrap_or_default();
    signing_credentials_configured(&secret_key, &password)?;
    sign_catalog(
        Path::new(catalog),
        Path::new(signature_out),
        Path::new(trusted_public_key),
        &secret_key,
        &password,
    )
}

fn signing_credentials_configured(secret_key: &str, password: &str) -> Result<(), String> {
    if secret_key.trim().is_empty() || password.is_empty() {
        return Err(format!(
            "{SECRET_KEY_ENV} and {PASSWORD_ENV} must both be configured; unsigned publication is forbidden"
        ));
    }
    Ok(())
}

pub(super) fn sign_catalog(
    catalog: &Path,
    signature_out: &Path,
    trusted_public_key_path: &Path,
    secret_key_text: &str,
    password: &str,
) -> Result<(), String> {
    let catalog_bytes = std::fs::read(catalog)
        .map_err(|error| format!("read native runtime catalog {}: {error}", catalog.display()))?;
    let trusted_public_key_text =
        std::fs::read_to_string(trusted_public_key_path).map_err(|error| {
            format!(
                "read trusted native runtime catalog public key {}: {error}",
                trusted_public_key_path.display()
            )
        })?;
    let trusted_public_key = PublicKeyBox::from_string(&trusted_public_key_text)
        .map_err(|error| format!("parse trusted native runtime catalog public key: {error}"))?
        .into_public_key()
        .map_err(|error| format!("decode trusted native runtime catalog public key: {error}"))?;
    let secret_key = SecretKeyBox::from_string(secret_key_text)
        .map_err(|error| format!("parse native runtime catalog secret key: {error}"))?
        .into_secret_key(Some(password.to_string()))
        .map_err(|error| format!("decrypt native runtime catalog secret key: {error}"))?;
    let derived_public_key = PublicKey::from_secret_key(&secret_key)
        .map_err(|error| format!("derive native runtime catalog public key: {error}"))?;
    if derived_public_key != trusted_public_key {
        return Err(
            "native runtime catalog signing key does not match the committed trusted public key"
                .into(),
        );
    }
    let signature = minisign::sign(
        Some(&trusted_public_key),
        &secret_key,
        Cursor::new(&catalog_bytes),
        Some("MeshLLM native runtime release catalog"),
        Some("signature from MeshLLM native runtime catalog key"),
    )
    .map_err(|error| format!("sign native runtime catalog: {error}"))?;
    minisign::verify(
        &trusted_public_key,
        &signature,
        Cursor::new(&catalog_bytes),
        true,
        false,
        false,
    )
    .map_err(|error| format!("verify newly signed native runtime catalog: {error}"))?;
    let signature = signature.into_string();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(signature_out)
        .map_err(|error| {
            format!(
                "create detached native runtime catalog signature {}: {error}",
                signature_out.display()
            )
        })?;
    output
        .write_all(signature.as_bytes())
        .and_then(|()| output.flush())
        .map_err(|error| {
            format!(
                "write detached native runtime catalog signature {}: {error}",
                signature_out.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisign::{KeyPair, SignatureBox};

    fn fixture_key(password: &str) -> (String, String) {
        let key_pair = KeyPair::generate_encrypted_keypair(Some(password.to_string())).unwrap();
        (
            key_pair
                .sk
                .to_box(Some("disposable catalog signing fixture"))
                .unwrap()
                .into_string(),
            key_pair.pk.to_box().unwrap().into_string(),
        )
    }

    #[test]
    fn signs_exact_catalog_bytes_with_matching_pinned_key() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = temp.path().join("native-runtimes.json");
        let signature = temp.path().join("native-runtimes.json.minisig");
        let public_key = temp.path().join("native-runtime-catalog.pub");
        let catalog_bytes = b"{\"schema_version\":2}\n";
        std::fs::write(&catalog, catalog_bytes).unwrap();
        let (secret_key, public_key_text) = fixture_key("fixture password");
        std::fs::write(&public_key, &public_key_text).unwrap();

        sign_catalog(
            &catalog,
            &signature,
            &public_key,
            &secret_key,
            "fixture password",
        )
        .unwrap();

        let public_key = PublicKeyBox::from_string(&public_key_text)
            .unwrap()
            .into_public_key()
            .unwrap();
        let signature =
            SignatureBox::from_string(&std::fs::read_to_string(signature).unwrap()).unwrap();
        minisign::verify(
            &public_key,
            &signature,
            Cursor::new(catalog_bytes),
            true,
            false,
            false,
        )
        .unwrap();
    }

    #[test]
    fn refuses_a_secret_key_that_does_not_match_the_trust_pin() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = temp.path().join("native-runtimes.json");
        let signature = temp.path().join("native-runtimes.json.minisig");
        let public_key = temp.path().join("native-runtime-catalog.pub");
        std::fs::write(&catalog, b"catalog").unwrap();
        let (secret_key, _) = fixture_key("fixture password");
        let (_, wrong_public_key) = fixture_key("other password");
        std::fs::write(&public_key, wrong_public_key).unwrap();

        let error = sign_catalog(
            &catalog,
            &signature,
            &public_key,
            &secret_key,
            "fixture password",
        )
        .unwrap_err();
        assert!(error.contains("does not match"), "{error}");
        assert!(!signature.exists());
    }

    #[test]
    fn wrong_password_never_writes_a_signature() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = temp.path().join("native-runtimes.json");
        let signature = temp.path().join("native-runtimes.json.minisig");
        let public_key = temp.path().join("native-runtime-catalog.pub");
        std::fs::write(&catalog, b"catalog").unwrap();
        let (secret_key, public_key_text) = fixture_key("correct disposable password");
        std::fs::write(&public_key, public_key_text).unwrap();
        let error = sign_catalog(
            &catalog,
            &signature,
            &public_key,
            &secret_key,
            "wrong password",
        )
        .unwrap_err();
        assert!(error.contains("decrypt"), "{error}");
        assert!(!signature.exists());
    }

    #[test]
    fn missing_or_partial_credentials_fail_closed() {
        assert!(signing_credentials_configured("", "").is_err());
        assert!(signing_credentials_configured("secret", "password").is_ok());
        assert!(signing_credentials_configured("secret", "").is_err());
        assert!(signing_credentials_configured("", "password").is_err());
    }
}
