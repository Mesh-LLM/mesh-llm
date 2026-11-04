//! Owner-control bearer credential for the local management API.
//!
//! Loopback reachability is not owner identity: any local OS account that can
//! reach the management listener could previously drive the owner-control
//! routes, and the host signed owner-control traffic with the server-side
//! owner keypair on the caller's behalf. The routes now additionally require
//! this host-generated token, presented as `Authorization: Bearer <token>`.
//!
//! The token is a 32-byte random value persisted next to the owner keystore
//! (0600 on Unix) so restarts keep accepting already-configured callers, and
//! it is held in memory behind [`Zeroizing`].

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use rand::RngExt;
use zeroize::Zeroizing;

use crate::crypto::default_keystore_path;

/// File name of the persisted token, stored beside the owner keystore.
pub(crate) const OWNER_CONTROL_TOKEN_FILE: &str = "owner-control-token-v2";

const OWNER_CONTROL_TOKEN_BYTES: usize = 32;

/// Generates a fresh URL-safe owner-control token from the OS RNG.
pub(crate) fn generate_owner_control_token() -> Zeroizing<String> {
    let mut bytes = [0u8; OWNER_CONTROL_TOKEN_BYTES];
    rand::rng().fill(&mut bytes);
    Zeroizing::new(hex::encode(bytes))
}

/// Directory that holds owner-local secrets: the owner keystore's directory,
/// falling back to the default keystore location when no keystore is selected.
pub(crate) fn owner_control_token_path(owner_key_path: Option<&Path>) -> Result<PathBuf> {
    let keystore = match owner_key_path {
        Some(path) => path.to_path_buf(),
        None => default_keystore_path().context("resolving the default owner keystore path")?,
    };
    let dir = keystore.parent().with_context(|| {
        format!(
            "owner keystore path {} has no parent directory",
            keystore.display()
        )
    })?;
    Ok(dir.join(OWNER_CONTROL_TOKEN_FILE))
}

/// Loads the persisted owner-control token, creating and persisting one on
/// first use.
pub(crate) fn load_or_create_owner_control_token(
    owner_key_path: Option<&Path>,
) -> Result<Zeroizing<String>> {
    let path = owner_control_token_path(owner_key_path)?;
    if let Some(token) = read_owner_control_token(&path)? {
        return Ok(token);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating owner-control token directory {}", dir.display()))?;
    }
    loop {
        let token = generate_owner_control_token();
        match write_owner_control_token_new(&path, token.as_bytes()) {
            Ok(()) => return Ok(token),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                // A concurrent host start won the race; adopt its token.
                if let Some(existing) = read_owner_control_token(&path)? {
                    return Ok(existing);
                }
            }
            Err(error) => {
                return Err(anyhow::Error::new(error)
                    .context(format!("writing owner-control token {}", path.display())));
            }
        }
    }
}

/// Compares a presented credential with the expected token without leaking
/// per-byte timing to a local attacker.
pub(crate) fn owner_control_tokens_match(expected: &str, presented: &str) -> bool {
    let (expected, presented) = (expected.as_bytes(), presented.as_bytes());
    if expected.len() != presented.len() {
        return false;
    }
    let mut difference = 0u8;
    for (left, right) in expected.iter().zip(presented) {
        difference |= left ^ right;
    }
    difference == 0
}

fn read_owner_control_token(path: &Path) -> Result<Option<Zeroizing<String>>> {
    match mesh_llm_config::read_private_file(path) {
        Ok(raw) => {
            let raw = Zeroizing::new(raw);
            let token = std::str::from_utf8(&raw)
                .context("owner-control token is not UTF-8")?
                .trim();
            anyhow::ensure!(
                !token.is_empty(),
                "owner-control token file {} is empty",
                path.display()
            );
            Ok(Some(Zeroizing::new(token.to_string())))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(anyhow::Error::new(error)
            .context(format!("reading owner-control token {}", path.display()))),
    }
}

fn write_owner_control_token_new(path: &Path, token: &[u8]) -> std::io::Result<()> {
    let mut contents = Zeroizing::new(Vec::with_capacity(token.len() + 1));
    contents.extend_from_slice(token);
    contents.push(b'\n');
    mesh_llm_config::create_private_file_if_absent(path, &contents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_preserves_existing_token_and_removes_temporary_file() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("token");
        write_owner_control_token_new(&path, b"first").expect("publish token");
        let error = write_owner_control_token_new(&path, b"second").expect_err("existing token");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read token"),
            "first\n"
        );
        assert_eq!(
            std::fs::read_dir(dir.path())
                .expect("read directory")
                .count(),
            1
        );
    }

    #[test]
    fn generated_tokens_are_64_hex_characters() {
        let token = generate_owner_control_token();
        assert_eq!(token.len(), OWNER_CONTROL_TOKEN_BYTES * 2);
        assert!(token.chars().all(|value| value.is_ascii_hexdigit()));
        assert_ne!(token, generate_owner_control_token());
    }

    #[test]
    fn token_file_is_persisted_once_and_reused() {
        let temp = tempfile::tempdir().unwrap();
        let keystore_path = temp.path().join("identity").join("owner-keystore.json");
        let first = load_or_create_owner_control_token(Some(&keystore_path)).unwrap();
        let second = load_or_create_owner_control_token(Some(&keystore_path)).unwrap();
        assert_eq!(first, second);
        let path = owner_control_token_path(Some(&keystore_path)).unwrap();
        assert_eq!(path.parent().unwrap(), keystore_path.parent().unwrap());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn token_path_falls_back_to_the_default_keystore_directory() {
        let path = owner_control_token_path(None).unwrap();
        assert_eq!(path.file_name().unwrap(), OWNER_CONTROL_TOKEN_FILE);
        assert_eq!(
            path.parent().unwrap(),
            default_keystore_path().unwrap().parent().unwrap()
        );
    }

    #[test]
    fn matching_uses_constant_time_comparison() {
        assert!(owner_control_tokens_match("abcd", "abcd"));
        assert!(!owner_control_tokens_match("abcd", "abce"));
        assert!(!owner_control_tokens_match("abcd", "abc"));
        assert!(!owner_control_tokens_match("abcd", "abcde"));
        assert!(!owner_control_tokens_match("", "a"));
    }
}
