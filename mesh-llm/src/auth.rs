//! Argon2id KDF derivation for mesh-llm owner keys.
//!
//! Pure-logic module: no I/O, no network, no CLI. All KDF parameters are
//! pinned as `const` and must never be made configurable.

use anyhow::{anyhow, Result};
use argon2::{Algorithm, Argon2, Params, Version};
use unicode_normalization::UnicodeNormalization;
use zeroize::Zeroize;

// KDF parameters — all const, never configurable at runtime.
// Salt is exactly 32 bytes (padded with null bytes).
const KDF_SALT: &[u8] = b"mesh-llm-owner-key-v1\0\0\0\0\0\0\0\0\0\0\0";
const KDF_M_COST: u32 = 47104; // 46 MiB — OWASP Argon2id minimum
const KDF_T_COST: u32 = 1;
const KDF_P_COST: u32 = 1;
const KDF_OUTPUT_LEN: usize = 32;
const KDF_ALGORITHM: Algorithm = Algorithm::Argon2id;
const KDF_VERSION: Version = Version::V0x13;

/// Normalize a passphrase for consistent KDF input.
///
/// Trims, lowercases, applies NFC, normalizes separators (hyphens become
/// spaces), and collapses consecutive spaces.
pub fn normalize_passphrase(input: &str) -> String {
    let nfc: String = input.trim().to_lowercase().nfc().collect();
    let mut result = String::with_capacity(nfc.len());
    let mut prev_space = false;
    for ch in nfc.chars() {
        if ch == ' ' || ch == '-' {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result
}

/// Derive a 32-byte owner key from a passphrase using Argon2id.
///
/// Rejects empty passphrases and single-word passphrases. The normalized
/// passphrase string is zeroized after the KDF completes.
pub fn derive_owner_key(passphrase: &str) -> Result<[u8; 32]> {
    if passphrase.trim().is_empty() {
        return Err(anyhow!("Passphrase cannot be empty"));
    }

    let mut normalized = normalize_passphrase(passphrase);

    let word_count = normalized.split_whitespace().count();
    if word_count < 2 {
        normalized.zeroize();
        return Err(anyhow!("Passphrase must contain at least 2 words"));
    }

    let params = Params::new(KDF_M_COST, KDF_T_COST, KDF_P_COST, Some(KDF_OUTPUT_LEN))
        .map_err(|e| anyhow!("Failed to build Argon2 params: {}", e))?;
    let argon2 = Argon2::new(KDF_ALGORITHM, KDF_VERSION, params);

    let mut output = [0u8; KDF_OUTPUT_LEN];
    argon2
        .hash_password_into(normalized.as_bytes(), KDF_SALT, &mut output)
        .map_err(|e| anyhow!("Argon2id KDF failed: {}", e))?;

    // Zeroize the normalized passphrase string after use
    normalized.zeroize();

    Ok(output)
}

pub fn format_key_hex(key: &[u8; 32]) -> String {
    hex::encode(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_determinism() {
        let key_bytes = derive_owner_key("alpha bravo charlie delta echo foxtrot golf")
            .expect("KDF should succeed");
        let hex = format_key_hex(&key_bytes);
        assert_eq!(
            hex, "1133f9fd57a8789af0936387090966ec76ee37f7cdf6533afc833e06a8bbf8e4",
            "KDF output changed — params or salt may have been modified"
        );
    }

    #[test]
    fn test_iroh_secret_key_roundtrip() {
        use iroh::SecretKey;
        let key_bytes =
            derive_owner_key("fire lake mountain river cloud").expect("KDF should succeed");
        let secret_key = SecretKey::from_bytes(&key_bytes);
        let _public_key = secret_key.public();
        let fp = crate::identity::owner_fingerprint_from_key_material(key_bytes);
        assert_eq!(fp.len(), 64, "fingerprint must be 64 hex chars");
        assert!(
            fp.chars().all(|c| c.is_ascii_hexdigit()),
            "fingerprint must be lowercase hex"
        );
    }

    #[test]
    fn test_fingerprint_compatibility() {
        let passphrase = "alpha bravo charlie delta echo foxtrot golf";
        let key_bytes = derive_owner_key(passphrase).expect("derivation failed");
        let fp = crate::identity::owner_fingerprint_from_key_material(key_bytes);
        assert_eq!(fp.len(), 64, "fingerprint must be 64 hex chars");
        assert!(
            fp.chars().all(|c| c.is_ascii_hexdigit()),
            "fingerprint must be lowercase hex"
        );
        assert!(!fp.is_empty());
    }

    #[test]
    fn test_determinism_across_calls() {
        let key1 = derive_owner_key("same passphrase here").expect("KDF should succeed");
        let key2 = derive_owner_key("same passphrase here").expect("KDF should succeed");
        assert_eq!(
            key1, key2,
            "Same passphrase must always produce the same key"
        );
    }

    #[test]
    fn test_normalization_equivalence() {
        let key1 = derive_owner_key("  Alpha  BRAVO  ").expect("KDF should succeed");
        let key2 = derive_owner_key("alpha bravo").expect("KDF should succeed");
        assert_eq!(
            key1, key2,
            "Normalized passphrases must produce identical keys"
        );
    }

    #[test]
    fn test_empty_passphrase_rejected() {
        let result = derive_owner_key("");
        assert!(result.is_err(), "Empty passphrase must be rejected");
        assert!(
            result.unwrap_err().to_string().contains("empty"),
            "Error message should mention 'empty'"
        );
    }

    #[test]
    fn test_single_word_rejected() {
        let result = derive_owner_key("singleword");
        assert!(result.is_err(), "Single-word passphrase must be rejected");
        assert!(
            result.unwrap_err().to_string().contains("2 words"),
            "Error message should mention word count requirement"
        );
    }

    #[test]
    fn test_hyphens_equivalent_to_spaces() {
        let key_spaces = derive_owner_key("alpha bravo charlie").expect("spaces should work");
        let key_hyphens = derive_owner_key("alpha-bravo-charlie").expect("hyphens should work");
        assert_eq!(
            key_spaces, key_hyphens,
            "Hyphens and spaces must produce identical keys"
        );
    }

    #[test]
    fn test_hyphenated_passphrase_accepted() {
        let result = derive_owner_key("mesh-test-default-passphrase");
        assert!(
            result.is_ok(),
            "Hyphenated multi-word passphrase must be accepted"
        );
    }

    #[test]
    fn test_different_passphrases_different_keys() {
        let key1 = derive_owner_key("first passphrase here").expect("KDF should succeed");
        let key2 = derive_owner_key("second passphrase here").expect("KDF should succeed");
        assert_ne!(
            key1, key2,
            "Different passphrases must produce different keys"
        );
    }
}
