use iroh::SecretKey;
use sha2::{Digest, Sha256};

/// `hex(sha256(owner_public_key))` — 64-char lowercase hex owner fingerprint.
pub(crate) fn owner_fingerprint_from_key_material(owner_key_material: [u8; 32]) -> String {
    let owner_key = SecretKey::from_bytes(&owner_key_material);
    hex::encode(Sha256::digest(owner_key.public().as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_64_hex_chars() {
        let fp = owner_fingerprint_from_key_material([0xAB; 32]);
        assert_eq!(fp.len(), 64, "fingerprint must be 64 hex chars");
        assert!(
            fp.chars().all(|c| c.is_ascii_hexdigit()),
            "fingerprint must be lowercase hex"
        );
    }

    #[test]
    fn fingerprint_is_deterministic() {
        let fp1 = owner_fingerprint_from_key_material([0x01; 32]);
        let fp2 = owner_fingerprint_from_key_material([0x01; 32]);
        assert_eq!(fp1, fp2, "same key material must produce same fingerprint");
    }

    #[test]
    fn different_key_material_produces_different_fingerprints() {
        let fp1 = owner_fingerprint_from_key_material([0x01; 32]);
        let fp2 = owner_fingerprint_from_key_material([0x02; 32]);
        assert_ne!(fp1, fp2);
    }
}
