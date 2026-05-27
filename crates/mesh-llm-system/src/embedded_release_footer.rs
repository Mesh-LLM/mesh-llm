use sha2::{Digest, Sha256};

/// Fixed footer appended after the raw signed payload bytes.
///
/// Layout is frozen as:
/// `[payload bytes][footer]`
///
/// Footer (24 bytes):
/// - magic `MLATTEST` (8 ASCII bytes)
/// - format version `1` (`u32` little-endian)
/// - payload length (`u64` little-endian)
/// - reserved `0` (`u32` little-endian)
pub const EMBEDDED_RELEASE_FOOTER_LEN: usize = 24;
pub const EMBEDDED_RELEASE_FOOTER_MAGIC: &[u8; 8] = b"MLATTEST";
pub const EMBEDDED_RELEASE_FOOTER_VERSION: u32 = 1;
const EMBEDDED_RELEASE_FOOTER_PREFIX_LEN: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddedReleaseFooterStatus {
    Missing,
    Valid,
    Invalid,
}

impl EmbeddedReleaseFooterStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Valid => "valid",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddedReleasePayloadSummary {
    /// Expected digest of the base executable bytes before payload/footer append.
    pub artifact_digest: String,
}

pub trait EmbeddedReleasePayloadVerifier {
    type Error: std::fmt::Display;

    /// Verify the exact embedded payload bytes as stamped into the binary.
    ///
    /// Callers must treat `payload_bytes` as the signed object bytes directly.
    /// They must not parse and reserialize before signature verification.
    fn verify_payload(
        &self,
        payload_bytes: &[u8],
    ) -> Result<EmbeddedReleasePayloadSummary, Self::Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddedReleaseFooter<'a> {
    pub base_bytes: &'a [u8],
    pub payload_bytes: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddedReleaseFooterVerification<'a> {
    pub status: EmbeddedReleaseFooterStatus,
    pub footer: Option<EmbeddedReleaseFooter<'a>>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmbeddedReleaseFooterError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u32),
    NonZeroReserved(u32),
    PayloadLengthOverflow(u64),
}

impl std::fmt::Display for EmbeddedReleaseFooterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "embedded release footer is truncated"),
            Self::BadMagic => write!(f, "embedded release footer magic is invalid"),
            Self::UnsupportedVersion(version) => {
                write!(
                    f,
                    "embedded release footer version {version} is unsupported"
                )
            }
            Self::NonZeroReserved(value) => {
                write!(
                    f,
                    "embedded release footer reserved field must be 0, got {value}"
                )
            }
            Self::PayloadLengthOverflow(length) => write!(
                f,
                "embedded release payload length {length} exceeds binary size"
            ),
        }
    }
}

impl std::error::Error for EmbeddedReleaseFooterError {}

pub fn stamp_embedded_release_payload(
    binary_bytes: &[u8],
    payload_bytes: &[u8],
) -> Result<Vec<u8>, EmbeddedReleaseFooterError> {
    let base_bytes = strip_embedded_release_footer(binary_bytes)?.to_vec();
    let mut stamped =
        Vec::with_capacity(base_bytes.len() + payload_bytes.len() + EMBEDDED_RELEASE_FOOTER_LEN);
    stamped.extend_from_slice(&base_bytes);
    stamped.extend_from_slice(payload_bytes);
    stamped.extend_from_slice(EMBEDDED_RELEASE_FOOTER_MAGIC);
    stamped.extend_from_slice(&EMBEDDED_RELEASE_FOOTER_VERSION.to_le_bytes());
    stamped.extend_from_slice(&(payload_bytes.len() as u64).to_le_bytes());
    stamped.extend_from_slice(&0u32.to_le_bytes());
    Ok(stamped)
}

pub fn strip_embedded_release_footer(
    binary_bytes: &[u8],
) -> Result<&[u8], EmbeddedReleaseFooterError> {
    Ok(
        read_embedded_release_footer(binary_bytes)?
            .map_or(binary_bytes, |footer| footer.base_bytes),
    )
}

pub fn read_embedded_release_footer(
    binary_bytes: &[u8],
) -> Result<Option<EmbeddedReleaseFooter<'_>>, EmbeddedReleaseFooterError> {
    if binary_bytes.len() < EMBEDDED_RELEASE_FOOTER_LEN {
        return detect_truncated_footer_tail(binary_bytes)
            .map(|_| None)
            .map_err(|()| EmbeddedReleaseFooterError::Truncated);
    }

    let footer_start = binary_bytes.len() - EMBEDDED_RELEASE_FOOTER_LEN;
    let footer_bytes = &binary_bytes[footer_start..];

    if &footer_bytes[..8] != EMBEDDED_RELEASE_FOOTER_MAGIC {
        if looks_like_corrupted_footer(binary_bytes.len(), footer_bytes) {
            return Err(EmbeddedReleaseFooterError::BadMagic);
        }
        return detect_truncated_footer_tail(binary_bytes)
            .map(|_| None)
            .map_err(|()| EmbeddedReleaseFooterError::Truncated);
    }

    let version = u32::from_le_bytes(footer_bytes[8..12].try_into().expect("slice length"));
    if version != EMBEDDED_RELEASE_FOOTER_VERSION {
        return Err(EmbeddedReleaseFooterError::UnsupportedVersion(version));
    }

    let payload_len = u64::from_le_bytes(footer_bytes[12..20].try_into().expect("slice length"));
    let reserved = u32::from_le_bytes(footer_bytes[20..24].try_into().expect("slice length"));
    if reserved != 0 {
        return Err(EmbeddedReleaseFooterError::NonZeroReserved(reserved));
    }

    let payload_len = usize::try_from(payload_len)
        .map_err(|_| EmbeddedReleaseFooterError::PayloadLengthOverflow(payload_len))?;
    if footer_start < payload_len {
        return Err(EmbeddedReleaseFooterError::PayloadLengthOverflow(
            payload_len as u64,
        ));
    }

    let payload_start = footer_start - payload_len;
    Ok(Some(EmbeddedReleaseFooter {
        base_bytes: &binary_bytes[..payload_start],
        payload_bytes: &binary_bytes[payload_start..footer_start],
    }))
}

pub fn verify_embedded_release_footer<'a, V>(
    binary_bytes: &'a [u8],
    verifier: &V,
) -> EmbeddedReleaseFooterVerification<'a>
where
    V: EmbeddedReleasePayloadVerifier,
{
    let footer = match read_embedded_release_footer(binary_bytes) {
        Ok(Some(footer)) => footer,
        Ok(None) => {
            return EmbeddedReleaseFooterVerification {
                status: EmbeddedReleaseFooterStatus::Missing,
                footer: None,
                error: None,
            };
        }
        Err(error) => {
            return EmbeddedReleaseFooterVerification {
                status: EmbeddedReleaseFooterStatus::Invalid,
                footer: None,
                error: Some(error.to_string()),
            };
        }
    };

    let payload_summary = match verifier.verify_payload(footer.payload_bytes) {
        Ok(summary) => summary,
        Err(error) => {
            return EmbeddedReleaseFooterVerification {
                status: EmbeddedReleaseFooterStatus::Invalid,
                footer: Some(footer),
                error: Some(error.to_string()),
            };
        }
    };

    let base_digest = format!("sha256:{}", hex::encode(Sha256::digest(footer.base_bytes)));
    if payload_summary.artifact_digest != base_digest {
        return EmbeddedReleaseFooterVerification {
            status: EmbeddedReleaseFooterStatus::Invalid,
            footer: Some(footer),
            error: Some(format!(
                "artifact digest mismatch: expected {}, computed {}",
                payload_summary.artifact_digest, base_digest
            )),
        };
    }

    EmbeddedReleaseFooterVerification {
        status: EmbeddedReleaseFooterStatus::Valid,
        footer: Some(footer),
        error: None,
    }
}

fn detect_truncated_footer_tail(binary_bytes: &[u8]) -> Result<(), ()> {
    let probe_len = binary_bytes.len().min(EMBEDDED_RELEASE_FOOTER_LEN - 1);
    let tail = &binary_bytes[binary_bytes.len().saturating_sub(probe_len)..];
    let footer_magic = *EMBEDDED_RELEASE_FOOTER_MAGIC;
    let footer_prefix = footer_prefix_bytes();

    for start in 0..tail.len() {
        let suffix = &tail[start..];
        if suffix.len() < EMBEDDED_RELEASE_FOOTER_MAGIC.len() {
            continue;
        }
        if suffix.len() < footer_prefix.len() {
            if footer_prefix.starts_with(suffix) {
                return Err(());
            }
            continue;
        }
        if footer_magic == suffix[..EMBEDDED_RELEASE_FOOTER_MAGIC.len()] {
            return Err(());
        }
        if suffix.starts_with(&footer_prefix) {
            return Err(());
        }
    }

    Ok(())
}

fn footer_prefix_bytes() -> [u8; EMBEDDED_RELEASE_FOOTER_PREFIX_LEN] {
    let mut prefix = [0u8; EMBEDDED_RELEASE_FOOTER_PREFIX_LEN];
    prefix[..8].copy_from_slice(EMBEDDED_RELEASE_FOOTER_MAGIC);
    prefix[8..12].copy_from_slice(&EMBEDDED_RELEASE_FOOTER_VERSION.to_le_bytes());
    prefix
}

fn looks_like_corrupted_footer(binary_len: usize, footer_bytes: &[u8]) -> bool {
    let version = u32::from_le_bytes(footer_bytes[8..12].try_into().expect("slice length"));
    let reserved = u32::from_le_bytes(footer_bytes[20..24].try_into().expect("slice length"));
    let payload_len = u64::from_le_bytes(footer_bytes[12..20].try_into().expect("slice length"));
    let Ok(payload_len) = usize::try_from(payload_len) else {
        return false;
    };
    let footer_start = binary_len - EMBEDDED_RELEASE_FOOTER_LEN;

    let plausible_payload = footer_start >= payload_len;
    let canonical_version = version == EMBEDDED_RELEASE_FOOTER_VERSION;
    let canonical_reserved = reserved == 0;

    plausible_payload && (canonical_version || canonical_reserved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    struct TestPayload {
        artifact_digest: String,
        signature: String,
    }

    struct TestPayloadVerifier {
        expected_payload_bytes: Vec<u8>,
    }

    impl EmbeddedReleasePayloadVerifier for TestPayloadVerifier {
        type Error = &'static str;

        fn verify_payload(
            &self,
            payload_bytes: &[u8],
        ) -> Result<EmbeddedReleasePayloadSummary, Self::Error> {
            if payload_bytes != self.expected_payload_bytes.as_slice() {
                return Err("payload signature verification failed");
            }
            let payload: TestPayload =
                serde_json::from_slice(payload_bytes).map_err(|_| "payload JSON is invalid")?;
            if payload.signature != "test-signature" {
                return Err("payload signature verification failed");
            }
            Ok(EmbeddedReleasePayloadSummary {
                artifact_digest: payload.artifact_digest,
            })
        }
    }

    fn payload_bytes_for(base_bytes: &[u8], signature: &str) -> Vec<u8> {
        serde_json::to_vec(&TestPayload {
            artifact_digest: format!("sha256:{}", hex::encode(Sha256::digest(base_bytes))),
            signature: signature.to_string(),
        })
        .expect("test payload serializes")
    }

    #[test]
    fn embedded_release_footer_round_trip_and_restamp() {
        let base_bytes = b"mesh-llm release binary";
        let payload_one = payload_bytes_for(base_bytes, "test-signature");
        let stamped_once =
            stamp_embedded_release_payload(base_bytes, &payload_one).expect("stamping should work");

        let footer = read_embedded_release_footer(&stamped_once)
            .expect("footer should parse")
            .expect("footer should exist");
        assert_eq!(footer.base_bytes, base_bytes);
        assert_eq!(footer.payload_bytes, payload_one.as_slice());
        assert_eq!(
            stamped_once.len(),
            base_bytes.len() + payload_one.len() + EMBEDDED_RELEASE_FOOTER_LEN
        );

        let verified_once = verify_embedded_release_footer(
            &stamped_once,
            &TestPayloadVerifier {
                expected_payload_bytes: payload_one.clone(),
            },
        );
        assert_eq!(verified_once.status, EmbeddedReleaseFooterStatus::Valid);
        assert_eq!(verified_once.status.as_str(), "valid");

        let payload_two = payload_bytes_for(base_bytes, "test-signature");
        let stamped_twice = stamp_embedded_release_payload(&stamped_once, &payload_two)
            .expect("re-stamping should work");
        let restamped_footer = read_embedded_release_footer(&stamped_twice)
            .expect("restamped footer should parse")
            .expect("restamped footer should exist");
        assert_eq!(restamped_footer.base_bytes, base_bytes);
        assert_eq!(restamped_footer.payload_bytes, payload_two.as_slice());
        assert_eq!(
            stamped_twice.len(),
            base_bytes.len() + payload_two.len() + EMBEDDED_RELEASE_FOOTER_LEN,
            "re-stamping must replace the old trailer instead of appending another one"
        );
    }

    #[test]
    fn embedded_release_footer_corruption_reports_invalid() {
        let base_bytes = b"mesh-llm release binary";
        let payload = payload_bytes_for(base_bytes, "test-signature");
        let stamped =
            stamp_embedded_release_payload(base_bytes, &payload).expect("stamping should work");
        let verifier = TestPayloadVerifier {
            expected_payload_bytes: payload.clone(),
        };

        let mut digest_mismatch = stamped.clone();
        digest_mismatch[0] ^= 0x01;
        let digest_result = verify_embedded_release_footer(&digest_mismatch, &verifier);
        assert_eq!(digest_result.status, EmbeddedReleaseFooterStatus::Invalid);
        assert_eq!(digest_result.status.as_str(), "invalid");
        assert!(
            digest_result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("artifact digest mismatch"))
        );

        let mut bad_magic = stamped.clone();
        let magic_index = bad_magic.len() - EMBEDDED_RELEASE_FOOTER_LEN;
        bad_magic[magic_index] ^= 0x01;
        let bad_magic_result = verify_embedded_release_footer(&bad_magic, &verifier);
        assert_eq!(
            bad_magic_result.status,
            EmbeddedReleaseFooterStatus::Invalid
        );

        let truncated = stamped[..stamped.len() - 1].to_vec();
        let truncated_result = verify_embedded_release_footer(&truncated, &verifier);
        assert_eq!(
            truncated_result.status,
            EmbeddedReleaseFooterStatus::Invalid
        );
        assert!(
            truncated_result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("truncated"))
        );
    }

    #[test]
    fn embedded_release_footer_missing_status_is_stable() {
        let verification = verify_embedded_release_footer(
            b"plain binary without embedded release payload",
            &TestPayloadVerifier {
                expected_payload_bytes: Vec::new(),
            },
        );
        assert_eq!(verification.status, EmbeddedReleaseFooterStatus::Missing);
        assert_eq!(verification.status.as_str(), "missing");
        assert!(verification.footer.is_none());
        assert!(verification.error.is_none());
    }

    #[test]
    fn embedded_release_footer_plain_binary_suffix_is_missing() {
        let verification = verify_embedded_release_footer(
            b"plain binary that happens to end with M",
            &TestPayloadVerifier {
                expected_payload_bytes: Vec::new(),
            },
        );
        assert_eq!(verification.status, EmbeddedReleaseFooterStatus::Missing);
    }

    #[test]
    fn embedded_release_footer_multi_field_corruption_is_invalid() {
        let base_bytes = b"mesh-llm release binary";
        let payload = payload_bytes_for(base_bytes, "test-signature");
        let mut stamped =
            stamp_embedded_release_payload(base_bytes, &payload).expect("stamping should work");
        let footer_start = stamped.len() - EMBEDDED_RELEASE_FOOTER_LEN;
        stamped[footer_start] ^= 0x01;
        stamped[footer_start + 20] ^= 0x01;

        let verification = verify_embedded_release_footer(
            &stamped,
            &TestPayloadVerifier {
                expected_payload_bytes: payload,
            },
        );
        assert_eq!(verification.status, EmbeddedReleaseFooterStatus::Invalid);
        assert!(
            verification
                .error
                .as_deref()
                .is_some_and(|error| error.contains("invalid"))
        );
    }

    #[test]
    fn embedded_release_footer_restamp_rejects_malformed_existing_footer() {
        let base_bytes = b"mesh-llm release binary";
        let payload = payload_bytes_for(base_bytes, "test-signature");
        let mut stamped =
            stamp_embedded_release_payload(base_bytes, &payload).expect("stamping should work");
        let footer_start = stamped.len() - EMBEDDED_RELEASE_FOOTER_LEN;
        stamped[footer_start + 20] ^= 0x01;

        let restamp = stamp_embedded_release_payload(&stamped, &payload);
        assert_eq!(restamp, Err(EmbeddedReleaseFooterError::NonZeroReserved(1)));
    }
}
