use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(transparent)]
pub(super) struct RunId(pub(super) NonZeroU64);

impl RunId {
    pub(super) fn parse(raw: &str) -> Result<Self, &'static str> {
        if raw.starts_with('0') || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("producer_run_id must be a positive decimal integer");
        }
        raw.parse().map(Self).map_err(|_| "invalid producer_run_id")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(try_from = "String", into = "String")]
pub(super) struct SourceSha(String);

impl TryFrom<String> for SourceSha {
    type Error = &'static str;

    fn try_from(raw: String) -> Result<Self, Self::Error> {
        if lower_hex(&raw, 40) {
            Ok(Self(raw))
        } else {
            Err("source SHA must contain exactly 40 lowercase hex digits")
        }
    }
}

impl From<SourceSha> for String {
    fn from(value: SourceSha) -> Self {
        value.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub(super) struct ArtifactDigest(String);

impl ArtifactDigest {
    pub(super) fn parse(raw: &str) -> Option<Self> {
        raw.strip_prefix("sha256:")
            .filter(|hex| lower_hex(hex, 64))
            .map(|_| Self(raw.to_owned()))
    }
}

fn lower_hex(raw: &str, length: usize) -> bool {
    raw.len() == length
        && raw
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}
