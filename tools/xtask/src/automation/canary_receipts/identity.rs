use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{cmp::Ordering, fmt, io::Read, path::Path};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub(crate) struct Digest(String);

impl Digest {
    pub(crate) fn of_bytes(bytes: &[u8]) -> Self {
        Self(hex::encode(Sha256::digest(bytes)))
    }

    pub(crate) fn of_file(path: &Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        Ok(Self(hex::encode(digest.finalize())))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Digest {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value))
        } else {
            Err("expected a lowercase SHA-256 digest")
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub(crate) struct Family(String);

impl Family {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Family {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if !value.is_empty()
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        {
            Ok(Self(value))
        } else {
            Err("unsafe family identity")
        }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub(crate) struct RunAttempt(String);

impl RunAttempt {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for RunAttempt {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.starts_with(|ch: char| ('1'..='9').contains(&ch))
            && value.bytes().all(|byte| byte.is_ascii_digit())
        {
            Ok(Self(value))
        } else {
            Err("invalid workflow run attempt")
        }
    }
}

impl Ord for RunAttempt {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.cmp(&other.0))
    }
}

impl PartialOrd for RunAttempt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Only the receipt-relevant projection; deserialization is not package verification.
#[derive(Debug, Deserialize)]
#[serde(remote = "Self")]
pub(crate) struct ProducerIdentity {
    pub(crate) candidate: String,
    pub(crate) branch: String,
    pub(crate) pass_id: String,
    pub(crate) run_id: String,
    pub(crate) run_attempt: RunAttempt,
}

impl<'de> Deserialize<'de> for ProducerIdentity {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        super::boundary::ordered_object_last_wins(deserializer, Self::deserialize)
    }
}

pub(crate) struct WorkflowRun {
    pub(crate) run_id: String,
    pub(crate) run_attempt: RunAttempt,
}
