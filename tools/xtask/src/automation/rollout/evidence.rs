use super::{Checked, Issue, reject};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const DOCUMENT_LIMIT: u64 = 16 * 1024 * 1024;
const EXECUTABLE_LIMIT: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(try_from = "String")]
pub(super) struct Sha256Hex(String);

impl TryFrom<String> for Sha256Hex {
    type Error = &'static str;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            Ok(Self(value))
        } else {
            Err("SHA-256 must be 64 lowercase hexadecimal characters")
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EvidenceFile {
    pub(super) path: PathBuf,
    pub(super) sha256: Sha256Hex,
}

impl EvidenceFile {
    pub(super) fn from_bytes(path: PathBuf, bytes: &[u8]) -> Self {
        Self {
            path,
            sha256: Sha256Hex(hex::encode(Sha256::digest(bytes))),
        }
    }

    pub(super) fn read(&self, base: &Path) -> Checked<Vec<u8>> {
        let bytes = read(&base.join(&self.path))?;
        self.check_digest(&hex::encode(Sha256::digest(&bytes)))?;
        Ok(bytes)
    }

    pub(super) fn verify_binary(&self, base: &Path) -> Checked<PathBuf> {
        let path = base.join(&self.path);
        let mut file = regular_file(&path, EXECUTABLE_LIMIT)?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 65536];
        let mut remaining = EXECUTABLE_LIMIT;
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| reject(Issue::Evidence, format!("{}: {error}", path.display())))?;
            if count == 0 {
                break;
            }
            remaining = remaining
                .checked_sub(
                    u64::try_from(count)
                        .map_err(|error| reject(Issue::Evidence, error.to_string()))?,
                )
                .ok_or_else(|| reject(Issue::Evidence, "executable exceeds byte limit"))?;
            digest.update(&buffer[..count]);
        }
        self.check_digest(&hex::encode(digest.finalize()))?;
        path.canonicalize()
            .map_err(|error| reject(Issue::Evidence, format!("{}: {error}", path.display())))
    }

    fn check_digest(&self, actual: &str) -> Checked<()> {
        if actual != self.sha256.0 {
            return Err(reject(
                Issue::Evidence,
                format!("SHA-256 mismatch: {}", self.path.display()),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub(super) struct CatalogBinding {
    pub(super) path: &'static str,
    pub(super) sha256: String,
}

fn regular_file(path: &Path, limit: u64) -> Checked<File> {
    let metadata = path
        .symlink_metadata()
        .map_err(|error| reject(Issue::Evidence, format!("{}: {error}", path.display())))?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(reject(
            Issue::Evidence,
            format!("not a bounded regular file: {}", path.display()),
        ));
    }
    File::open(path)
        .map_err(|error| reject(Issue::Evidence, format!("{}: {error}", path.display())))
}

pub(super) fn read(path: &Path) -> Checked<Vec<u8>> {
    let file = regular_file(path, DOCUMENT_LIMIT)?;
    let mut bytes = Vec::new();
    file.take(DOCUMENT_LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| reject(Issue::Evidence, format!("{}: {error}", path.display())))?;
    if u64::try_from(bytes.len()).map_err(|error| reject(Issue::Evidence, error.to_string()))?
        > DOCUMENT_LIMIT
    {
        return Err(reject(Issue::Evidence, "document exceeds byte limit"));
    }
    Ok(bytes)
}
