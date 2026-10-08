use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: PathBuf,
    pub path_in_repo: String,
    pub sha256: String,
    pub byte_size: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub schema_version: u32,
    pub repo: String,
    pub parent_commit: String,
    pub shards: Vec<Artifact>,
    pub sidecars: Vec<Artifact>,
    pub credential_file: Option<PathBuf>,
    pub execution_timeout_ms: u64,
}
#[derive(Serialize)]
pub struct Receipt<'a> {
    pub schema_version: u32,
    pub request_sha256: &'a str,
    pub status: &'a str,
    pub publication: Option<&'a super::super::model_publication::Receipt>,
    pub source_custody_verified: bool,
    pub error: Option<&'a str>,
}
