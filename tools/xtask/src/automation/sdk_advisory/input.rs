use super::error::{Checked, Rejected};
use super::identity::{RunId, SourceSha};
use super::rows::ProductRow;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::io::Read;
use std::num::NonZeroU64;
use std::path::Path;

pub(super) const MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq, Deserialize)]
pub(super) struct Repository {
    pub(super) id: NonZeroU64,
    pub(super) full_name: String,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
pub(super) struct ProducerRun {
    pub(super) id: RunId,
    pub(super) run_attempt: NonZeroU64,
    pub(super) name: String,
    pub(super) path: String,
    pub(super) event: String,
    pub(super) head_branch: String,
    pub(super) head_sha: SourceSha,
    pub(super) status: String,
    pub(super) conclusion: Option<String>,
    pub(super) repository: Repository,
    pub(super) head_repository: Repository,
}

#[derive(Deserialize)]
pub(super) struct WorkflowEvent {
    pub(super) action: String,
    pub(super) repository: Repository,
    pub(super) workflow_run: ProducerRun,
}

#[derive(Deserialize)]
pub(super) struct ManualEvent {
    pub(super) repository: Repository,
    pub(super) inputs: ManualInputs,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ManualInputs {
    pub(super) producer_run_id: String,
    pub(super) product_row: ProductRow,
}

#[derive(Deserialize)]
pub(super) struct ArtifactList {
    pub(super) total_count: usize,
    pub(super) artifacts: Vec<Artifact>,
}

#[derive(Deserialize)]
pub(super) struct Artifact {
    pub(super) id: NonZeroU64,
    pub(super) name: String,
    pub(super) expired: bool,
    pub(super) digest: Option<String>,
    pub(super) workflow_run: Option<ArtifactRun>,
}

#[derive(Deserialize)]
pub(super) struct ArtifactRun {
    pub(super) id: RunId,
    pub(super) repository_id: NonZeroU64,
    pub(super) head_repository_id: NonZeroU64,
    pub(super) head_branch: String,
    pub(super) head_sha: SourceSha,
}

pub(super) fn parse<T: DeserializeOwned>(bytes: &[u8]) -> Checked<T> {
    if bytes.len() > MAX_BYTES {
        return Err(Rejected::InputSize);
    }
    serde_json::from_slice(bytes).map_err(Rejected::Input)
}

pub(super) fn read(path: &Path) -> Checked<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(Rejected::Read)?;
    let mut bytes = Vec::new();
    file.take(32 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(Rejected::Read)?;
    if bytes.len() > MAX_BYTES {
        return Err(Rejected::InputSize);
    }
    Ok(bytes)
}
