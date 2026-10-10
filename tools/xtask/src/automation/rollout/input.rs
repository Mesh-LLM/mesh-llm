use super::evidence::EvidenceFile;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(try_from = "String")]
pub(super) struct CommitSha(String);

impl TryFrom<String> for CommitSha {
    type Error = &'static str;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 40
            && value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            Ok(Self(value))
        } else {
            Err("commit SHA must be 40 lowercase hexadecimal characters")
        }
    }
}

impl CommitSha {
    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub(super) repository: PathBuf,
    pub(super) catalog_sha: CommitSha,
    pub(super) support_sha: CommitSha,
    pub(super) protected_sha: CommitSha,
    pub(super) source_sha: CommitSha,
    pub(super) planner_sha: CommitSha,
    pub(super) workspace_sha: CommitSha,
    pub(super) runner_policy_sha: CommitSha,
    pub(super) bootstrap: BootstrapObservation,
    pub(super) predecessors: Vec<Predecessor>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BootstrapObservation {
    pub(super) source_sha: CommitSha,
    pub(super) report: EvidenceFile,
    pub(super) binary: EvidenceFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Predecessor {
    pub(super) task: u8,
    pub(super) evidence: EvidenceFile,
}
