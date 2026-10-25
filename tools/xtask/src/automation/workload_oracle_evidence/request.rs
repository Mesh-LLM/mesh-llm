use super::WorkloadClass;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub(crate) struct WriteRequest {
    pub(crate) output: PathBuf,
    pub(crate) comparison_log: PathBuf,
    pub(crate) model_class: String,
    pub(crate) smoke_lane: String,
    pub(crate) model_id: String,
    pub(crate) model_sha256: String,
    pub(crate) projector_path: Option<PathBuf>,
    pub(crate) candidate_executable: PathBuf,
    pub(crate) oracle_executable: PathBuf,
    pub(crate) pinned_patch_sha: String,
    pub(crate) work_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct VerifyRequest {
    pub(crate) evidence: PathBuf,
    pub(crate) model_class: WorkloadClass,
    pub(crate) smoke_lane: String,
    pub(crate) oracle_lane: String,
    pub(crate) model_id: String,
    pub(crate) model_path: PathBuf,
    pub(crate) projector_path: Option<PathBuf>,
    pub(crate) candidate_executable: PathBuf,
    pub(crate) oracle_executable: PathBuf,
    pub(crate) pinned_patch_sha: String,
}
