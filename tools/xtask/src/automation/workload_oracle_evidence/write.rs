use super::document::{self, Identity, sha256};
use super::{Error, WriteRequest};
use crate::repository::text::{splitlines, strip};
use std::fs;

pub(crate) fn write_evidence(request: &WriteRequest) -> Result<(), Error> {
    super::with_worker(|| write(request))
}

fn write(request: &WriteRequest) -> Result<(), Error> {
    let lane = request
        .smoke_lane
        .strip_suffix("-smoke")
        .ok_or_else(|| Error::SmokeLane(request.smoke_lane.clone()))?;
    let oracle_lane = format!("{lane}-oracle");
    let log = fs::read_to_string(&request.comparison_log).map_err(|source| Error::Io {
        path: request.comparison_log.clone(),
        source,
    })?;
    let prefix = format!("{} local-monolithic oracle passed: ", request.model_class);
    let comparison = splitlines(&log)
        .into_iter()
        .map(strip)
        .rfind(|line| !line.is_empty())
        .filter(|line| line.starts_with(&prefix))
        .ok_or(Error::ComparatorLog)?;
    let oracle_name = request
        .oracle_executable
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let identity = Identity {
        model_class: &request.model_class,
        smoke_lane: &request.smoke_lane,
        oracle_lane: &oracle_lane,
        model_id: &request.model_id,
        model_sha256: request.model_sha256.clone(),
        projector_sha256: request.projector_path.as_deref().map(sha256).transpose()?,
        oracle_executable: &oracle_name,
        oracle_executable_sha256: sha256(&request.oracle_executable)?,
        candidate_executable_sha256: sha256(&request.candidate_executable)?,
        pinned_patch_sha: &request.pinned_patch_sha,
    };
    let metrics = if request.model_class == "speech_synthesis" {
        Some(document::tts_result(
            &request.work_dir.join("tts-oracle-result.json"),
            &request.pinned_patch_sha,
        )?)
    } else {
        None
    };
    fs::write(&request.output, identity.encode(comparison, metrics)).map_err(|source| Error::Io {
        path: request.output.clone(),
        source,
    })
}
