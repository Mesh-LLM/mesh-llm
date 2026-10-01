use super::document::{EvidenceInput, Identity, sha256};
use super::{Error, VerifyRequest};
use std::ffi::OsStr;

pub(crate) fn verify_evidence(request: &VerifyRequest) -> Result<(), Error> {
    super::with_worker(|| verify(request))
}

fn verify(request: &VerifyRequest) -> Result<(), Error> {
    let class = request.model_class;
    if class.requires_projector() && request.projector_path.is_none() {
        return Err(Error::Projector(class.name()));
    }
    let evidence = EvidenceInput::load(&request.evidence)?;
    let expected = Identity {
        model_class: class.name(),
        smoke_lane: &request.smoke_lane,
        oracle_lane: &request.oracle_lane,
        model_id: &request.model_id,
        model_sha256: sha256(&request.model_path)?,
        projector_sha256: request.projector_path.as_deref().map(sha256).transpose()?,
        oracle_executable: class.executable(),
        oracle_executable_sha256: sha256(&request.oracle_executable)?,
        candidate_executable_sha256: sha256(&request.candidate_executable)?,
        pinned_patch_sha: &request.pinned_patch_sha,
    };
    evidence.verify_identity(&expected)?;
    if request.oracle_executable.file_name() != Some(OsStr::new(class.executable())) {
        return Err(Error::Executable);
    }
    evidence.verify_comparison(class)
}
