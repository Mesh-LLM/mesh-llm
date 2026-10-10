//! Exact closed serialized Options contract of the pinned layer-job helper, not a network serializer.
use super::contract::Input;
use crate::automation::hf_certify::admission;
use serde::Serialize;
use std::path::{Path, PathBuf};
#[derive(Serialize)]
struct RepoOptions {
    repo: String,
    credential_file: PathBuf,
    output_directory: PathBuf,
    timeout_seconds: u64,
    confirm: bool,
}
pub(super) fn repo_hash(
    input: &Input,
    out: &Path,
    seconds: u64,
) -> crate::command::DynResult<String> {
    Ok(admission::digest(&serde_json::to_vec(&RepoOptions {
        repo: input.target_repo.clone(),
        credential_file: input.credential_file.clone().ok_or("generic credential")?,
        output_directory: out.into(),
        timeout_seconds: seconds,
        confirm: true,
    })?))
}
