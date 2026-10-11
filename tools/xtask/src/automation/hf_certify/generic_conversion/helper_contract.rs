//! Exact closed serialized Options contract of the pinned layer-job helper, not a network serializer.
use super::contract::Input;
use crate::automation::hf_certify::admission;
use serde::Serialize;
use std::path::Path;
#[derive(Serialize)]
struct RepoOptions<'a> {
    repo: &'a str,
    credential_file: &'a str,
    output_directory: &'a str,
    timeout_seconds: u64,
    confirm: bool,
}
/// The exact helper argument text for `path`; non-Unicode paths are refused.
pub(super) fn path_text<'a>(path: &'a Path, label: &str) -> crate::command::DynResult<&'a str> {
    path.to_str()
        .ok_or_else(|| format!("generic {label} path must be Unicode").into())
}
pub(super) fn repo_hash(
    input: &Input,
    out: &Path,
    seconds: u64,
) -> crate::command::DynResult<String> {
    let credential = input
        .credential_file
        .as_deref()
        .ok_or("generic credential")?;
    Ok(admission::digest(&serde_json::to_vec(&RepoOptions {
        repo: &input.target_repo,
        credential_file: path_text(credential, "credential")?,
        output_directory: path_text(out, "output")?,
        timeout_seconds: seconds,
        confirm: true,
    })?))
}
