use super::contract::{Input, Mode, revision_valid};
use crate::automation::canary_receipts::{Digest, PackageVerification, verify_package};
use crate::command::DynResult;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
pub(super) struct Identity {
    pub(super) candidate: String,
    pub(super) base: String,
    pub(super) branch: String,
    pub(super) pass_id: String,
}

pub(super) fn previous(input: &Input) -> DynResult<Option<(PathBuf, Identity)>> {
    let Some(previous) = &input.previous else {
        return Ok(None);
    };
    let identity = verified(input, &previous.package, &previous.identity)?;
    revision_valid(&previous.candidate)?;
    if identity.candidate != previous.candidate || identity.base != input.selected_revision {
        return Err("previous candidate/base does not match frozen dependency outputs".into());
    }
    if !["repair-1", "repair-2", "repair-3"].contains(&identity.pass_id.as_str()) {
        return Err("independent verifier requires a repair producer package".into());
    }
    if identity.candidate == identity.base || !previous.package.join("candidate.bundle").is_file() {
        return Err("independent verifier requires a changed candidate bundle".into());
    }
    Ok(Some((previous.package.canonicalize()?, identity)))
}

pub(super) fn verified(input: &Input, package: &Path, expected: &str) -> DynResult<Identity> {
    let expected = Digest::try_from(expected.to_owned())?;
    verify_package(
        package,
        PackageVerification {
            expected_identity_sha256: expected.clone(),
            current_run_id: input.run_id.clone(),
            current_run_attempt: input.run_attempt.clone(),
            controller_revision: Some(input.controller_revision.clone()),
            selected_source: input.mesh_source.clone(),
        },
    )?;
    let bytes = fs::read(package.join("identity.json"))?;
    if Digest::of_bytes(&bytes) != expected {
        return Err("canary package changed after admission".into());
    }
    let identity: Identity = serde_json::from_slice(&bytes)?;
    if !identity.branch.starts_with("llama-canary/repair-")
        || identity.branch.ends_with(['.', '/'])
        || identity.branch.contains("..")
        || identity.branch.contains("//")
        || !identity
            .branch
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._/-".contains(&byte))
    {
        return Err("invalid identity-bound canary candidate branch".into());
    }
    Ok(identity)
}

pub(super) fn exported(input: &Input, previous: Option<&Identity>) -> DynResult<Identity> {
    let digest = Digest::of_file(&input.export.join("identity.json"))?;
    let identity = verified(input, &input.export, digest.as_str())?;
    let expected_candidate = match input.mode {
        Mode::Verify => {
            &previous
                .ok_or("missing verified previous candidate")?
                .candidate
        }
        Mode::Pinned => &input.selected_revision,
        Mode::Repair => &identity.candidate,
    };
    if identity.candidate != *expected_candidate
        || identity.base != input.selected_revision
        || identity.pass_id != input.pass_id
    {
        return Err("exported canary package differs from frozen build context".into());
    }
    Ok(identity)
}
