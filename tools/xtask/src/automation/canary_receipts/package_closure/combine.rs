//! Bind two platform-native producer packages to one candidate receipt identity.
use super::{admission, archive, producer_receipt::Context};
use crate::{
    automation::canary_receipts::{Digest, PackageVerification, placement, verify_package},
    command::DynResult,
};
use serde::Deserialize;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub(super) context: Context,
    pub(super) mac_package: PathBuf,
    pub(super) mac_identity_sha256: Digest,
    pub(super) linux_package: PathBuf,
    pub(super) linux_identity_sha256: Digest,
    pub(super) output: PathBuf,
}

pub(super) fn run(input: &Input) -> DynResult<Value> {
    input.context.validate()?;
    let mac = input.mac_package.canonicalize()?;
    let linux = input.linux_package.canonicalize()?;
    if !mac.is_dir() || !linux.is_dir() || !input.output.is_absolute() {
        return Err("combined package inputs must be absolute directories".into());
    }
    for (package, digest) in [
        (&mac, &input.mac_identity_sha256),
        (&linux, &input.linux_identity_sha256),
    ] {
        verify_package(
            package,
            PackageVerification {
                expected_identity_sha256: digest.clone(),
                current_run_id: input.context.run_id.clone(),
                current_run_attempt: input.context.run_attempt.clone(),
                controller_revision: Some(input.context.controller_revision.clone()),
                selected_source: input.context.selected_source.clone(),
            },
        )?;
    }
    let mut mac_identity: Value = serde_json::from_slice(&fs::read(mac.join("identity.json"))?)?;
    let linux_identity: Value = serde_json::from_slice(&fs::read(linux.join("identity.json"))?)?;
    if mac_identity["platform"] != "macos-arm64-metal"
        || linux_identity["platform"] != "linux-arm64-cuda"
    {
        return Err("native producer platform identity mismatch".into());
    }
    archive::binaries_for(&mac.join("binaries.tar"), false)?;
    archive::binaries_for(&linux.join("binaries.tar"), true)?;
    // Each verified producer may come from an earlier attempt of this run.
    // Retain the macOS attempt in the combined identity; all source fields match.
    for key in [
        "candidate",
        "base",
        "branch",
        "pass_id",
        "run_id",
        "controller",
        "mesh_source",
        "manifest_sha256",
        "plan_sha256",
        "source_plan_identity_sha256",
        "llama_provenance_sha256",
        "summary_sha256",
        "bundle_sha256",
    ] {
        if mac_identity[key] != linux_identity[key] {
            return Err(format!("native producers disagree on {key}").into());
        }
    }
    for name in [
        "plan.json",
        "source-plan.json",
        "llama-source.json",
        "upstream-summary.md",
        "candidate.bundle",
    ] {
        let mac_file = mac.join(name);
        let linux_file = linux.join(name);
        if mac_file.exists() != linux_file.exists()
            || mac_file.exists() && Digest::of_file(&mac_file)? != Digest::of_file(&linux_file)?
        {
            return Err(format!("native producers disagree on {name}").into());
        }
    }
    let provenance: Value = serde_json::from_slice(&fs::read(mac.join("llama-source.json"))?)?;
    let head = provenance["head"]
        .as_str()
        .ok_or("missing native source head")?;
    admission::source_bundle(&mac.join("llama-source.bundle"), head)?;
    admission::source_bundle(&linux.join("llama-source.bundle"), head)?;
    let parent = input
        .output
        .parent()
        .ok_or("combined package has no parent")?
        .canonicalize()?;
    if input.output.exists() || input.output.starts_with(&mac) || input.output.starts_with(&linux) {
        return Err("combined package destination overlaps a producer".into());
    }
    let stage = tempfile::Builder::new()
        .prefix("llama-combined-")
        .tempdir_in(&parent)?;
    for name in [
        "plan.json",
        "source-plan.json",
        "llama-source.bundle",
        "llama-source.json",
        "upstream-summary.md",
        "candidate.bundle",
        "binaries.tar",
        "workload-oracles.tar",
    ] {
        copy_if_present(&mac, stage.path(), name, name)?;
    }
    copy_if_present(&linux, stage.path(), "binaries.tar", "linux-binaries.tar")?;
    copy_if_present(
        &linux,
        stage.path(),
        "workload-oracles.tar",
        "linux-workload-oracles.tar",
    )?;
    mac_identity["schema"] = Value::from(4);
    mac_identity["platform"] = Value::from("macos-arm64-metal+linux-arm64-cuda");
    mac_identity["linux_binaries_sha256"] =
        Value::from(Digest::of_file(&stage.path().join("linux-binaries.tar"))?.as_str());
    mac_identity["linux_workload_oracles_sha256"] =
        Value::from(Digest::of_file(&stage.path().join("linux-workload-oracles.tar"))?.as_str());
    let bytes = serde_json::to_vec_pretty(&mac_identity)?;
    let identity_sha256 = Digest::of_bytes(&bytes);
    fs::write(stage.path().join("identity.json"), bytes)?;
    verify_package(
        stage.path(),
        PackageVerification {
            expected_identity_sha256: identity_sha256.clone(),
            current_run_id: input.context.run_id.clone(),
            current_run_attempt: input.context.run_attempt.clone(),
            controller_revision: Some(input.context.controller_revision.clone()),
            selected_source: input.context.selected_source.clone(),
        },
    )?;
    let matrix = serde_json::to_value(placement::project(&fs::read(
        stage.path().join("plan.json"),
    )?)?)?;
    let published = stage.keep();
    fs::rename(published, &input.output)?;
    Ok(
        serde_json::json!({"matrix":matrix,"identity_sha256":identity_sha256,"candidate":mac_identity["candidate"],"branch":mac_identity["branch"]}),
    )
}

fn copy_if_present(from: &Path, to: &Path, source: &str, destination: &str) -> DynResult<()> {
    let path = from.join(source);
    if path.exists() {
        if !fs::symlink_metadata(&path)?.is_file() {
            return Err("native package contains a nonregular member".into());
        }
        fs::copy(path, to.join(destination))?;
    }
    Ok(())
}
