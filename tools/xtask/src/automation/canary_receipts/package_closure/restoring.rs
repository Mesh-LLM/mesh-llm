//! Verify-only restore into a fresh dedicated consumer role, then publish atomically.
use super::{
    admission, archive, archive_extract, candidate_view, process, producer_receipt::Context,
    restore_transaction::Publication, source, workload,
};
use crate::{
    automation::canary_receipts::{Digest, PackageVerification, verify_package},
    command::DynResult,
};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub(super) context: Context,
    pub(super) root: PathBuf,
    pub(super) package: PathBuf,
    pub(super) identity_sha256: Digest,
}
#[derive(Deserialize)]
struct Identity {
    candidate: String,
    base: String,
    branch: String,
    manifest_sha256: Digest,
    plan_sha256: Digest,
    source_plan_identity_sha256: Digest,
    bundle_sha256: Option<Digest>,
}

pub(super) fn restore(input: &Input) -> DynResult<serde_json::Value> {
    input.context.validate()?;
    if !input.root.is_absolute() || !input.package.is_absolute() {
        return Err("restore paths must be absolute".into());
    }
    let root = input.root.canonicalize()?;
    let controller = input.context.controller_root.canonicalize()?;
    let package = input.package.canonicalize()?;
    if root != controller.join("canary-source") || package.starts_with(&root) {
        return Err(
            "restore requires the dedicated controller/canary-source role and external package"
                .into(),
        );
    }
    verify(input, &package)?;
    let identity: Identity = serde_json::from_slice(&fs::read(package.join("identity.json"))?)?;
    source::revision(&identity.base)?;
    source::revision(&identity.candidate)?;
    source::branch(&identity.branch)?;
    if input.context.selected_source.is_empty()
        && identity.base != input.context.controller_revision
    {
        return Err("ordinary restore base differs from frozen controller".into());
    }
    fresh(&root, &identity.base)?;
    let stage = candidate_view::materialize(
        &root,
        &identity.base,
        root.parent().ok_or("restore root has no parent")?,
    )?;
    let staged = stage.path.join("source");
    if identity.candidate != identity.base {
        if identity.bundle_sha256.is_none() {
            return Err("changed candidate has no admitted source bundle".into());
        }
        let heads = process::git(
            &staged,
            &[
                "bundle".into(),
                "unbundle".into(),
                package.join("candidate.bundle").into(),
            ],
            None,
        )?;
        source::candidate_bundle_identity(&heads, &identity.candidate, &identity.branch)?;
        candidate_view::policy(&staged, &identity.base, &identity.candidate)?;
        process::text(
            &staged,
            &["checkout", "--quiet", "--detach", &identity.candidate],
        )?;
    } else if identity.bundle_sha256.is_some() {
        return Err("unchanged source unexpectedly carries candidate bundle".into());
    }
    admitted(&package, &staged, input, &identity)?;
    // The staged clone must not carry source-controlled aliases for generated roots.
    for name in [".deps", "target"] {
        if fs::symlink_metadata(staged.join(name)).is_ok() {
            return Err("source snapshot owns a generated restore root".into());
        }
    }
    let provenance: source::Provenance =
        serde_json::from_slice(&fs::read(package.join("llama-source.json"))?)?;
    admission::restore_native(
        &staged,
        &package,
        &provenance,
        &staged.join(".deps/llama.cpp"),
    )?;
    archive::binaries(&package.join("binaries.tar"))?;
    workload::archive(
        &package.join("workload-oracles.tar"),
        &identity.candidate,
        &provenance.head,
    )?;
    fs::create_dir(staged.join("target"))?;
    archive_extract::extract(&package.join("binaries.tar"), &staged.join("target/debug"))?;
    archive_extract::extract(
        &package.join("workload-oracles.tar"),
        &staged.join(".deps/canary-workload-oracles"),
    )?;
    archive_extract::normalize_workload(&staged.join(".deps/canary-workload-oracles"))?;
    closure(
        &staged,
        &stage.path.join("staged.diff"),
        &provenance,
        &identity,
    )?;
    admission::compare_binaries(
        &staged,
        &package,
        &staged.join("target/debug/skippy-mm-test"),
    )?;
    verify(input, &package)?;
    admitted(&package, &staged, input, &identity)?;
    input.context.validate()?;
    fresh(&root, &identity.base)?;
    let mut transaction = Publication::new(&root, stage)?;
    transaction.publish()?;
    let result = (|| {
        closure(
            &root,
            &transaction.stage.path.join("published.diff"),
            &provenance,
            &identity,
        )?;
        admission::compare_binaries(&root, &package, &root.join("target/debug/skippy-mm-test"))?;
        verify(input, &package)?;
        admitted(&package, &root, input, &identity)?;
        input.context.validate()?;
        process::check()
    })();
    if let Err(error) = result {
        return match transaction.rollback() {
            Ok(()) => Err(error),
            Err(rollback) => Err(format!("restore rejected: {error}; {rollback}").into()),
        };
    }
    transaction.commit()?;
    Ok(
        serde_json::json!({"candidate":identity.candidate,"identity_sha256":input.identity_sha256,"status":"restored_verified_package"}),
    )
}

fn verify(input: &Input, package: &Path) -> DynResult<()> {
    verify_package(
        package,
        PackageVerification {
            expected_identity_sha256: input.identity_sha256.clone(),
            current_run_id: input.context.run_id.clone(),
            current_run_attempt: input.context.run_attempt.clone(),
            controller_revision: Some(input.context.controller_revision.clone()),
            selected_source: input.context.selected_source.clone(),
        },
    )?;
    Ok(())
}
fn fresh(root: &Path, base: &str) -> DynResult<()> {
    if fs::symlink_metadata(root)?.file_type().is_symlink()
        || process::text(root, &["rev-parse", "HEAD"])? != base
    {
        return Err("consumer root is not fresh frozen base checkout".into());
    }
    process::text(root, &["diff-index", "--quiet", "HEAD", "--"])?;
    if !process::text(root, &["ls-files", "--others", "--exclude-standard"])?.is_empty()
        || !process::text(
            root,
            &["ls-files", "--others", "--ignored", "--exclude-standard"],
        )?
        .is_empty()
    {
        return Err(
            "fresh consumer has untracked or ignored data; restore cannot replace it".into(),
        );
    }
    Ok(())
}
fn admitted(package: &Path, root: &Path, input: &Input, identity: &Identity) -> DynResult<()> {
    let (plan, _, manifest) = admission::admitted_bytes(
        package,
        &identity.source_plan_identity_sha256,
        root,
        &input.context.controller_revision,
        &identity.candidate,
    )?;
    if manifest != identity.manifest_sha256 || Digest::of_bytes(&plan) != identity.plan_sha256 {
        return Err("restored package differs from admitted plan/manifest".into());
    }
    Ok(())
}
fn closure(
    root: &Path,
    capture: &Path,
    provenance: &source::Provenance,
    identity: &Identity,
) -> DynResult<()> {
    if process::text(root, &["rev-parse", "HEAD"])? != identity.candidate {
        return Err("restored source differs from exact candidate".into());
    }
    candidate_view::policy(root, &identity.base, &identity.candidate)?;
    let native = source::prepared(root)?;
    if native.head != provenance.head || native.markers != provenance.markers {
        return Err("restored native differs from admitted provenance".into());
    }
    workload::verify_producer(
        root,
        &root.join(".deps/canary-workload-oracles"),
        capture,
        &native.head,
    )?;
    Ok(())
}

/// Certify only the complete previously restored package; never regenerate it.
pub(super) fn verify_consumer(input: &Input, capture: &Path) -> DynResult<()> {
    input.context.validate()?;
    let root = input.root.canonicalize()?;
    let package = input.package.canonicalize()?;
    if root
        != input
            .context
            .controller_root
            .canonicalize()?
            .join("canary-source")
        || package.starts_with(&root)
    {
        return Err("certify requires dedicated restored consumer and external package".into());
    }
    verify(input, &package)?;
    let identity: Identity = serde_json::from_slice(&fs::read(package.join("identity.json"))?)?;
    let provenance: source::Provenance =
        serde_json::from_slice(&fs::read(package.join("llama-source.json"))?)?;
    admitted(&package, &root, input, &identity)?;
    closure(&root, capture, &provenance, &identity)?;
    archive::binaries(&package.join("binaries.tar"))?;
    workload::archive(
        &package.join("workload-oracles.tar"),
        &identity.candidate,
        &provenance.head,
    )?;
    admission::compare_binaries(&root, &package, &root.join("target/debug/skippy-mm-test"))?;
    verify(input, &package)?;
    input.context.validate()?;
    process::check()
}
