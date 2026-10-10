use super::{
    admission, archive,
    archive_write::{self, Content},
    candidate_view, executable, process,
    producer_receipt::{self, Context},
    source, workload,
};
use crate::{
    automation::canary_receipts::{Digest, PackageVerification, placement, verify_package},
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
    pub(super) output: PathBuf,
    pub(super) candidate: String,
    pub(super) base: String,
    pub(super) branch: String,
    pub(super) pass_id: String,
    pub(super) mode: String,
    pub(super) test_build: PathBuf,
    pub(super) bundle: Option<PathBuf>,
    pub(super) summary: PathBuf,
    pub(super) workload_oracles: PathBuf,
    pub(super) admitted_plan: PathBuf,
    pub(super) admitted_identity_sha256: Digest,
    pub(super) producer_receipt: PathBuf,
    pub(super) producer_receipt_sha256: Digest,
}

pub(super) fn pack(input: &Input) -> DynResult<serde_json::Value> {
    input.context.validate()?;
    source::branch(&input.branch)?;
    validate_mode(input)?;
    if !input.root.is_absolute() || !input.workload_oracles.is_absolute() {
        return Err("producer root and closure must be absolute".into());
    }
    let root = input.root.canonicalize()?;
    candidate_view::producer(&root, &input.base, &input.candidate)?;
    let destination = producer_receipt::new_output(
        &input.output,
        &[&root, &input.context.controller_root.canonicalize()?],
    )?;
    let parent = destination.parent().ok_or("package output has no parent")?;
    let stage = candidate_view::Owned::new(parent, "package")?;
    let (plan, receipt, manifest) = admission::admitted_bytes(
        &input.admitted_plan,
        &input.admitted_identity_sha256,
        &root,
        &input.context.controller_revision,
        &input.candidate,
    )?;
    producer_receipt::create_file(&stage.path.join("plan.json"), &plan)?;
    producer_receipt::create_file(&stage.path.join("source-plan.json"), &receipt)?;
    let sealed = producer_receipt::consume(
        &input.producer_receipt,
        &input.producer_receipt_sha256,
        &input.context,
        &root,
        &input.workload_oracles,
        &input.candidate,
    )?;
    let native = source::prepared(&root)?;
    export_native(&root, &stage.path, &native)?;
    binaries(&root, &stage.path, &input.test_build)?;
    workloads(&input.workload_oracles, &stage.path, &sealed)?;
    archive::binaries(&stage.path.join("binaries.tar"))?;
    workload::archive(
        &stage.path.join("workload-oracles.tar"),
        &input.candidate,
        &native.head,
    )?;
    let summary = copy_bound(&input.summary, &stage.path.join("upstream-summary.md"))?;
    let bundle = if let Some(bundle) = &input.bundle {
        candidate_bundle(
            &root,
            bundle,
            &input.base,
            &input.candidate,
            &input.branch,
            parent,
        )?;
        Some(copy_bound(bundle, &stage.path.join("candidate.bundle"))?)
    } else {
        None
    };
    let identity = serde_json::json!({"schema":3,"platform":"macos-arm64-metal","candidate":input.candidate,"base":input.base,"branch":input.branch,"pass_id":input.pass_id,
        "run_id":input.context.run_id,"run_attempt":input.context.run_attempt,"controller":input.context.controller_revision,"mesh_source":input.context.selected_source,
        "manifest_sha256":manifest,"plan_sha256":Digest::of_bytes(&plan),"source_plan_identity_sha256":input.admitted_identity_sha256,
        "binaries_sha256":Digest::of_file(&stage.path.join("binaries.tar"))?,"workload_oracles_sha256":Digest::of_file(&stage.path.join("workload-oracles.tar"))?,
        "llama_bundle_sha256":Digest::of_file(&stage.path.join("llama-source.bundle"))?,"llama_provenance_sha256":Digest::of_file(&stage.path.join("llama-source.json"))?,"summary_sha256":summary,"bundle_sha256":bundle});
    let bytes = serde_json::to_vec_pretty(&identity)?;
    let digest = Digest::of_bytes(&bytes);
    producer_receipt::create_file(&stage.path.join("identity.json"), &bytes)?;
    verify_package(
        &stage.path,
        PackageVerification {
            expected_identity_sha256: digest.clone(),
            current_run_id: input.context.run_id.clone(),
            current_run_attempt: input.context.run_attempt.clone(),
            controller_revision: Some(input.context.controller_revision.clone()),
            selected_source: input.context.selected_source.clone(),
        },
    )?;
    producer_receipt::consume(
        &input.producer_receipt,
        &input.producer_receipt_sha256,
        &input.context,
        &root,
        &input.workload_oracles,
        &input.candidate,
    )?;
    let (_, final_receipt, final_manifest) = admission::admitted_bytes(
        &input.admitted_plan,
        &input.admitted_identity_sha256,
        &root,
        &input.context.controller_revision,
        &input.candidate,
    )?;
    if final_receipt != receipt || final_manifest != manifest {
        return Err("admitted candidate changed while packing".into());
    }
    input.context.validate()?;
    process::check()?;
    let matrix = serde_json::to_value(placement::project(&plan)?)?;
    stage.publish(&destination)?;
    Ok(
        serde_json::json!({"matrix":matrix,"identity_sha256":digest,"candidate":input.candidate,"branch":input.branch,"admitted_identity_sha256":input.admitted_identity_sha256}),
    )
}

fn validate_mode(input: &Input) -> DynResult<()> {
    let repair = ["repair-1", "repair-2", "repair-3"].contains(&input.pass_id.as_str());
    let verify = ["verify-1", "verify-2", "verify-3"].contains(&input.pass_id.as_str());
    match input.mode.as_str() {
        "repair-build" if repair && input.candidate != input.base && input.bundle.is_some() => {}
        "verify-build" if verify && input.candidate != input.base && input.bundle.is_some() => {}
        "pinned-build"
            if input.pass_id == "repair-1"
                && input.candidate == input.base
                && input.bundle.is_none() => {}
        _ => return Err("package mode/pass/candidate/bundle contract mismatch".into()),
    }
    if input.context.selected_source.is_empty() {
        if input.base != input.context.controller_revision {
            return Err("ordinary package base differs from frozen controller".into());
        }
    } else if input.mode != "pinned-build"
        || input.candidate != input.context.selected_source
        || input.base != input.candidate
    {
        return Err("selected package changed frozen source".into());
    }
    Ok(())
}

fn binaries(root: &Path, stage: &Path, jsonl: &Path) -> DynResult<()> {
    if !jsonl.is_absolute() {
        return Err("test-build stream must be absolute".into());
    }
    let test = executable::test_binary(&fs::read(jsonl)?)?.canonicalize()?;
    if test.parent() != Some(root.join("target/debug/deps").canonicalize()?.as_path()) {
        return Err("test compiler artifact escaped exact producer target".into());
    }
    let mut members = Vec::new();
    for name in archive::BINARIES {
        let path = if name == "skippy-mm-test" {
            test.clone()
        } else {
            root.join("target/debug").join(name)
        };
        let mut file = fs::File::open(&path)?;
        let size = file.metadata()?.len();
        executable::inspect(&mut file, 0, size)?;
        members.push(archive_write::Input {
            name: name.to_owned(),
            digest: Digest::of_file(&path)?,
            content: Content::File(path),
            executable: true,
        });
    }
    archive_write::write(&stage.join("binaries.tar"), members)
}

fn workloads(root: &Path, stage: &Path, sealed: &[u8]) -> DynResult<()> {
    let root = root.canonicalize()?;
    let mut members = Vec::new();
    for (name, digest) in workload::members(sealed)? {
        let path = root.join(&name).canonicalize()?;
        if !path.starts_with(&root) {
            return Err("workload member escaped producer closure".into());
        }
        members.push(archive_write::Input {
            executable: name != "native/.mesh-llm-build-stamp",
            name,
            content: Content::File(path),
            digest,
        });
    }
    members.push(archive_write::Input {
        name: "producer.json".into(),
        content: Content::Bytes(sealed.to_vec()),
        digest: Digest::of_bytes(sealed),
        executable: false,
    });
    archive_write::write(&stage.join("workload-oracles.tar"), members)
}

pub(super) fn copy_bound(source: &Path, destination: &Path) -> DynResult<Digest> {
    if !source.is_absolute() || !fs::symlink_metadata(source)?.is_file() {
        return Err("package source must be an absolute regular file".into());
    }
    let expected = Digest::of_file(source)?;
    fs::copy(source, destination)?;
    if Digest::of_file(destination)? != expected || Digest::of_file(source)? != expected {
        return Err("package source changed while copying".into());
    }
    Ok(expected)
}

fn export_native(root: &Path, stage: &Path, provenance: &source::Provenance) -> DynResult<()> {
    let scratch = candidate_view::Owned::new(
        stage.parent().ok_or("package stage has no parent")?,
        "native-export",
    )?;
    let target = scratch.path.join("source");
    let native = root.join(".deps/llama.cpp").canonicalize()?;
    let uri = format!("file://{}", native.to_str().ok_or("non-UTF8 native path")?);
    process::git(
        &scratch.path,
        &[
            "clone".into(),
            "--quiet".into(),
            "--no-checkout".into(),
            "--no-tags".into(),
            "--depth=1".into(),
            uri.into(),
            target.clone().into(),
        ],
        None,
    )?;
    if process::text(&target, &["rev-parse", "HEAD"])? != provenance.head {
        return Err("prepared native HEAD changed during export clone".into());
    }
    process::git(
        &target,
        &[
            "bundle".into(),
            "create".into(),
            stage.join("llama-source.bundle").into(),
            "HEAD".into(),
        ],
        None,
    )?;
    let after = source::prepared(root)?;
    if after.head != provenance.head || after.markers != provenance.markers {
        return Err("prepared source changed during export".into());
    }
    producer_receipt::create_file(
        &stage.join("llama-source.json"),
        &serde_json::to_vec_pretty(provenance)?,
    )?;
    admission::source_bundle(&stage.join("llama-source.bundle"), &provenance.head)?;
    admission::verify_bundle_payload(root, stage, provenance)?;
    scratch.cleanup()
}

pub(super) fn candidate_bundle(
    root: &Path,
    bundle: &Path,
    base: &str,
    candidate: &str,
    branch: &str,
    parent: &Path,
) -> DynResult<()> {
    source::branch(branch)?;
    let view = candidate_view::materialize(root, base, parent)?;
    let target = view.path.join("source");
    let heads = process::git(
        &target,
        &[
            "bundle".into(),
            "unbundle".into(),
            bundle.canonicalize()?.into(),
        ],
        None,
    )?;
    source::candidate_bundle_identity(&heads, candidate, branch)?;
    candidate_view::policy(&target, base, candidate)?;
    view.cleanup()
}
