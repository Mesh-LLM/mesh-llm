use super::{archive, executable, process, source, workload};
use crate::{
    automation::canary_receipts::{
        Digest, PackageVerification, SourceFamilyPlan, placement, verify_package,
    },
    command::DynResult,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    controller_root: PathBuf,
    candidate_root: PathBuf,
    package: PathBuf,
    identity_sha256: Digest,
    controller_revision: String,
    selected_source: String,
    run_id: String,
    run_attempt: String,
    admitted_plan: PathBuf,
    admitted_identity_sha256: Digest,
    test_build: PathBuf,
}

#[derive(Deserialize)]
struct Identity {
    candidate: String,
    base: String,
    controller: String,
    manifest_sha256: Digest,
    plan_sha256: Digest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanIdentity {
    schema: u8,
    controller_revision: String,
    selected_revision: String,
    manifest_sha256: Digest,
    plan_sha256: Digest,
    cache_admission: String,
    gguf_admission: String,
}

pub(super) fn verify_document(bytes: &[u8]) -> DynResult<Vec<u8>> {
    let input: Input = serde_json::from_slice(bytes)?;
    for root in [
        &input.controller_root,
        &input.candidate_root,
        &input.package,
        &input.admitted_plan,
    ] {
        if !root.is_absolute() || !root.is_dir() {
            return Err("package admission paths must be existing absolute directories".into());
        }
    }
    source::revision(&input.controller_revision)?;
    if !input.test_build.is_absolute() || !input.test_build.is_file() {
        return Err("producer test-build stream must be an existing absolute file".into());
    }
    if process::text(&input.controller_root, &["rev-parse", "HEAD"])? != input.controller_revision {
        return Err("package controller checkout differs from frozen revision".into());
    }
    verify_package(
        &input.package,
        PackageVerification {
            expected_identity_sha256: input.identity_sha256.clone(),
            current_run_id: input.run_id.clone(),
            current_run_attempt: input.run_attempt.clone(),
            controller_revision: Some(input.controller_revision.clone()),
            selected_source: input.selected_source.clone(),
        },
    )?;
    let identity: Identity =
        serde_json::from_slice(&fs::read(input.package.join("identity.json"))?)?;
    if process::text(&input.candidate_root, &["rev-parse", "HEAD"])? != identity.candidate {
        return Err("package source checkout differs from exact candidate".into());
    }
    candidate_source(&input.candidate_root, &identity)?;
    admitted_plan(&input, &identity)?;
    let provenance: source::Provenance =
        serde_json::from_slice(&fs::read(input.package.join("llama-source.json"))?)?;
    source::validate_recipe(&input.candidate_root, &provenance)?;
    let prepared = source::prepared(&input.candidate_root)?;
    if provenance.head != prepared.head || provenance.markers != prepared.markers {
        return Err("packaged prepared source differs from actual clean native checkout".into());
    }
    source_bundle(&input.package.join("llama-source.bundle"), &provenance.head)?;
    verify_bundle_payload(&input.candidate_root, &input.package, &provenance)?;
    archive::binaries(&input.package.join("binaries.tar"))?;
    workload::archive(
        &input.package.join("workload-oracles.tar"),
        &identity.candidate,
        &provenance.head,
    )?;
    let test = executable::test_binary(&fs::read(&input.test_build)?)?;
    let expected = input
        .candidate_root
        .join("target/debug/deps")
        .canonicalize()?;
    let test = test.canonicalize()?;
    if test.parent() != Some(expected.as_path()) {
        return Err("multimodal test compiler artifact escapes exact candidate target".into());
    }
    let mut reader = fs::File::open(&test)?;
    let size = reader.metadata()?.len();
    executable::inspect(&mut reader, 0, size)?;
    // Compare immutable archives to the actual producer outputs, not only format.
    compare_binaries(&input.candidate_root, &input.package, &test)?;
    verify_package(
        &input.package,
        PackageVerification {
            expected_identity_sha256: input.identity_sha256.clone(),
            current_run_id: input.run_id.clone(),
            current_run_attempt: input.run_attempt.clone(),
            controller_revision: Some(input.controller_revision.clone()),
            selected_source: input.selected_source.clone(),
        },
    )?;
    admitted_plan(&input, &identity)?;
    candidate_source(&input.candidate_root, &identity)?;
    let final_prepared = source::prepared(&input.candidate_root)?;
    if final_prepared.head != provenance.head || final_prepared.markers != provenance.markers {
        return Err("prepared native source changed during package admission".into());
    }
    if process::text(&input.controller_root, &["rev-parse", "HEAD"])? != input.controller_revision
        || process::text(&input.candidate_root, &["rev-parse", "HEAD"])? != identity.candidate
    {
        return Err("controller or candidate changed during package admission".into());
    }
    Ok(serde_json::to_vec(
        &serde_json::json!({"schema":1,"status":"producer_package_closure_admitted", "identity_sha256": input.identity_sha256,
        "controller_revision":input.controller_revision,"candidate_revision":identity.candidate,"admitted_identity_sha256":input.admitted_identity_sha256}),
    )?)
}

fn candidate_source(root: &Path, identity: &Identity) -> DynResult<()> {
    if identity.candidate != identity.base {
        if process::text(root, &["rev-parse", &format!("{}^", identity.candidate)])?
            != identity.base
        {
            return Err("package candidate is not a direct child of frozen base".into());
        }
        let changed = process::text(
            root,
            &[
                "diff",
                "--name-only",
                &identity.base,
                &identity.candidate,
                "--",
                ".github",
                ".agents",
                "scripts",
                ".gitattributes",
                "ci/ci.md",
                "ci/llama-canary/agent-repair-prompt.md",
            ],
        )?;
        if !changed.is_empty() {
            return Err("candidate changed protected orchestration".into());
        }
    }
    process::text(root, &["diff-index", "--quiet", &identity.candidate, "--"])?;
    if !process::text(root, &["ls-files", "--others", "--exclude-standard"])?.is_empty() {
        return Err("candidate source has untracked members".into());
    }
    Ok(())
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) fn verify_bundle_payload(
    root: &Path,
    package: &Path,
    provenance: &source::Provenance,
) -> DynResult<()> {
    let mut random = [0; 16];
    getrandom::fill(&mut random)
        .map_err(|error| format!("package scratch randomness unavailable: {error}"))?;
    let directory =
        std::env::temp_dir().join(format!("mesh-canary-bundle-{}", hex::encode(random)));
    fs::create_dir(&directory)?;
    let scratch = Scratch(directory.canonicalize()?);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&scratch.0, fs::Permissions::from_mode(0o700))?;
    }
    let target = scratch.0.join("source");
    restore_native(root, package, provenance, &target)?;
    fs::remove_dir_all(&scratch.0)?;
    Ok(())
}

/// Restore the already admitted one-tree native bundle, without preparing/building.
pub(super) fn restore_native(
    root: &Path,
    package: &Path,
    provenance: &source::Provenance,
    target: &Path,
) -> DynResult<()> {
    source::validate_recipe(root, provenance)?;
    source_bundle(&package.join("llama-source.bundle"), &provenance.head)?;
    let parent = target.parent().ok_or("native restore has no parent")?;
    fs::create_dir_all(parent)?;
    process::git(
        parent,
        &[
            "clone".into(),
            "--no-checkout".into(),
            "--no-tags".into(),
            "--no-hardlinks".into(),
            package.join("llama-source.bundle").canonicalize()?.into(),
            target.to_path_buf().into(),
        ],
        None,
    )?;
    if process::text(target, &["rev-parse", "HEAD"])? != provenance.head {
        return Err("native bundle payload HEAD differs from provenance".into());
    }
    let objects = process::text(
        target,
        &[
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objecttype)",
        ],
    )?;
    if objects.lines().filter(|kind| *kind == "commit").count() != 1
        || objects
            .lines()
            .any(|kind| !["commit", "tree", "blob"].contains(&kind))
    {
        return Err(
            "native bundle carries history or unsupported objects beyond one source tree".into(),
        );
    }
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target.join(".git/shallow"))?
        .write_all(format!("{}\n", provenance.head).as_bytes())?;
    process::text(target, &["checkout", "--detach", &provenance.head])?;
    for (name, bytes) in &provenance.markers {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(target.join(name))?
            .write_all(bytes.as_bytes())?;
    }
    process::text(target, &["diff-index", "--quiet", "HEAD", "--"])?;
    Ok(())
}

fn admitted_plan(input: &Input, identity: &Identity) -> DynResult<()> {
    let (plan, _, manifest) = admitted_bytes(
        &input.admitted_plan,
        &input.admitted_identity_sha256,
        &input.candidate_root,
        &input.controller_revision,
        &identity.candidate,
    )?;
    if identity.controller != input.controller_revision
        || identity.manifest_sha256 != manifest
        || identity.plan_sha256 != Digest::of_bytes(&plan)
        || fs::read(input.package.join("plan.json"))? != plan
    {
        return Err("package differs from exact admitted candidate plan".into());
    }
    Ok(())
}

pub(super) fn admitted_bytes(
    directory: &Path,
    expected: &Digest,
    root: &Path,
    controller: &str,
    candidate: &str,
) -> DynResult<(Vec<u8>, Vec<u8>, Digest)> {
    let identity_bytes = fs::read(directory.join("source-plan.json"))?;
    if Digest::of_bytes(&identity_bytes) != *expected {
        return Err("admitted plan receipt differs from frozen dependency digest".into());
    }
    let admission: PlanIdentity = serde_json::from_slice(&identity_bytes)?;
    let plan = fs::read(directory.join("plan.json"))?;
    let manifest = Digest::of_file(&root.join("ci/llama-canary/family-certified.json"))?;
    if admission.schema != 1
        || admission.controller_revision != controller
        || admission.selected_revision != candidate
        || admission.manifest_sha256 != manifest
        || admission.plan_sha256 != Digest::of_bytes(&plan)
        || admission.cache_admission != "gguf_metadata"
        || admission.gguf_admission != "metadata_admitted"
    {
        return Err("package does not consume exact final admitted candidate plan and GGUF metadata receipt".into());
    }
    SourceFamilyPlan::parse(&plan)?;
    placement::project(&plan)?;
    Ok((plan, identity_bytes, manifest))
}

pub(super) fn compare_binaries(root: &Path, package: &Path, test: &Path) -> DynResult<()> {
    let mut file = fs::File::open(package.join("binaries.tar"))?;
    let members = archive::scan(&mut file)?;
    for name in archive::BINARIES {
        let path = if name == "skippy-mm-test" {
            test.to_owned()
        } else {
            root.join("target/debug").join(name)
        };
        if Digest::of_file(&path)? != members[name].sha256 {
            return Err("packaged executable differs from exact producer artifact".into());
        }
    }
    Ok(())
}

/// One independent native source tree and no prerequisites, matching legacy export.
pub(super) fn source_bundle(path: &Path, head: &str) -> DynResult<()> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line != "# v2 git bundle\n" {
        return Err("native source bundle must be SHA-1 v2 bundle".into());
    }
    let mut refs = BTreeMap::new();
    let mut consumed = line.len();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("truncated native source bundle header".into());
        }
        consumed += line.len();
        if consumed > 64 * 1024 {
            return Err("native source bundle header exceeds bound".into());
        }
        if line == "\n" {
            break;
        }
        let (revision, name) = line
            .trim_end_matches('\n')
            .split_once(' ')
            .ok_or("malformed native source bundle ref")?;
        source::revision(revision)?;
        if refs.insert(name.to_owned(), revision.to_owned()).is_some() {
            return Err("duplicate native source bundle ref".into());
        }
    }
    if refs.len() != 1 || refs.get("HEAD").is_none_or(|revision| revision != head) {
        return Err("native source bundle does not contain exact independent HEAD".into());
    }
    Ok(())
}
