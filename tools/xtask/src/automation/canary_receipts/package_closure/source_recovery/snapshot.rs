use super::super::{process, source};
use super::{
    archive::{self, Root},
    git::Git,
};
use crate::command::DynResult;
use serde::Serialize;
use std::{
    ffi::OsStr,
    fs,
    io::{Read, Write},
    os::unix::ffi::OsStrExt,
    path::Path,
};

const MAX_PATCH: u64 = 256 * 1024 * 1024;
#[derive(Serialize)]
struct Snapshot {
    base: String,
    head: String,
    tracked_patch_bytes: u64,
    untracked_files: usize,
}
#[derive(Serialize)]
struct Manifest {
    #[serde(flatten)]
    source: Snapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    prepared_llama_cpp: Option<Snapshot>,
    verified: bool,
}

pub(super) fn save(root: &Path, output: &Path, base: &str) -> DynResult<()> {
    source::revision(base)?;
    let source_root = Root::open(root)?;
    let root = root.canonicalize()?;
    source_root.unchanged(&root)?;
    let parent_admission = Root::open(output.parent().ok_or("recovery output lacks parent")?)?;
    let parent = output
        .parent()
        .ok_or("recovery output lacks parent")?
        .canonicalize()?;
    let output = parent.join(output.file_name().ok_or("recovery output lacks name")?);
    if output == root || root.starts_with(&output) || output.symlink_metadata().is_ok() {
        return Err("recovery output must be fresh and cannot contain the source root".into());
    }
    let output_parent = Root::open(&parent)?;
    parent_admission.unchanged(&parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".canary-recovery-")
        .tempdir_in(&parent)?;
    let stage_root = Root::open(stage.path())?;
    let git = Git::new();
    let source = capture(
        &root,
        &source_root,
        stage.path(),
        &output,
        stage.path(),
        base,
        &git,
    )?;
    let nested = if let Some(nested_root) = source_root.nested(Path::new(".deps/llama.cpp"))? {
        let nested = root.join(".deps/llama.cpp");
        if nested.join(".git").symlink_metadata().is_ok() {
            nested_root.unchanged(&nested)?;
            let nested_base = git.text(&nested, &["rev-parse", "HEAD"])?;
            source::revision(&nested_base)?;
            let destination = stage.path().join("prepared-llama-cpp");
            fs::create_dir(&destination)?;
            Some(capture(
                &nested,
                &nested_root,
                &destination,
                &output,
                stage.path(),
                &nested_base,
                &git,
            )?)
        } else {
            None
        }
    } else {
        None
    };
    let manifest = Manifest {
        source,
        prepared_llama_cpp: nested,
        verified: false,
    };
    stage_root.unchanged(stage.path())?;
    let mut file = stage_root.create_file(Path::new("manifest.json"))?;
    serde_json::to_writer_pretty(&mut file, &manifest)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    git.remaining()?;
    source_root.unchanged(&root)?;
    output_parent.unchanged(&parent)?;
    stage_root.unchanged(stage.path())?;
    if output.symlink_metadata().is_ok() {
        return Err("recovery destination appeared before publication".into());
    }
    output_parent.publish(
        stage
            .path()
            .file_name()
            .ok_or("recovery stage has no name")?,
        output.file_name().ok_or("recovery output has no name")?,
    )?;
    stage_root.unchanged(&output)?;
    output_parent.unchanged(&parent)?;
    process::check()?;
    Ok(())
}

fn capture(
    root: &Path,
    admitted: &Root,
    output: &Path,
    destination: &Path,
    stage: &Path,
    base: &str,
    git: &Git,
) -> DynResult<Snapshot> {
    admitted.unchanged(root)?;
    git.text(root, &["cat-file", "-e", &format!("{base}^{{commit}}")])?;
    let head = git.text(root, &["rev-parse", "HEAD"])?;
    source::revision(&head)?;
    let listing = git.run(
        root,
        &[
            "ls-files".into(),
            "-z".into(),
            "--others".into(),
            "--exclude-standard".into(),
        ],
    )?;
    let names = listing
        .split(|byte| *byte == 0)
        .filter(|name| !name.is_empty())
        .filter(|name| {
            let path = root.join(OsStr::from_bytes(name));
            !path.starts_with(destination) && !path.starts_with(stage)
        })
        .collect::<Vec<_>>();
    let patch = output.join("tracked.patch");
    let mut output_arg = std::ffi::OsString::from("--output=");
    output_arg.push(&patch);
    git.run(
        root,
        &[
            "diff".into(),
            "--binary".into(),
            "--no-ext-diff".into(),
            "--no-textconv".into(),
            output_arg,
            base.into(),
            "--".into(),
        ],
    )?;
    let patch_root = Root::open(output)?;
    let mut patch_file = patch_root.file(Path::new("tracked.patch"))?;
    let bytes = patch_file.metadata()?.len();
    if bytes > MAX_PATCH {
        return Err("recovery tracked patch exceeds 256 MiB".into());
    }
    // Git owns exact diff bytes; diagnostic line capture never persists this payload.
    let mut total = 0;
    let mut buffer = [0; 65536];
    loop {
        git.remaining()?;
        let count = patch_file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > MAX_PATCH {
            return Err("recovery patch grew beyond its bound".into());
        }
    }
    if total != bytes {
        return Err("recovery patch changed while reading".into());
    }
    let untracked_files = archive::write(
        admitted,
        &names,
        patch_root.create_file(Path::new("untracked.tar.gz"))?,
        git,
    )?;
    admitted.unchanged(root)?;
    if git.text(root, &["rev-parse", "HEAD"])? != head {
        return Err("recovery source HEAD changed during capture".into());
    }
    Ok(Snapshot {
        base: base.into(),
        head,
        tracked_patch_bytes: bytes,
        untracked_files,
    })
}
