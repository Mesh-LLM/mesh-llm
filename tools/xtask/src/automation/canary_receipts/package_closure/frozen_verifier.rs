//! The immutable verifier checkout and executable, independent of candidate source.
use super::{process, source};
use crate::{automation::canary_receipts::Digest, command::DynResult};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FrozenVerifier {
    pub(super) root: PathBuf,
    pub(super) revision: String,
    pub(super) executable_sha256: Digest,
}
impl FrozenVerifier {
    pub(super) fn validate(&self) -> DynResult<PathBuf> {
        source::revision(&self.revision)?;
        if !self.root.is_absolute() {
            return Err("verification controller root must be absolute".into());
        }
        let root = self.root.canonicalize()?;
        source_root(&root)?;
        if process::text(&root, &["rev-parse", "HEAD"])? != self.revision {
            return Err("verification controller moved from its frozen revision".into());
        }
        clean_source(&root)?;
        if executable_digest(&std::env::current_exe()?)? != self.executable_sha256 {
            return Err("running verifier differs from pre-admitted executable".into());
        }
        process::check()?;
        Ok(root)
    }
}
pub(super) fn source_root(root: &Path) -> DynResult<()> {
    if process::text(root, &["rev-parse", "--is-inside-work-tree"])? != "true"
        || !process::text(root, &["rev-parse", "--show-prefix"])?.is_empty()
    {
        return Err("verification root must be the Git source checkout root".into());
    }
    Ok(())
}
pub(super) fn clean_source(root: &Path) -> DynResult<()> {
    // Git deliberately avoids checking entries with these index promises.
    // Read-only verification must refuse them rather than modifying the index.
    let entries = process::git(
        root,
        &[
            "ls-files".into(),
            "--cached".into(),
            "-v".into(),
            "-z".into(),
        ],
        None,
    )?;
    for entry in entries
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
    {
        if entry.len() < 3 || entry[0] != b'H' || entry[1] != b' ' {
            return Err(
                "verification source has hidden, skipped or nonordinary tracked entries".into(),
            );
        }
    }
    process::text(
        root,
        &[
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--exit-code",
            "HEAD",
            "--",
        ],
    )?;
    process::text(
        root,
        &[
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--exit-code",
            "HEAD",
            "--",
        ],
    )?;
    if !process::text(root, &["ls-files", "--others", "--exclude-standard"])?.is_empty() {
        return Err("verification source has untracked files outside ignored build outputs".into());
    }
    Ok(())
}
pub(super) fn executable_digest(path: &Path) -> DynResult<Digest> {
    let before = fs::symlink_metadata(path)?;
    if !before.is_file() || before.len() > 512 * 1024 * 1024 {
        return Err("verification controller must be a bounded regular executable".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != before.len() {
        return Err("verification controller changed during admission".into());
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    let mut read = 0u64;
    loop {
        process::check()?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        read += count as u64;
        if read > before.len() {
            return Err("verification controller grew during admission".into());
        }
        digest.update(&buffer[..count]);
    }
    if read != before.len() || file.metadata()?.modified()? != metadata.modified()? {
        return Err("verification controller changed during hashing".into());
    }
    Ok(Digest::try_from(hex::encode(digest.finalize()))?)
}
