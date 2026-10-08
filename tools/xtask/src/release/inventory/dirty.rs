//! A binary-safe working-HEAD snapshot observed before report publication, never a build identity.
use super::{
    dirty_file,
    observation::Observation,
    provenance::{Error, Frozen, Git, Result},
    publication::Prepared,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct Status {
    pub(crate) index: String,
    pub(crate) worktree: String,
    pub(crate) path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) old_path: Option<String>,
}
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct Untracked {
    pub(crate) path: String,
    pub(crate) kind: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct Patch {
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub(crate) is_dirty: bool,
    pub(crate) diff_base: String,
    pub(crate) observation_scope: &'static str,
    pub(crate) evidence_algorithm: &'static str,
    pub(crate) platform: &'static str,
    pub(crate) evidence_sha256: String,
    pub(crate) porcelain: Vec<String>,
    pub(crate) status_entries: Vec<Status>,
    pub(crate) worktree_against_head: Patch,
    pub(crate) staged_against_head: Patch,
    pub(crate) unstaged: Patch,
    pub(crate) untracked_files: Vec<Untracked>,
}
fn read(git: &mut impl Git, args: &[&str], observation: &Observation) -> Result<Vec<u8>> {
    observation.check()?;
    let out = git.read(args)?;
    observation.check()?;
    if out.code != 0 {
        return Err(Error("Git dirty evidence command failed".into()));
    }
    Ok(out.stdout)
}
fn names(bytes: &[u8]) -> Result<Vec<&str>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if bytes.last() != Some(&0) {
        return Err(Error("Git dirty evidence missing NUL terminator".into()));
    }
    bytes[..bytes.len() - 1]
        .split(|b| *b == 0)
        .map(|s| {
            std::str::from_utf8(s).map_err(|_| Error("Git dirty pathname is not UTF-8".into()))
        })
        .collect()
}
fn statuses(bytes: &[u8], root: &Path, prepared: Option<&Prepared>) -> Result<Vec<Status>> {
    let names = names(bytes)?;
    let mut rows = Vec::new();
    let mut cursor = 0;
    while cursor < names.len() {
        let record = names[cursor];
        cursor += 1;
        let raw = record.as_bytes();
        if raw.len() < 4 || raw[2] != b' ' || !raw[..2].iter().all(|b| b" MADRCUT?!".contains(b)) {
            return Err(Error("Git porcelain status record invalid".into()));
        }
        let path = &record[3..];
        dirty_file::relative(path)?;
        let renamed = raw[..2].iter().any(|b| matches!(*b, b'R' | b'C'));
        let old_path = if renamed {
            let old = names
                .get(cursor)
                .ok_or_else(|| Error("Git porcelain rename incomplete".into()))?;
            cursor += 1;
            dirty_file::relative(old)?;
            Some((*old).into())
        } else {
            None
        };
        // Only freshly created, publisher-owned untracked scratch names are excluded.
        if &raw[..2] == b"??" && prepared.is_some_and(|p| p.owns_transient(&root.join(path))) {
            continue;
        }
        rows.push(Status {
            index: (raw[0] as char).to_string(),
            worktree: (raw[1] as char).to_string(),
            path: path.into(),
            old_path,
        });
    }
    Ok(rows)
}
fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn patch(hash: &mut Sha256, bytes: &[u8]) -> Patch {
    frame(hash, bytes);
    Patch {
        bytes: bytes.len() as u64,
        sha256: hex::encode(Sha256::digest(bytes)),
    }
}
fn read_patch(
    git: &mut impl Git,
    args: &[&str],
    observation: &Observation,
    hash: &mut Sha256,
) -> Result<Patch> {
    let bytes = read(git, args, observation)?;
    Ok(patch(hash, &bytes))
}
pub(crate) fn capture(
    git: &mut impl Git,
    root: &Path,
    source: &Frozen,
    observation: &Observation,
    prepared: Option<&Prepared>,
) -> Result<Snapshot> {
    observation.check()?;
    let status = read(
        git,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        observation,
    )?;
    let porcelain = statuses(&status, root, prepared)?;
    let mut hash = Sha256::new();
    frame(&mut hash, b"release-inventory-dirty-v1");
    frame(&mut hash, std::env::consts::OS.as_bytes());
    frame(&mut hash, source.working_tree_head.as_bytes());
    let status_json = serde_json::to_vec(&porcelain)
        .map_err(|_| Error("dirty status rendering failed".into()))?;
    frame(&mut hash, &status_json);
    let worktree_against_head = read_patch(
        git,
        &[
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            &source.working_tree_head,
            "--",
        ],
        observation,
        &mut hash,
    )?;
    let staged_against_head = read_patch(
        git,
        &[
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--cached",
            &source.working_tree_head,
            "--",
        ],
        observation,
        &mut hash,
    )?;
    let unstaged = read_patch(
        git,
        &["diff", "--no-ext-diff", "--no-textconv", "--binary", "--"],
        observation,
        &mut hash,
    )?;
    let untracked = read(
        git,
        &["ls-files", "--others", "--exclude-standard", "-z"],
        observation,
    )?;
    let mut untracked_files = Vec::new();
    for path in names(&untracked)? {
        observation.check()?;
        dirty_file::relative(path)?;
        if prepared.is_some_and(|p| p.owns_transient(&root.join(path))) {
            continue;
        }
        if untracked_files.len() >= 100000 {
            return Err(Error("untracked inventory exceeds100000 entries".into()));
        }
        let content = dirty_file::content(root, path, observation)?;
        frame(&mut hash, path.as_bytes());
        frame(&mut hash, content.kind.as_bytes());
        frame(&mut hash, &content.bytes.to_le_bytes());
        frame(&mut hash, content.sha256.as_bytes());
        untracked_files.push(Untracked {
            path: path.into(),
            kind: content.kind.into(),
            bytes: content.bytes,
            sha256: content.sha256,
        });
    }
    observation.check()?;
    let after = read(
        git,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        observation,
    )?;
    if statuses(&after, root, prepared)? != porcelain {
        return Err(Error(
            "working-tree status changed during dirty observation".into(),
        ));
    }
    Ok(Snapshot {
        is_dirty: !porcelain.is_empty(),
        diff_base: source.working_tree_head.clone(),
        observation_scope: "Working-tree HEAD snapshot observed before inventory publication; not candidate or build artifact identity",
        evidence_algorithm: "sha256-release-inventory-dirty-v1-length-framed",
        platform: std::env::consts::OS,
        evidence_sha256: hex::encode(hash.finalize()),
        porcelain: porcelain
            .iter()
            .map(|s| match &s.old_path {
                Some(old) => format!("{}{} {old} -> {}", s.index, s.worktree, s.path),
                None => format!("{}{} {}", s.index, s.worktree, s.path),
            })
            .collect(),
        status_entries: porcelain,
        worktree_against_head,
        staged_against_head,
        unstaged,
        untracked_files,
    })
}
pub(crate) fn revalidate(
    snapshot: &Snapshot,
    git: &mut impl Git,
    root: &Path,
    source: &Frozen,
    observation: &Observation,
    prepared: Option<&Prepared>,
) -> Result<()> {
    let current = capture(git, root, source, observation, prepared)?;
    if &current != snapshot {
        return Err(Error(
            "working-tree dirty snapshot changed before publication".into(),
        ));
    }
    Ok(())
}
