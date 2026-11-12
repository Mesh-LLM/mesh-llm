//! Records the commit this crate is built from as `SKIPPY_GIT_COMMIT`, the
//! `git.commit.sha` telemetry resource attribute. Read from the repository's
//! own files rather than by running git, so a build without git on `PATH`
//! still knows its commit; a build from an exported tree, with no
//! repository, reports `unknown` unless `SKIPPY_GIT_COMMIT` says otherwise.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-env-changed=SKIPPY_GIT_COMMIT");
    let commit = env::var("SKIPPY_GIT_COMMIT")
        .ok()
        .filter(|commit| !commit.trim().is_empty())
        .or_else(|| head_commit(&PathBuf::from(env::var_os("CARGO_MANIFEST_DIR")?)))
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SKIPPY_GIT_COMMIT={commit}");
}

/// The commit `HEAD` names in the repository enclosing `manifest_dir`.
fn head_commit(manifest_dir: &Path) -> Option<String> {
    let dot_git = manifest_dir
        .ancestors()
        .map(|dir| dir.join(".git"))
        .find(|candidate| candidate.exists())?;
    let git_dir = git_dir(&dot_git)?;
    let head = read_watched(&git_dir.join("HEAD"))?;
    let Some(reference) = head.strip_prefix("ref: ") else {
        return Some(head);
    };
    // a worktree's refs live in the main repository's git directory
    let common_dir = read_watched(&git_dir.join("commondir"))
        .map(|common| git_dir.join(common))
        .unwrap_or(git_dir);
    read_watched(&common_dir.join(reference))
        .or_else(|| packed_ref(&common_dir.join("packed-refs"), reference))
}

/// `.git` is the git directory, or a file naming it for a worktree or submodule.
fn git_dir(dot_git: &Path) -> Option<PathBuf> {
    if dot_git.is_dir() {
        return Some(dot_git.to_path_buf());
    }
    let pointer = read_watched(dot_git)?;
    let target = pointer.strip_prefix("gitdir: ")?;
    Some(dot_git.parent()?.join(target))
}

fn packed_ref(packed_refs: &Path, reference: &str) -> Option<String> {
    read_watched(packed_refs)?
        .lines()
        .filter_map(|line| line.split_once(' '))
        .find(|(_, name)| *name == reference)
        .map(|(commit, _)| commit.to_string())
}

/// The trimmed contents of a file the build must watch; absent files are not
/// watched, since cargo reruns the script every build for a missing path.
fn read_watched(path: &Path) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    println!("cargo:rerun-if-changed={}", path.display());
    Some(contents.trim().to_string())
}
