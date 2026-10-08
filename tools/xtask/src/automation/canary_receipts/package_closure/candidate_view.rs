use super::{process, source};
use crate::command::DynResult;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub(super) struct Owned {
    pub(super) path: PathBuf,
    remove: bool,
}
impl Owned {
    pub(super) fn retain(&mut self) {
        self.remove = false;
    }
    pub(super) fn cleanup(mut self) -> DynResult<()> {
        match fs::remove_dir_all(&self.path) {
            Ok(()) => {
                self.remove = false;
                Ok(())
            }
            Err(error) => {
                self.remove = false;
                Err(format!(
                    "owned canary stage cleanup failed at {}: {error}",
                    self.path.display()
                )
                .into())
            }
        }
    }
    pub(super) fn new(parent: &Path, label: &str) -> DynResult<Self> {
        let parent = parent.canonicalize()?;
        let mut random = [0; 16];
        getrandom::fill(&mut random)
            .map_err(|error| format!("canary scratch randomness failed: {error}"))?;
        let path = parent.join(format!(".canary-{label}-{}", hex::encode(random)));
        fs::create_dir(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self { path, remove: true })
    }
    pub(super) fn publish(mut self, destination: &Path) -> DynResult<()> {
        process::check()?;
        fs::create_dir(destination)?;
        match fs::rename(&self.path, destination) {
            Ok(()) => {
                self.remove = false;
                Ok(())
            }
            Err(error) => {
                let _ = fs::remove_dir(destination);
                Err(error.into())
            }
        }
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        if self.remove {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

pub(super) fn policy(root: &Path, base: &str, candidate: &str) -> DynResult<()> {
    source::revision(base)?;
    source::revision(candidate)?;
    if candidate != base {
        let parents = process::text(root, &["rev-list", "--parents", "-n", "1", candidate])?;
        if parents.split_whitespace().collect::<Vec<_>>() != [candidate, base] {
            return Err("candidate is not a single-parent direct child of frozen base".into());
        }
        if !process::text(
            root,
            &[
                "diff",
                "--name-only",
                base,
                candidate,
                "--",
                ".github",
                ".agents",
                "scripts",
                ".gitattributes",
                "ci/ci.md",
                "ci/llama-canary/agent-repair-prompt.md",
            ],
        )?
        .is_empty()
        {
            return Err("candidate changed protected orchestration".into());
        }
    }
    Ok(())
}

pub(super) fn producer(root: &Path, base: &str, candidate: &str) -> DynResult<()> {
    policy(root, base, candidate)?;
    let head = process::text(root, &["rev-parse", "HEAD"])?;
    if head != base && head != candidate {
        return Err("producer checkout differs from frozen base/candidate".into());
    }
    process::text(root, &["diff", "--cached", "--exit-code", candidate, "--"])?;
    process::text(root, &["diff", "--exit-code", candidate, "--"])?;
    if !process::text(root, &["ls-files", "--others", "--exclude-standard"])?.is_empty() {
        return Err("producer has untracked source after snapshot".into());
    }
    Ok(())
}

pub(super) fn materialize(root: &Path, candidate: &str, parent: &Path) -> DynResult<Owned> {
    source::revision(candidate)?;
    let owned = Owned::new(parent, "candidate-view")?;
    let target = owned.path.join("source");
    process::git(
        &owned.path,
        &[
            "clone".into(),
            "--quiet".into(),
            "--no-checkout".into(),
            "--no-hardlinks".into(),
            root.canonicalize()?.into(),
            target.clone().into(),
        ],
        None,
    )?;
    process::text(&target, &["checkout", "--quiet", "--detach", candidate])?;
    if process::text(&target, &["rev-parse", "HEAD"])? != candidate {
        return Err("materialized source differs from candidate".into());
    }
    Ok(owned)
}
