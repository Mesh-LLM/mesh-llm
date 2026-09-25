mod affected_crates;
pub(crate) mod check_args;
pub(crate) mod check_report;
mod conventional_commit;
mod env_census;
pub(crate) mod python_text;
mod upstream_pin;

use crate::cli::RepositoryCheck;
use crate::command::DynResult;
use std::path::{Path, PathBuf};

/// Runs a ported check. `--repo-root` (when given) replaces the working
/// directory for Cargo/git discovery and the default checkout for the
/// Python-derived checks; otherwise they resolve the checkout lazily, only
/// when their own `--root`/`--repository` is absent.
pub(crate) fn run_check(
    check: RepositoryCheck,
    args: &[String],
    explicit_root: Option<RepositoryRoot>,
) -> DynResult<()> {
    let cwd = match &explicit_root {
        Some(root) => root.as_path().to_path_buf(),
        None => std::env::current_dir()?,
    };
    let default_root = move || -> DynResult<PathBuf> {
        let root = match explicit_root {
            Some(root) => root,
            None => RepositoryRoot::resolve(None)?,
        };
        Ok(root.0)
    };
    match check {
        RepositoryCheck::AffectedCrates => affected_crates::run(&cwd, args),
        RepositoryCheck::ConventionalCommits => conventional_commit::run(&cwd, args),
        RepositoryCheck::EnvMutationCensus => env_census::run(args, default_root),
        RepositoryCheck::LlamaUpstreamPin => upstream_pin::run(args, default_root),
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RepositoryRoot(PathBuf);

impl RepositoryRoot {
    pub(crate) fn resolve(explicit: Option<&Path>) -> DynResult<Self> {
        let start = match explicit {
            Some(path) => path
                .canonicalize()
                .map_err(|error| format!("invalid repo root {}: {error}", path.display()))?,
            None => std::env::current_dir()?,
        };
        let root = match explicit {
            Some(_) => Some(start.as_path()),
            None => start.ancestors().find(|path| Self::has_markers(path)),
        }
        .ok_or_else(|| format!("could not find repo root from {}", start.display()))?;
        if !Self::has_markers(root) {
            return Err(format!(
                "invalid repo root {}: missing workspace markers",
                root.display()
            )
            .into());
        }
        Ok(Self(root.to_path_buf()))
    }

    fn has_markers(path: &Path) -> bool {
        path.join("Cargo.toml").is_file() && path.join("tools/xtask/Cargo.toml").is_file()
    }

    pub(crate) fn as_path(&self) -> &Path {
        &self.0
    }
}
