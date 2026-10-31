pub mod affected_crates;
pub mod check_args;
pub mod check_report;
pub mod conventional_commit;
pub mod text;
pub mod upstream_pin;

use crate::DynResult;
use std::path::{Path, PathBuf};

/// Separates xtask's optional leading repository root from its command tail.
pub fn parse_root_prefix(args: &[String]) -> DynResult<(Option<PathBuf>, &[String])> {
    match args {
        [flag, value, rest @ ..] if flag == "--repo-root" => {
            if value.starts_with("--") {
                return Err("missing value for --repo-root".into());
            }
            Ok((Some(Path::new(value).to_path_buf()), rest))
        }
        [flag, ..] if flag == "--repo-root" => Err("missing value for --repo-root".into()),
        _ => Ok((None, args)),
    }
}

#[derive(Debug, Clone)]
pub struct RepositoryRoot(PathBuf);

impl RepositoryRoot {
    pub fn resolve(explicit: Option<&Path>) -> DynResult<Self> {
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

    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

/// Discovery roots for `repository` checks. `--repo-root` (when given)
/// replaces the working directory for Cargo/git discovery and the default
/// checkout; otherwise checks resolve the checkout lazily, only when their own
/// `--root`/`--repository` is absent.
pub struct CheckRoots {
    cwd: PathBuf,
    explicit: Option<RepositoryRoot>,
}

impl CheckRoots {
    pub fn new(explicit: Option<RepositoryRoot>) -> DynResult<Self> {
        let cwd = match &explicit {
            Some(root) => root.as_path().to_path_buf(),
            None => std::env::current_dir()?,
        };
        Ok(Self { cwd, explicit })
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn default_root(self) -> impl FnOnce() -> DynResult<PathBuf> {
        move || {
            let root = match self.explicit {
                Some(root) => root,
                None => RepositoryRoot::resolve(None)?,
            };
            Ok(root.as_path().to_path_buf())
        }
    }
}
