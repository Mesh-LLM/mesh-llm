//! Scratch space for one suite run: tool directories placed first on
//! `PATH`, and fixture manifest sets materialized as `ci/*.yml`.

use crate::command::{DynResult, unique_temp_dir};
use std::fs;
use std::path::{Path, PathBuf};

/// A private scratch directory, removed when the run ends.
pub(super) struct Scratch(PathBuf);

impl Scratch {
    pub(super) fn new() -> DynResult<Self> {
        let path = unique_temp_dir("xtask-ci-parity");
        fs::create_dir_all(&path)?;
        Ok(Self(path.canonicalize()?))
    }

    pub(super) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

/// Which tool directory goes first on `PATH`.
#[derive(Clone, Copy)]
pub(super) enum Tools {
    /// The real Cargo; only the optional bash override is prepended.
    Real,
    /// A `cargo` that replays the frozen fixture metadata, as the goldens
    /// were generated.
    FixtureCargo,
}

pub(super) struct Stage {
    scratch: Scratch,
    fixtures: PathBuf,
}

impl Stage {
    pub(super) fn new(fixtures: &Path, bash: Option<&Path>) -> DynResult<Self> {
        let stage = Self {
            scratch: Scratch::new()?,
            fixtures: fixtures.to_path_buf(),
        };
        for directory in ["real-bin", "fixture-bin"] {
            let bin = stage.root().join(directory);
            fs::create_dir_all(&bin)?;
            if let Some(bash) = bash {
                link(bash, &bin.join("bash"))?;
            }
        }
        let metadata = fixtures.join("cargo-metadata.json");
        if !metadata.is_file() {
            return Err(format!("missing fixture {}", metadata.display()).into());
        }
        let cargo = stage.root().join("fixture-bin/cargo");
        fs::write(
            &cargo,
            format!("#!/bin/sh\nexec cat '{}'\n", metadata.display()),
        )?;
        make_executable(&cargo)?;
        Ok(stage)
    }

    pub(super) fn root(&self) -> &Path {
        self.scratch.path()
    }

    pub(super) fn fixtures(&self) -> &Path {
        &self.fixtures
    }

    /// `PATH` with the selected tool directory first.
    pub(super) fn search_path(&self, tools: Tools) -> String {
        let directory = match tools {
            Tools::Real => "real-bin",
            Tools::FixtureCargo => "fixture-bin",
        };
        let inherited = std::env::var("PATH").unwrap_or_default();
        format!("{}:{inherited}", self.root().join(directory).display())
    }

    /// The directory under which frozen manifest sets are materialized; the
    /// goldens name it `<manifests>/`.
    pub(super) fn manifests(&self) -> PathBuf {
        self.root().join("manifests")
    }

    /// Materializes fixture manifest set `name` as `<name>/ci/*.yml`.
    pub(super) fn manifest_set(&self, name: &str) -> DynResult<PathBuf> {
        let target = self.manifests().join(name);
        if target.is_dir() {
            return Ok(target);
        }
        let source = self.fixtures.join("manifests").join(name);
        fs::create_dir_all(target.join("ci"))?;
        for catalog in ["ownership", "slices"] {
            fs::copy(
                source.join(format!("{catalog}.json")),
                target.join("ci").join(format!("{catalog}.yml")),
            )?;
        }
        Ok(target)
    }

    /// A fresh directory for one comparison's files.
    pub(super) fn directory(&self, name: &str) -> DynResult<PathBuf> {
        let path = self.root().join("work").join(name);
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}

#[cfg(unix)]
fn link(target: &Path, link: &Path) -> DynResult<()> {
    std::os::unix::fs::symlink(target, link)?;
    Ok(())
}

#[cfg(not(unix))]
fn link(_target: &Path, _link: &Path) -> DynResult<()> {
    Err("a bash override requires a Unix host".into())
}

#[cfg(unix)]
fn make_executable(path: &Path) -> DynResult<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> DynResult<()> {
    Err("the fixture cargo requires a Unix host".into())
}
