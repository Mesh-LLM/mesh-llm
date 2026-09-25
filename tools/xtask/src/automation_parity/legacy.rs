//! The transitional shadow comparison's legacy launches. Both targets are
//! fixed files in the protected checkout; a caller chooses the interpreter,
//! never the script. Task 28 deletes this file with the legacy planner.

use super::process::{Captured, run_bounded};
use crate::command::DynResult;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const PLANNER: &str = "scripts/plan-ci.py";
const IDENTITY: &str = "scripts/runner-image-identity.py";

/// The protected planner file, rejecting any other path the caller names.
pub(super) fn protected_planner(root: &Path, requested: Option<&Path>) -> DynResult<PathBuf> {
    let planner = root.join(PLANNER);
    let metadata = fs::symlink_metadata(&planner)?;
    if !metadata.file_type().is_file() {
        let shown = planner.display();
        return Err(format!("{shown} must be a regular file in the protected checkout").into());
    }
    if let Some(requested) = requested {
        let resolved = requested
            .canonicalize()
            .map_err(|error| format!("invalid planner {}: {error}", requested.display()))?;
        if resolved != planner.canonicalize()? {
            return Err(format!(
                "planner {} is not the protected checkout's {}; a PR-controlled planner is never executed",
                requested.display(),
                planner.display()
            )
            .into());
        }
    }
    Ok(planner)
}

/// One legacy side: an operator-chosen interpreter over protected scripts.
pub(super) struct Legacy {
    pub(super) interpreter: PathBuf,
    pub(super) root: PathBuf,
}

/// A fixed protected script plus its arguments, stdin and `PATH`.
struct Launch<'a> {
    script: &'a Path,
    args: &'a [&'a str],
    search_path: &'a str,
    stdin: &'a [u8],
}

impl Legacy {
    fn spawn(&self, launch: &Launch<'_>) -> DynResult<Captured> {
        let python = self.interpreter.as_path();
        let mut command = Command::new(python);
        command
            .current_dir(&self.root)
            .arg(launch.script)
            .args(launch.args)
            .env("PATH", launch.search_path)
            .env("PYTHONDONTWRITEBYTECODE", "1");
        run_bounded(&mut command, launch.stdin)
    }

    /// The protected planner with `args` and `stdin`.
    pub(super) fn plan(
        &self,
        args: &[&str],
        search_path: &str,
        stdin: &[u8],
    ) -> DynResult<Captured> {
        let script = protected_planner(&self.root, None)?;
        self.spawn(&Launch {
            script: &script,
            args,
            search_path,
            stdin,
        })
    }

    /// Runner-image identity `--root <checkout> <subcommand>`; `check` and
    /// `diagnose` import the protected planner through importlib.
    pub(super) fn identity(&self, subcommand: &str, search_path: &str) -> DynResult<Captured> {
        let root = self.root.to_str().ok_or("non-UTF8 checkout path")?;
        let script = self.root.join(IDENTITY);
        let args = ["--root", root, subcommand];
        self.spawn(&Launch {
            script: &script,
            args: &args,
            search_path,
            stdin: b"",
        })
    }
}
