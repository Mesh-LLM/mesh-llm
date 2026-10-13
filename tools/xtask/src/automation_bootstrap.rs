//! Bootstrap contract for the repository automation tool: prove the running
//! `xtask` binary is Cargo's own build output and that it compiled without any
//! product, native-runtime, or UI crate (so no llama.cpp or UI preparation).

mod dependency_boundary;
#[cfg(test)]
mod tests;

use crate::command::{DynResult, run_command, trimmed_stderr_or_stdout};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

const TOOL_PACKAGE: &str = "xtask";
const TOOLCHAIN_HINT: &str = "install the pinned Rust toolchain (CI: dtolnay/rust-toolchain via .github/actions/prepare-automation; local: rustup) and ensure `cargo` is on PATH or CARGO names it";

pub(crate) fn run(repo_root: &Path, args: &[String]) -> DynResult<()> {
    if !args.is_empty() {
        return Err("usage: cargo xtool automation bootstrap".into());
    }
    let cargo = Cargo::from_env();
    let host = cargo.host_triple()?;
    let metadata = cargo.resolved_metadata(repo_root, &host)?;
    dependency_boundary::check(&metadata, TOOL_PACKAGE)?;
    let binary = tool_binary(&metadata.target_directory, &std::env::current_exe()?)?;
    println!("binary_path={}", binary.display());
    println!("target_directory={}", metadata.target_directory.display());
    println!("host={host}");
    Ok(())
}

struct Cargo(OsString);

impl Cargo {
    fn from_env() -> Self {
        Self(std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
    }

    fn output(&self, repo_root: Option<&Path>, args: &[&str]) -> DynResult<std::process::Output> {
        let mut command = Command::new(&self.0);
        if let Some(root) = repo_root {
            command.current_dir(root);
        }
        command.args(args);
        let output = run_command(&mut command).map_err(|error| {
            format!(
                "automation bootstrap: cannot run `{}` ({error}); {TOOLCHAIN_HINT}",
                self.0.to_string_lossy()
            )
        })?;
        if !output.status.success() {
            return Err(format!(
                "automation bootstrap: `cargo {}` failed: {}",
                args.join(" "),
                trimmed_stderr_or_stdout(&output)
            )
            .into());
        }
        Ok(output)
    }

    fn host_triple(&self) -> DynResult<String> {
        let output = self.output(None, &["-vV"])?;
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .map(str::to_owned)
            .ok_or_else(|| {
                format!("automation bootstrap: `cargo -vV` reported no host; {TOOLCHAIN_HINT}")
                    .into()
            })
    }

    fn resolved_metadata(
        &self,
        repo_root: &Path,
        host: &str,
    ) -> DynResult<dependency_boundary::ResolvedMetadata> {
        let output = self.output(
            Some(repo_root),
            &[
                "metadata",
                "--format-version=1",
                "--locked",
                "--filter-platform",
                host,
            ],
        )?;
        Ok(serde_json::from_slice(&output.stdout)?)
    }
}

/// Returns the absolute tool path when the running executable is Cargo's
/// build output directly beneath the resolved target directory.
fn tool_binary(target_directory: &Path, current_exe: &Path) -> DynResult<PathBuf> {
    let target = target_directory.canonicalize().map_err(|error| {
        format!(
            "automation bootstrap: Cargo target directory {} is not built ({error}); run `just automation-bootstrap`",
            target_directory.display()
        )
    })?;
    let binary = current_exe.canonicalize()?;
    let profile_parent = binary.parent().and_then(Path::parent);
    if profile_parent != Some(target.as_path()) {
        return Err(format!(
            "automation bootstrap: running tool {} is not Cargo's build output under {}; run `just automation-bootstrap`",
            binary.display(),
            target.display()
        )
        .into());
    }
    Ok(binary)
}
