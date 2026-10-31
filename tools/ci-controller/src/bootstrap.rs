//! Bootstrap contract for repository automation tools: prove the running
//! binary is Cargo's own build output and that it compiled only the workspace
//! crates its descriptor admits (so no llama.cpp or UI preparation).

mod dependency_boundary;
#[cfg(test)]
mod tests;

use crate::DynResult;
use crate::child_output::trimmed_stderr_or_stdout;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

const TOOLCHAIN_HINT: &str = "install the pinned Rust toolchain (CI: dtolnay/rust-toolchain via .github/actions/prepare-automation; local: rustup) and ensure `cargo` is on PATH or CARGO names it";

/// One automation executable's bootstrap identity. Each binary owns its
/// descriptor; the check and its report format are shared.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapTool {
    /// Workspace package that builds the running executable.
    pub package: &'static str,
    /// Workspace packages its compiled closure may contain.
    pub local_packages: &'static [&'static str],
    /// Usage error for unexpected arguments.
    pub usage: &'static str,
    /// Just recipe that rebuilds the executable in place.
    pub recipe: &'static str,
}

/// The narrow CI controller: no other workspace crate may be compiled in.
pub const CONTROLLER: BootstrapTool = BootstrapTool {
    package: "ci-controller",
    local_packages: &["ci-controller"],
    usage: "usage: ci-controller automation bootstrap",
    recipe: "just ci-controller-bootstrap",
};

pub fn run(repo_root: &Path, args: &[String], tool: &BootstrapTool) -> DynResult<()> {
    if !args.is_empty() {
        return Err(tool.usage.into());
    }
    let cargo = Cargo::from_env();
    let host = cargo.host_triple()?;
    let metadata = cargo.resolved_metadata(repo_root, &host)?;
    dependency_boundary::check(&metadata, tool)?;
    let binary = tool_binary(
        &metadata.target_directory,
        &std::env::current_exe()?,
        tool.recipe,
    )?;
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
        let output = command.output().map_err(|error| {
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
fn tool_binary(target_directory: &Path, current_exe: &Path, recipe: &str) -> DynResult<PathBuf> {
    let target = target_directory.canonicalize().map_err(|error| {
        format!(
            "automation bootstrap: Cargo target directory {} is not built ({error}); run `{recipe}`",
            target_directory.display()
        )
    })?;
    let binary = current_exe.canonicalize()?;
    let profile_parent = binary.parent().and_then(Path::parent);
    if profile_parent != Some(target.as_path()) {
        return Err(format!(
            "automation bootstrap: running tool {} is not Cargo's build output under {}; run `{recipe}`",
            binary.display(),
            target.display()
        )
        .into());
    }
    Ok(binary)
}
