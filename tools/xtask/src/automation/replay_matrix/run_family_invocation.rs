use super::super::parameters::Parameters;
use super::Options;
use super::run_family_process::InvocationContext;
use super::run_family_report::RunFamilyReport;
use crate::automation::codepoint_json::{strings::JsonString as ReplayString, value::Value};
use crate::automation::replay_matrix::invocation;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub(super) struct PreparedInvocation {
    pub(super) python: PathBuf,
    pub(super) arguments: Vec<OsString>,
}

pub(super) fn prepare(
    options: &Options,
    parameters: &Parameters,
    matrix: &Value,
    context: &InvocationContext<'_>,
) -> Result<PreparedInvocation, RunFamilyReport> {
    let family: ReplayString = context.family.into();
    let selected = match invocation::select(matrix.get("models"), parameters, &family) {
        Ok(selected) => selected,
        Err(error) => {
            return Err(RunFamilyReport::failure(format!(
                "run-family selection: {error:?}\n"
            )));
        }
    };
    let python = match executable(context.python) {
        Ok(path) => path,
        Err(message) => return Err(RunFamilyReport::failure(format!("{message}\n"))),
    };
    let script = match context
        .repo_root
        .join("evals/agentic-replay.py")
        .canonicalize()
    {
        Ok(script) => script,
        Err(error) => {
            return Err(RunFamilyReport::failure(format!(
                "run-family script: {error}\n"
            )));
        }
    };
    let worktree_root = std::env::var_os("AGENTIC_REPLAY_WORKTREE_ROOT");
    let dataset = pathlib_path(context.dataset);
    let output = pathlib_path(context.output);
    let invocation = invocation::ReplayInvocation {
        python: python.as_os_str(),
        script: &script,
        dataset: &dataset,
        output: &output,
        worktree_root: worktree_root.as_deref(),
        refs: &options.refs,
    };
    let command = invocation::argv(&selected, parameters, &invocation);
    let arguments = invocation::os_argv(&command[1..])
        .map_err(|error| RunFamilyReport::failure(format!("run-family argv: {error:?}\n")))?;
    Ok(PreparedInvocation { python, arguments })
}

pub(super) fn executable(path: &Path) -> Result<PathBuf, &'static str> {
    if !path.is_absolute() {
        return Err("--python must be an absolute executable path");
    }
    let path = path
        .canonicalize()
        .map_err(|_| "--python must name an existing absolute executable")?;
    if !path.is_absolute() || !path.is_file() {
        return Err("--python must name an existing absolute executable");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = path
            .metadata()
            .map_err(|_| "--python must name an existing absolute executable")?;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err("--python must be an executable file");
        }
    }
    #[cfg(windows)]
    if path
        .extension()
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("exe"))
    {
        return Err("--python must be an explicit .exe executable");
    }
    Ok(path)
}

fn pathlib_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => (),
            std::path::Component::RootDir
            | std::path::Component::Prefix(_)
            | std::path::Component::ParentDir
            | std::path::Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        normalized.push(".");
    }
    normalized
}
