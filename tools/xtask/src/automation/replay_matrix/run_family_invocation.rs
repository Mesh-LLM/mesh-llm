use super::{Options, run_family_process::InvocationContext, run_family_report::RunFamilyReport};
use crate::automation::replay_matrix::family_workload;
use std::{ffi::OsString, path::PathBuf};

pub(super) struct PreparedInvocation {
    pub executable: PathBuf,
    pub arguments: Vec<OsString>,
    pub workload: family_workload::Prepared,
}

pub(super) fn prepare(
    options: &Options,
    context: &InvocationContext<'_>,
) -> Result<PreparedInvocation, RunFamilyReport> {
    prepare_input(options, context)
        .map_err(|error| RunFamilyReport::failure(format!("run-family preparation: {error}\n")))
}

fn prepare_input(
    options: &Options,
    context: &InvocationContext<'_>,
) -> crate::command::DynResult<PreparedInvocation> {
    let repo = context.repo_root.canonicalize()?;
    let output = std::path::absolute(context.output)?;
    let python = executable(context.python)?;
    let worktree_root = std::env::var_os("AGENTIC_REPLAY_WORKTREE_ROOT")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or(
            repo.parent()
                .ok_or("repository has no parent")?
                .join(".agentic-replay-worktrees"),
        );
    let worktree_root = std::path::absolute(worktree_root)?;
    let refs = options
        .refs
        .iter()
        .map(|reference| {
            reference
                .to_str()
                .map(str::to_owned)
                .ok_or("non-Unicode replay ref")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let git = tool("git")?;
    let just = tool("just")?;
    let workload = family_workload::prepare(&family_workload::Context {
        matrix: &options.matrix,
        family: context.family,
        model: context.model,
        dataset: context.dataset,
        python: &python,
        output: &output,
        refs: &refs,
        repo: &repo,
        worktree_root: &worktree_root,
        git: &git,
        just: &just,
        timeout_seconds: context.timeout.as_secs(),
    })?;
    let input = workload.staging.root().join("input.json");
    crate::command::write_json_file(&input, &workload.input)?;
    Ok(PreparedInvocation {
        executable: std::env::current_exe()?.canonicalize()?,
        arguments: vec![
            "--repo-root".into(),
            repo.into_os_string(),
            "automation".into(),
            "replay-matrix".into(),
            "execute-run".into(),
            "--input".into(),
            input.into_os_string(),
        ],
        workload,
    })
}

pub(super) use crate::automation::replay_matrix::executable_resolution::{executable, tool};
