use super::{Error, Git, admission::Admitted, replay};
use crate::command_interrupt::Interrupt;
use std::{io::Write, path::Path};

pub(super) fn execute(
    admitted: Admitted<'_>,
    git: Option<&Git>,
    context: (&Interrupt, &mut impl Write),
) -> Result<(), Error> {
    let (interrupt, output) = context;
    let plan = admitted.plan();
    if let Some((workspace, root)) = &plan.replay {
        if root.is_symlink() {
            return Err(Error::ReplayRoot(root.clone()));
        }
        let git = git.ok_or(Error::Input(
            "replay requires an explicit absolute Git adapter",
        ))?;
        let listing = git.list(workspace, interrupt)?;
        let worktrees = replay::owned_worktrees(root, &listing)?;
        for path in worktrees {
            git.remove(workspace, &path, interrupt)?;
        }
    }
    for target in &plan.targets {
        interrupt.check()?;
        remove(&target.path)?;
        writeln!(output, "Cleaned job output: {}", target.path.display())?;
        output.flush()?;
    }
    Ok(())
}

fn remove(path: &Path) -> Result<(), Error> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || metadata.is_file() => {
            std::fs::remove_file(path)?
        }
        Ok(_) => std::fs::remove_dir_all(path)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
