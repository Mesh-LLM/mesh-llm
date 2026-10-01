use super::{Behavior, audit, fixture_interruption, signals};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
};
#[path = "cleanup_ownership.rs"]
mod ownership;

pub(super) fn run(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var_os("HOME").ok_or("Git fixture requires owned HOME")?);
    let executable = std::env::current_exe()?;
    let owner = executable
        .parent()
        .ok_or("missing fixture executable parent")?;
    ownership::validate(owner, &executable)?;
    if root != owner.join("native") {
        return Err("Git fixture requires registered native root".into());
    }
    ownership::validate(owner, &root)?;
    let workspace = owner.join("workspace");
    let owned = owner.join("temp/agentic-replay-worktrees/owned");
    ownership::validate(owner, &workspace)?;
    ownership::validate(owner, &owned)?;
    for path in [&executable, &root, &workspace, &owned] {
        if path.is_symlink() {
            return Err("Git fixture refuses symlink resources".into());
        }
    }
    let listing = format!("worktree {}\0", owned.display()).into_bytes();
    if std::fs::read(root.join("listing"))? != listing {
        return Err("Git fixture requires exactly the registered owned worktree".into());
    }
    if std::env::var("GIT_MASTER")?.as_str() != "1" {
        return Err("Git fixture requires GIT_MASTER=1".into());
    }
    match arguments {
        [flag, directory, worktree, list, porcelain, nul]
            if flag == "-C"
                && Path::new(directory) == workspace.as_path()
                && worktree == "worktree"
                && list == "list"
                && porcelain == "--porcelain"
                && nul == "-z" =>
        {
            io::stdout().write_all(&listing)?;
            Ok(())
        }
        [flag, directory, worktree, remove, force, target]
            if flag == "-C"
                && Path::new(directory) == workspace.as_path()
                && Path::new(target) == owned.as_path()
                && worktree == "worktree"
                && remove == "remove"
                && force == "--force" =>
        {
            signals::install()?;
            audit(&root, arguments)?;
            std::env::set_current_dir(&root)?;
            fixture_interruption::branch(&root, Behavior::LiveDescendant)
        }
        _ => Err("unexpected Git fixture argv".into()),
    }
}
