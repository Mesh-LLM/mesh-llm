use super::fixtures::{Fixture, ownership};
use std::{ffi::OsString, path::Path};

pub(super) fn workspace(fixture: &Fixture) {
    assert_eq!(fixture.workspace, fixture.root.path().join("workspace"));
    ownership::check(&fixture.root, &fixture.workspace).unwrap();
    let git = fixture.workspace.join(".git");
    ownership::check(&fixture.root, &git).unwrap();
    if git.exists() {
        assert!(std::fs::symlink_metadata(&git).unwrap().is_dir());
        assert_eq!(git.canonicalize().unwrap(), git);
    }
}

pub(super) fn check(fixture: &Fixture, args: &[OsString]) {
    match args {
        [init, template] if init == "init" && template == "--template=" => {}
        [worktree, list, porcelain, nul]
            if worktree == "worktree"
                && list == "list"
                && porcelain == "--porcelain"
                && nul == "-z" => {}
        [worktree, add, detach, path, head]
            if worktree == "worktree" && add == "add" && detach == "--detach" && head == "HEAD" =>
        {
            ownership::check(&fixture.root, Path::new(path)).unwrap();
            assert!(!Path::new(path).is_symlink());
        }
        [worktree, lock, path] if worktree == "worktree" && lock == "lock" => {
            ownership::check(&fixture.root, Path::new(path)).unwrap();
            assert!(!Path::new(path).is_symlink());
        }
        [
            name_flag,
            name,
            email_flag,
            email,
            sign_flag,
            sign,
            hooks_flag,
            hooks,
            commit,
            empty,
            message,
            text,
        ] if name_flag == "-c"
            && name == "user.name=Cleanup Test"
            && email_flag == "-c"
            && email == "user.email=cleanup@example.invalid"
            && sign_flag == "-c"
            && sign == "commit.gpgsign=false"
            && hooks_flag == "-c"
            && hooks == "core.hooksPath=disabled-hooks"
            && commit == "commit"
            && empty == "--allow-empty"
            && message == "-m"
            && text == "fixture" => {}
        _ => panic!("unregistered fixture Git command"),
    }
}
