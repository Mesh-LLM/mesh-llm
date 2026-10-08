use crate::cleanup_owner as ownership;
use crate::{
    protocol::Behavior,
    pty::{self, Terminal},
    support::{Case, Sentinel, assert_absent},
};
use std::{
    fs::File,
    process::{Command, Stdio},
};

#[test]
fn cleanup_git_when_caller_supplies_repository_workspace_refuses_before_child_start() {
    let case = Case::new(Behavior::Clean, vec![]);
    ownership::register(&case.root);
    let root = case.native.parent().unwrap();
    let owned = root.join("temp/agentic-replay-worktrees/owned");
    std::fs::write(
        case.native.join("listing"),
        format!("worktree {}\0", owned.display()),
    )
    .unwrap();
    ownership::check(&case.root, &case.binary).unwrap();
    let mut command = Command::new(&case.binary);
    command
        .env_clear()
        .env("HOME", &case.native)
        .env("GIT_MASTER", "1")
        .arg("-C")
        .arg(crate::support::repository())
        .args(["worktree", "remove", "--force"])
        .arg(&owned);

    let output = command.output().unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!case.native.join("audit.json").exists());
    assert!(!case.native.join("leaf.pid").exists());
}

#[test]
fn runner_cleanup_sigterm_during_git_remove_retains_finalization_and_owned_cleanup() {
    let case = Case::new(Behavior::Clean, vec![]);
    ownership::register(&case.root);
    let root = case.native.parent().unwrap();
    let workspace = root.join("workspace");
    let temporary = root.join("temp");
    let owned = temporary.join("agentic-replay-worktrees/owned");
    let venv = workspace.join("ci/agentic-replay-nightly/.venv");
    for path in [&owned, &venv, &case.native, &case.binary] {
        ownership::check(&case.root, path).unwrap();
    }
    for path in [&owned, &venv] {
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(path.join("payload"), b"retained").unwrap();
    }
    std::fs::write(
        case.native.join("listing"),
        format!("worktree {}\0", owned.display()),
    )
    .unwrap();
    let mut sentinel = Sentinel::new(&case);
    let (master, slave) = pty::pair().unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(&workspace)
        .env_clear()
        .env("GITHUB_WORKSPACE", &workspace)
        .env("RUNNER_TEMP", &temporary)
        .env("HOME", &case.native)
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &case.native)
        .args([
            "ci-ops",
            "runner-cleanup",
            "--job",
            "replay",
            "--evidence-uploaded",
            "false",
            "--git",
        ])
        .arg(&case.binary)
        .stdin(Stdio::from(slave))
        .stdout(File::create(case.native.join("cli.stdout")).unwrap())
        .stderr(File::create(case.native.join("cli.stderr")).unwrap());
    for path in [
        &workspace,
        &temporary,
        &owned,
        &venv,
        &case.native,
        &case.binary,
    ] {
        ownership::check(&case.root, path).unwrap();
    }
    let mut terminal = Terminal::start_prepared(&case, command, master);
    terminal.wait_file("startup.armed");
    let git_pid = case.audit().pid;
    let leaf_pid = std::fs::read_to_string(case.native.join("leaf.pid"))
        .unwrap()
        .parse()
        .unwrap();

    terminal.terminate();
    let status = terminal.finish();

    let diagnostic = std::fs::read_to_string(case.native.join("cli.stderr")).unwrap();
    assert_eq!(status.code(), Some(1), "{status:?}; {diagnostic}");
    assert!(
        diagnostic.contains("cleanup finalization failed"),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("cancelled by command interruption"),
        "{diagnostic}"
    );
    assert!(diagnostic.contains("outcome=Cancelled"), "{diagnostic}");
    assert!(diagnostic.contains("complete: true"), "{diagnostic}");
    assert_absent(git_pid);
    assert_absent(leaf_pid);
    assert!(sentinel.0.try_wait().unwrap().is_none());
    assert_eq!(std::fs::read(venv.join("payload")).unwrap(), b"retained");
    assert_eq!(std::fs::read(owned.join("payload")).unwrap(), b"retained");
    assert!(
        std::fs::read(case.native.join("cli.stdout"))
            .unwrap()
            .is_empty()
    );
    terminal.disarm();
}
