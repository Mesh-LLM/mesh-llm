use super::{
    Plan, Target,
    fixtures::{Fixture, ownership},
};
use std::path::Path;

#[cfg(unix)]
#[test]
fn unregistered_git_destination_when_caller_supplies_root_is_refused() {
    let fixture = Fixture::new();
    let arguments = [
        "worktree".into(),
        "add".into(),
        "--detach".into(),
        Path::new(env!("CARGO_MANIFEST_DIR")).as_os_str().to_owned(),
        "HEAD".into(),
    ];

    let refusal =
        std::panic::catch_unwind(|| super::fixture_git_admission::check(&fixture, &arguments));

    assert!(refusal.is_err());
    assert!(!fixture.workspace.join(".git").exists());
}

#[test]
fn caller_roots_when_not_owned_are_refused_before_deletion() {
    let fixture = Fixture::new();
    let safe = fixture.workspace.join("target/retained");
    fixture.seed(&safe);
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    let mut roots = vec![
        repository.to_path_buf(),
        Path::new("/").to_path_buf(),
        fixture.root.path().to_path_buf(),
        fixture.root.path().parent().unwrap().to_path_buf(),
    ];
    roots.extend(
        ["HOME", "USERPROFILE", "CARGO_HOME", "XDG_CACHE_HOME"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(std::path::PathBuf::from),
    );
    for root in roots {
        let plan = Plan::files(vec![
            Target::new(&fixture.workspace, safe.clone()),
            Target::new(&root, root.join("target/debug")),
        ]);
        let refusal = std::panic::catch_unwind(|| fixture.delete(&plan));
        assert!(refusal.is_err());
        assert_eq!(std::fs::read(safe.join("payload")).unwrap(), b"sentinel");
    }
}

#[test]
fn copied_marker_when_another_tempdir_is_supplied_does_not_transfer_ownership() {
    let fixture = Fixture::new();
    let other = Fixture::new();
    let target = other.workspace.join("target/debug");
    other.seed(&target);
    std::fs::copy(
        fixture.root.path().join(".runner-cleanup-fixture-owner"),
        other.root.path().join(".runner-cleanup-fixture-owner"),
    )
    .unwrap();

    let refusal = ownership::check(&other.root, &target);

    assert!(refusal.is_err());
    assert_eq!(std::fs::read(target.join("payload")).unwrap(), b"sentinel");
}

#[test]
fn missing_marker_when_the_owned_plan_is_executed_preserves_outputs() {
    let fixture = Fixture::new();
    let plan = fixture.plan(super::Profile::RunnerContract, true);
    fixture.seed(&plan.targets[0].path);
    std::fs::remove_file(fixture.root.path().join(".runner-cleanup-fixture-owner")).unwrap();

    let refusal = std::panic::catch_unwind(|| fixture.delete(&plan));

    assert!(refusal.is_err());
    assert_eq!(
        std::fs::read(plan.targets[0].path.join("payload")).unwrap(),
        b"sentinel"
    );
}

#[cfg(unix)]
#[test]
fn parent_symlink_when_fixture_targets_escape_preserves_both_owners() {
    let fixture = Fixture::new();
    let other = Fixture::new();
    let safe = fixture.workspace.join("safe");
    let external = other.workspace.join("external");
    fixture.seed(&safe);
    other.seed(&external);
    let link = fixture.workspace.join("linked-parent");
    std::os::unix::fs::symlink(&other.workspace, &link).unwrap();
    let plan = Plan::files(vec![
        Target::new(&fixture.workspace, safe.clone()),
        Target::new(&fixture.workspace, link.join("external")),
    ]);

    let refusal = std::panic::catch_unwind(|| fixture.delete(&plan));

    assert!(refusal.is_err());
    assert_eq!(std::fs::read(safe.join("payload")).unwrap(), b"sentinel");
    assert_eq!(
        std::fs::read(external.join("payload")).unwrap(),
        b"sentinel"
    );
}

#[cfg(unix)]
#[test]
fn external_git_directory_when_workspace_is_rebound_refuses_before_spawn() {
    let repository = super::replay_tests::Repository::new();
    let other = Fixture::new();
    let external = other.workspace.join("git-sentinel");
    other.seed(&external);
    let git = repository.fixture.workspace.join(".git");
    let retained = repository.fixture.workspace.join(".git-retained");
    std::fs::rename(&git, &retained).unwrap();
    std::os::unix::fs::symlink(&external, &git).unwrap();

    let refusal = std::panic::catch_unwind(|| repository.list());

    assert!(refusal.is_err());
    assert_eq!(
        std::fs::read(external.join("payload")).unwrap(),
        b"sentinel"
    );
    assert!(retained.is_dir());
}
