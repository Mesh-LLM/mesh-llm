use super::{Plan, Profile, Target, admission, boundary, fixtures::Fixture};

#[test]
fn repeated_cleanup_when_outputs_are_missing_preserves_sentinels_and_line_order() {
    for profile in [
        Profile::Build,
        Profile::Family,
        Profile::Replay,
        Profile::CudaRelease,
        Profile::Smoke,
        Profile::RunnerContract,
        Profile::CanaryPreflight,
    ] {
        let fixture = Fixture::new();
        let mut plan = fixture.plan(profile, true);
        plan.replay = None;
        for target in &plan.targets {
            fixture.seed(&target.path);
        }
        let sentinels = [
            fixture.workspace.join("source"),
            fixture.root.path().join("model-cache"),
            fixture
                .workspace
                .join(".deps/canary-input-999-1-repair-1-3"),
        ];
        for sentinel in &sentinels {
            fixture.seed(sentinel);
        }
        let expected = plan
            .targets
            .iter()
            .map(|target| format!("Cleaned job output: {}\n", target.path.display()))
            .collect::<String>()
            .into_bytes();
        assert_eq!(fixture.delete(&plan), expected);
        assert_eq!(fixture.delete(&plan), expected);
        assert!(plan.targets.iter().all(|target| !target.path.exists()));
        assert!(sentinels.iter().all(|path| path.join("payload").exists()));
    }
}

#[test]
fn smoke_rejection_when_any_input_is_unsafe_preserves_all_sentinels() {
    let fixture = Fixture::new();
    let sentinels = ["native-runtimes", "target", "ci-artifacts/linux", "source"]
        .map(|name| fixture.workspace.join(name));
    for path in &sentinels {
        fixture.seed(path);
    }
    for (key, value) in [
        ("CLEANUP_ARTIFACT_PATH", ""),
        ("CLEANUP_ARTIFACT_PATH", "scripts"),
        ("CLEANUP_ARTIFACT_PATH", "/tmp/output"),
        ("CLEANUP_ARTIFACT_PATH", "target/../../source"),
        ("CLEANUP_BINARY_PATH", "target"),
        ("CLEANUP_BINARY_PATH", "ci-artifacts"),
        ("CLEANUP_BINARY_PATH", "/"),
        ("CLEANUP_BINARY_PATH", "target/../source"),
    ] {
        let mut env = fixture.env.clone();
        env.insert(key.into(), value.into());
        assert!(
            boundary::from_environment(
                &super::Options::parse("smoke", "true", None).unwrap(),
                &env
            )
            .is_err()
        );
        assert!(sentinels.iter().all(|path| path.join("payload").exists()));
    }
}

#[test]
fn complete_list_rejection_when_last_target_escapes_preserves_first_output() {
    let fixture = Fixture::new();
    let safe = fixture.workspace.join("safe");
    fixture.seed(&safe);
    for invalid in [
        fixture.workspace.clone(),
        fixture.root.path().join("outside"),
        fixture.workspace.join("target/../../outside"),
    ] {
        let plan = Plan::files(vec![
            Target::new(&fixture.workspace, safe.clone()),
            Target::new(&fixture.workspace, invalid),
        ]);
        assert!(admission::admit(&plan).is_err());
        assert!(safe.join("payload").exists());
    }
}

#[cfg(unix)]
#[test]
fn symlinks_when_leaf_is_unlinked_but_parent_rejects_the_complete_list() {
    let fixture = Fixture::new();
    let outside = fixture.root.path().join("outside");
    fixture.seed(&outside.join("debug"));
    let link = fixture.workspace.join("target");
    std::os::unix::fs::symlink(&outside, &link).unwrap();
    let safe = fixture.workspace.join("safe");
    fixture.seed(&safe);
    let plan = Plan::files(vec![
        Target::new(&fixture.workspace, safe.clone()),
        Target::new(&fixture.workspace, link.join("debug")),
    ]);
    assert!(admission::admit(&plan).is_err());
    assert!(safe.join("payload").exists());
    fixture.delete(&Plan::files(vec![Target::new(
        &fixture.workspace,
        link.clone(),
    )]));
    assert!(!link.is_symlink());
    assert!(outside.join("debug/payload").exists());
}

#[cfg(unix)]
#[test]
fn resolve_follows_workspace_symlink_but_absolute_source_does_not() {
    let mut fixture = Fixture::new();
    let link = fixture.root.path().join("workspace-link");
    std::os::unix::fs::symlink(&fixture.workspace, &link).unwrap();
    fixture
        .env
        .insert("GITHUB_WORKSPACE".into(), link.clone().into_os_string());
    fixture
        .env
        .insert("CANARY_SOURCE_ROOT".into(), link.into_os_string());
    assert!(
        boundary::from_environment(
            &super::Options::parse("build", "false", None).unwrap(),
            &fixture.env
        )
        .is_err()
    );
    fixture.env.insert(
        "CANARY_SOURCE_ROOT".into(),
        fixture.workspace.clone().into_os_string(),
    );
    let plan = fixture.plan(Profile::Build, false);
    assert_eq!(plan.targets[0].path, fixture.workspace.join("target/debug"));
}
