use super::*;

#[test]
fn retained_duplicate_id_after_intentional_stop_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let mut owner = observer(vec![launch(root.path(), MemberId::Seed, "ready-hang")]);
    owner.stop = Some(MemberId::Seed);
    owner.after_stop = Some(launch(root.path(), MemberId::Seed, "retained-crash"));
    let report = run(&mut owner, &limits(), &Cancellation::default()).unwrap();
    assert_eq!(report.outcome, Outcome::IoFailure);
    assert!(matches!(report.failure, Some(Failure::InvalidSpec(_))));
    assert_eq!(report.members.len(), 1);
    assert!(!root.path().join("retained-crash.pid").exists());
    assert_stopped(root.path(), &["ready-hang"]);

    let restarted = tempfile::tempdir().unwrap();
    let next = MemberId::Seed.next_generation().unwrap();
    let mut owner = observer(vec![launch(root.path(), MemberId::Seed, "ready-hang")]);
    owner.stop = Some(MemberId::Seed);
    owner.after_stop = Some(launch(restarted.path(), next, "ready-hang"));
    owner.complete = true;
    let report = run(&mut owner, &limits(), &Cancellation::default()).unwrap();
    assert!(report.success());
    assert_eq!(report.members.len(), 2);
    assert_eq!(report.members[0].member, MemberId::Seed);
    assert!(matches!(
        report.members[0].disposition,
        Disposition::IntentionalStop
    ));
    assert!(report.members[0].process.cleanup.complete);
    assert_eq!(report.members[1].member, next);
    assert!(matches!(
        report.members[1].disposition,
        Disposition::SessionCleanup
    ));
    assert_stopped(restarted.path(), &["ready-hang"]);
}

#[test]
fn member_identity_rejects_invalid_names() {
    for name in [
        "",
        "contains space",
        "../seed",
        "worker/one",
        "worker\n",
        "non-ascii-\u{e9}",
        &"a".repeat(65),
    ] {
        assert!(matches!(
            MemberId::new(name, 0),
            Err(Failure::InvalidMemberName)
        ));
    }
    let member = MemberId::new("worker_4-test", 7).unwrap();
    assert_eq!(member.name(), b"worker_4-test");
    assert_eq!(member.generation(), 7);
}

#[test]
fn member_generation_overflow_is_rejected() {
    let member = MemberId::new("worker", u32::MAX).unwrap();
    assert!(matches!(
        member.next_generation(),
        Err(Failure::MemberGenerationExhausted)
    ));
}

#[test]
fn retained_next_generation_of_live_member_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let next = MemberId::Seed.next_generation().unwrap();
    let mut owner = observer(vec![
        launch(root.path(), MemberId::Seed, "ready-hang"),
        launch(root.path(), next, "retained-crash"),
    ]);
    let report = run(&mut owner, &limits(), &Cancellation::default()).unwrap();
    assert!(matches!(report.failure, Some(Failure::InvalidSpec(_))));
    assert_eq!(report.members.len(), 1);
    assert!(!root.path().join("retained-crash.pid").exists());
    assert_stopped(root.path(), &["ready-hang"]);
}

#[test]
fn retained_skipped_restart_generation_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let mut owner = observer(vec![launch(root.path(), MemberId::Seed, "ready-hang")]);
    owner.stop = Some(MemberId::Seed);
    owner.after_stop = Some(launch(
        root.path(),
        MemberId::new("seed", 2).unwrap(),
        "retained-crash",
    ));
    let report = run(&mut owner, &limits(), &Cancellation::default()).unwrap();
    assert!(matches!(report.failure, Some(Failure::InvalidSpec(_))));
    assert_eq!(report.members.len(), 1);
    assert!(matches!(
        report.members[0].disposition,
        Disposition::IntentionalStop
    ));
    assert!(!root.path().join("retained-crash.pid").exists());
}

#[test]
fn retained_seventeenth_live_member_has_typed_capacity_failure() {
    let roots: Vec<_> = (0..17).map(|_| tempfile::tempdir().unwrap()).collect();
    let mut owner = observer(
        roots
            .iter()
            .enumerate()
            .map(|(index, root)| {
                launch(
                    root.path(),
                    MemberId::new(&format!("member-{index}"), 0).unwrap(),
                    "ready-hang",
                )
            })
            .collect(),
    );
    let mut policy = limits();
    policy.execution = Duration::from_secs(30);
    let report = run(&mut owner, &policy, &Cancellation::default()).unwrap();
    assert!(matches!(
        report.failure,
        Some(Failure::RetainedMemberLimit { limit: 16 })
    ));
    assert_eq!(report.members.len(), 16);
    assert!(!roots[16].path().join("ready-hang.pid").exists());
    for root in roots.iter().take(16) {
        assert_stopped(root.path(), &["ready-hang"]);
    }
}

#[test]
fn retained_stopped_member_releases_capacity_without_losing_evidence() {
    let roots: Vec<_> = (0..17).map(|_| tempfile::tempdir().unwrap()).collect();
    let launches = roots
        .iter()
        .take(16)
        .enumerate()
        .map(|(index, root)| {
            launch(
                root.path(),
                MemberId::new(&format!("member-{index}"), 0).unwrap(),
                "ready-hang",
            )
        })
        .collect();
    let mut owner = observer(launches);
    owner.stop = Some(MemberId::new("member-0", 0).unwrap());
    owner.after_stop = Some(launch(
        roots[16].path(),
        MemberId::new("member-16", 0).unwrap(),
        "ready-hang",
    ));
    owner.complete = true;
    let mut policy = limits();
    policy.execution = Duration::from_secs(30);
    let report = run(&mut owner, &policy, &Cancellation::default()).unwrap();
    assert!(report.success());
    assert_eq!(report.members.len(), 17);
    assert!(matches!(
        report.members[0].disposition,
        Disposition::IntentionalStop
    ));
    for root in &roots {
        assert_stopped(root.path(), &["ready-hang"]);
    }
}
