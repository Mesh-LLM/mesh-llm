use super::*;

fn class(provider: &str, os: &str, label: &str) -> RunnerClass {
    RunnerClass {
        provider: provider.into(),
        os: os.into(),
        architecture: "arm64".into(),
        label: label.into(),
        group: None,
    }
}

fn snapshot() -> Snapshot {
    let mac = Role {
        runner: class("github", "macos", "macos-15"),
        pools: ["org", "mac"].into_iter().map(str::to_owned).collect(),
    };
    let linux = Role {
        runner: class("github", "linux", "ubuntu-24.04-arm"),
        pools: ["org", "linux"].into_iter().map(str::to_owned).collect(),
    };
    Snapshot {
        observed_at: 100,
        max_age: 10,
        pools: [("org", 3), ("mac", 2), ("linux", 3)]
            .into_iter()
            .map(|(name, value)| (name.into(), value))
            .collect(),
        roles: [("mac", mac), ("linux", linux)]
            .into_iter()
            .map(|(name, value)| (name.into(), value))
            .collect(),
        active: Vec::new(),
        pending: Vec::new(),
    }
}

fn main(revision: char) -> WorkKind {
    WorkKind::Main {
        repository_id: 42,
        revision: revision.to_string().repeat(40),
    }
}

fn pr(repository_id: u64, number: u64) -> WorkKind {
    WorkKind::PullRequest {
        repository_id,
        number,
    }
}

fn work(id: &str, kind: WorkKind, role: &str, workers: u64) -> Work {
    Work {
        id: id.into(),
        kind,
        role: role.into(),
        workers,
    }
}

fn pending(
    id: &str,
    kind: WorkKind,
    role: &str,
    workers: u64,
    supersedes: Option<&str>,
) -> Pending {
    Pending {
        work: work(id, kind, role, workers),
        supersedes: supersedes.map(str::to_owned),
    }
}

fn active(snapshot: &Snapshot, id: &str, kind: WorkKind, role: &str, workers: u64) -> Active {
    Active {
        work: work(id, kind, role, workers),
        observed_class: Some(snapshot.roles[role].runner.clone()),
    }
}

#[test]
fn overlapping_main_revisions_and_release_stay_in_fifo_when_a_pool_is_full() {
    let mut input = snapshot();
    input
        .active
        .push(active(&input, "main-1", main('a'), "mac", 1));
    input.pending = vec![
        pending("main-2", main('b'), "mac", 1, None),
        pending("main-3", main('c'), "mac", 1, None),
        pending(
            "release",
            WorkKind::Release { repository_id: 42 },
            "linux",
            1,
            None,
        ),
        pending(
            "warmer",
            WorkKind::Warmer { repository_id: 42 },
            "linux",
            1,
            None,
        ),
    ];
    let result = preview(&input, 105).unwrap();
    assert_eq!(result.candidates, ["main-2"]);
    assert_eq!(result.deferred, ["main-3", "release", "warmer"]);
    assert!(result.superseded_pending.is_empty());
    assert!(result.active_cancellation_candidates.is_empty());
    assert_eq!(input.pending.len(), 4, "preview never edits the queue");
}

#[test]
fn pr_supersession_keeps_running_capacity_until_completion_is_observed() {
    let mut input = snapshot();
    input
        .active
        .push(active(&input, "old-active", pr(42, 7), "mac", 2));
    input.pending = vec![
        pending("old-pending", pr(42, 7), "mac", 1, Some("old-active")),
        pending("new-pr", pr(42, 7), "mac", 1, Some("old-pending")),
        pending("main", main('b'), "mac", 1, None),
    ];
    let result = preview(&input, 105).unwrap();
    assert_eq!(result.active_cancellation_candidates, ["old-active"]);
    assert_eq!(result.superseded_pending, ["old-pending"]);
    assert!(result.candidates.is_empty());
    assert_eq!(result.deferred, ["new-pr", "main"]);

    // A later supplied snapshot must separately establish that old work ended
    // and reconcile its supersession before capacity can be considered free.
    input.active.clear();
    input.pending.remove(0);
    input.pending[0].supersedes = None;
    let after_completion = preview(&input, 105).unwrap();
    assert_eq!(after_completion.candidates, ["new-pr", "main"]);
    assert!(after_completion.deferred.is_empty());
}

#[test]
fn shared_pool_counts_both_providers_before_proposing_new_work() {
    let mut input = snapshot();
    input.pools.insert("depot-mac".into(), 2);
    input.roles.insert(
        "depot-mac".into(),
        Role {
            runner: class("depot", "macos", "depot-macos-15"),
            pools: ["org", "depot-mac"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        },
    );
    input
        .active
        .push(active(&input, "hosted", main('a'), "mac", 2));
    input
        .pending
        .push(pending("depot", main('b'), "depot-mac", 2, None));
    let result = preview(&input, 105).unwrap();
    assert!(result.candidates.is_empty());
    assert_eq!(result.deferred, ["depot"]);
    input.pending[0].work.workers = 1;
    assert_eq!(preview(&input, 105).unwrap().candidates, ["depot"]);
}

#[test]
fn incomplete_stale_or_inconsistent_observations_fail_closed() {
    let base = snapshot();
    assert_eq!(preview(&base, 99), Err(PreviewError::StaleSnapshot));
    assert_eq!(preview(&base, 111), Err(PreviewError::StaleSnapshot));

    let mut input = base.clone();
    input
        .roles
        .get_mut("mac")
        .unwrap()
        .pools
        .insert("missing".into());
    assert_eq!(preview(&input, 105), Err(PreviewError::InvalidCatalog));

    let mut input = base.clone();
    input
        .pending
        .push(pending("job", main('a'), "unknown", 1, None));
    assert_eq!(preview(&input, 105), Err(PreviewError::UnknownRole));

    let mut input = base.clone();
    input
        .active
        .push(active(&input, "job", main('a'), "mac", 1));
    input.active[0].observed_class = None;
    assert_eq!(preview(&input, 105), Err(PreviewError::MissingRunnerClass));
    input.active[0].observed_class = Some(class("depot", "macos", "depot-macos-15"));
    assert_eq!(preview(&input, 105), Err(PreviewError::RunnerClassMismatch));

    let mut input = base.clone();
    input.pools.insert("mac".into(), 0);
    input
        .active
        .push(active(&input, "job", main('a'), "mac", 1));
    assert_eq!(preview(&input, 105), Err(PreviewError::AlreadyOverCapacity));

    let mut input = base;
    input
        .pending
        .push(pending("job", main('a'), "mac", 0, None));
    assert_eq!(preview(&input, 105), Err(PreviewError::InvalidWork));
}

#[test]
fn main_other_repositories_and_unseen_or_duplicate_predecessors_cannot_be_superseded() {
    let mut input = snapshot();
    input.pending = vec![
        pending("first", main('a'), "mac", 1, None),
        pending("second", main('b'), "mac", 1, Some("first")),
    ];
    assert_eq!(preview(&input, 105), Err(PreviewError::InvalidSupersession));

    input.pending[0].work.kind = pr(42, 7);
    input.pending[1].work.kind = pr(43, 7);
    assert_eq!(preview(&input, 105), Err(PreviewError::InvalidSupersession));
    input.pending[1].work.kind = pr(42, 7);
    input.pending[1].supersedes = Some("unseen".into());
    assert_eq!(preview(&input, 105), Err(PreviewError::InvalidSupersession));
    input.pending[1].supersedes = Some("first".into());
    input
        .pending
        .push(pending("third", pr(42, 7), "mac", 1, Some("first")));
    assert_eq!(preview(&input, 105), Err(PreviewError::InvalidSupersession));

    input.pending.pop();
    input.pending[1].work.id = "first".into();
    assert_eq!(preview(&input, 105), Err(PreviewError::DuplicateWork));
}

#[test]
fn exhausted_quotas_and_integer_overflow_never_create_capacity() {
    let mut input = snapshot();
    input.pools.insert("mac".into(), 0);
    input
        .pending
        .push(pending("main", main('a'), "mac", 1, None));
    assert_eq!(preview(&input, 105).unwrap().deferred, ["main"]);

    input.pools.insert("org".into(), u64::MAX);
    input.pools.insert("mac".into(), u64::MAX);
    input
        .active
        .push(active(&input, "existing", main('b'), "mac", u64::MAX));
    assert_eq!(preview(&input, 105), Err(PreviewError::ArithmeticOverflow));
}
